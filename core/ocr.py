"""
core/ocr.py — Phase 2 + Phase 5: OCR e Text Detection (ottimizzato)

Responsabilità:
  - Analizza l'intero frame (nessuna zona predefinita) con PaddleOCR
  - Fallback a EasyOCR sulle regioni a bassa confidenza (testo su texture,
    font irregolari, manoscritto)
  - Filtra i risultati per confidenza minima e lunghezza minima del testo
  - Text delta: non riemette nulla se il set di testi rilevati è identico
    al frame precedente (riduce il carico sul TranslationWorker)
  - Throttling: non esegue OCR più spesso di config.ocr_throttle_ms

Ottimizzazioni Phase 5:
  - OCR incrementale per zone: processa solo le celle della griglia cambiate,
    riutilizzando i risultati OCR delle zone stabili (60-80% meno carico)
  - Lazy loading EasyOCR: caricato solo al primo fallback effettivo (~500MB
    di RAM risparmiati all'avvio)
  - Adaptive downscale: risoluzione OCR scelta in base a VRAM disponibile
  - Thread priority BELOW_NORMAL su Windows (non compete con il gioco)
  - GPU detection centralizzata via core.gpu_utils

Classi pubbliche:
  TextRegion  — dataclass immutabile che rappresenta una regione di testo
  OCRWorker   — thread consumer: frame_queue → OCR → output_queue

NON sostituire PaddleOCR con Tesseract (vedere KNOWLEDGE_BASE.md §13).
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import time
import queue
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Sotto questa soglia di confidenza PaddleOCR, non vale nemmeno la pena
# tentare il fallback EasyOCR: il rilevamento è quasi certamente spazzatura
# (pattern HUD, icone, rumore) e si sprecherebbe solo tempo CPU.
_EASYOCR_FALLBACK_MIN_CONF = 0.30

# Quante misurazioni di tempo OCR mantenere per calcolare la media mobile
# esposta in .stats (avg_ocr_ms).
_STATS_WINDOW = 50


# ══════════════════════════════════════════════════════════════════
# TextRegion
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TextRegion:
    """
    Una singola regione di testo rilevata sullo schermo.

    Attributes:
        bbox: 4 punti (x, y) in coordinate pixel assolute del frame
              originale, in ordine orario a partire dall'angolo in alto
              a sinistra: ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
        text: testo riconosciuto (già .strip()-ato)
        confidence: punteggio di confidenza 0.0-1.0
        source: "paddle", "easy", oppure "merged" (regione risultante
                dall'unione di più box adiacenti di sorgenti diverse —
                vedere merge_adjacent_regions())
    """

    bbox: tuple
    text: str
    confidence: float
    source: str

    @property
    def width(self) -> float:
        xs = [p[0] for p in self.bbox]
        return max(xs) - min(xs)

    @property
    def height(self) -> float:
        ys = [p[1] for p in self.bbox]
        return max(ys) - min(ys)

    @property
    def center(self) -> tuple[float, float]:
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (sum(xs) / 4, sum(ys) / 4)


# ══════════════════════════════════════════════════════════════════
# Bounding Box Merging
# ══════════════════════════════════════════════════════════════════
#
# PaddleOCR isola spesso singole parole in bounding box separate (es.
# "Press" e "Start" come due regioni distinte invece di "Press Start"),
# perché la sua detection lavora a livello di parola/token, non di frase
# o riga completa. Passate così com'è a MarianMT, ogni parola viene
# tradotta SENZA il contesto delle altre, producendo traduzioni prive di
# senso o palesemente sbagliate (es. "Press" tradotto isolatamente può
# diventare "Stampa" invece di "Premi").
#
# La soluzione qui implementata è puramente geometrica (non NLP): unisce
# bounding box vicine sulla stessa riga in un'unica regione, sul
# presupposto ragionevole che parole scritte vicine sulla stessa riga in
# un'interfaccia di gioco appartengano quasi sempre alla stessa frase o
# elemento UI (es. "Press Start", "New Game", "HP: 100/100").

def _region_bounds(region: TextRegion) -> tuple[float, float, float, float]:
    """Restituisce (left, top, right, bottom) della bbox della regione."""
    xs = [p[0] for p in region.bbox]
    ys = [p[1] for p in region.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_region_group(group: list[TextRegion]) -> TextRegion:
    """
    Unisce un gruppo di TextRegion (già note per essere adiacenti sulla
    stessa riga) in un'unica TextRegion.

    - bbox: inviluppo rettangolare (assiale) di tutte le box del gruppo
    - text: concatenazione dei testi da sinistra a destra, separati da
      uno spazio
    - confidence: minimo delle confidenze del gruppo (approccio
      prudente — la regione più debole determina la confidenza
      dell'insieme unito)
    - source: quella comune se tutte le regioni condividono la stessa
      sorgente, altrimenti "merged"
    """
    group_sorted = sorted(group, key=lambda r: _region_bounds(r)[0])
    bounds = [_region_bounds(r) for r in group_sorted]
    left = min(b[0] for b in bounds)
    top = min(b[1] for b in bounds)
    right = max(b[2] for b in bounds)
    bottom = max(b[3] for b in bounds)
    bbox = ((left, top), (right, top), (right, bottom), (left, bottom))

    text = " ".join(r.text for r in group_sorted)
    confidence = min(r.confidence for r in group_sorted)
    sources = {r.source for r in group_sorted}
    source = sources.pop() if len(sources) == 1 else "merged"

    return TextRegion(bbox=bbox, text=text, confidence=confidence, source=source)


def merge_adjacent_regions(
    regions: list[TextRegion],
    max_gap_em: float,
    max_vertical_offset_em: float,
) -> list[TextRegion]:
    """
    Unisce bounding box adiacenti sulla stessa riga in un'unica regione.

    Euristica in due passi:
      1. Raggruppamento per riga: due regioni sono considerate sulla
         stessa riga se il loro centro verticale differisce per meno di
         `max_vertical_offset_em` volte l'altezza media delle due box.
      2. Unione orizzontale: all'interno di ogni riga, le regioni
         vengono ordinate da sinistra a destra e unite finché il gap
         orizzontale (bordo destro della precedente → bordo sinistro
         della successiva) resta sotto `max_gap_em` volte l'altezza
         media delle due box confrontate.

    Le soglie sono relative all'altezza del testo ("em") invece che in
    pixel assoluti, per restare valide indipendentemente dalla
    risoluzione dello schermo o dal downscale applicato prima dell'OCR.

    Deliberatamente NON unisce:
      - regioni su righe diverse (titoli e paragrafi restano separati)
      - regioni sulla stessa riga ma troppo distanti orizzontalmente
        (evita di fondere elementi UI indipendenti che capitano sulla
        stessa riga, es. "HP: 100" e "MP: 50" in un HUD)

    Args:
        regions: TextRegion già filtrate (confidenza/lunghezza minima)
        max_gap_em: gap orizzontale massimo, in multipli dell'altezza
            media delle due box, per considerarle parte della stessa
            frase (vedere config.ocr_merge_max_gap_em)
        max_vertical_offset_em: tolleranza verticale, in multipli
            dell'altezza media delle due box, per considerare due
            regioni sulla stessa riga (vedere
            config.ocr_merge_max_vertical_offset_em)

    Returns:
        Nuova lista di TextRegion, con le regioni adiacenti unite.
        Le regioni che non vengono unite a nessun'altra sono restituite
        invariate (stesso oggetto).
    """
    if not regions:
        return []

    # Ordina per (centro_y, sinistra): raggruppa naturalmente le
    # regioni della stessa riga in sequenze contigue, permettendo un
    # singolo passaggio sequenziale per il raggruppamento in righe.
    sorted_regions = sorted(
        regions, key=lambda r: (r.center[1], _region_bounds(r)[0])
    )

    rows: list[list[TextRegion]] = []
    for region in sorted_regions:
        region_height = region.height or 1.0
        placed = False
        for row in rows:
            ref = row[-1]
            ref_height = ref.height or 1.0
            avg_height = (region_height + ref_height) / 2
            if abs(region.center[1] - ref.center[1]) <= max_vertical_offset_em * avg_height:
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    merged: list[TextRegion] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: _region_bounds(r)[0])
        current_group: list[TextRegion] = [row_sorted[0]]

        for region in row_sorted[1:]:
            prev = current_group[-1]
            _, _, prev_right, _ = _region_bounds(prev)
            left, _, _, _ = _region_bounds(region)
            gap = left - prev_right
            avg_height = ((prev.height or 1.0) + (region.height or 1.0)) / 2

            if gap <= max_gap_em * avg_height:
                current_group.append(region)
            else:
                merged.append(
                    current_group[0] if len(current_group) == 1
                    else _merge_region_group(current_group)
                )
                current_group = [region]

        merged.append(
            current_group[0] if len(current_group) == 1
            else _merge_region_group(current_group)
        )

    return merged


def merge_vertical_blocks(
    regions: list[TextRegion],
    max_line_gap_em: float,
    min_horizontal_overlap_ratio: float = 0.25,
) -> list[TextRegion]:
    """
    Unisce regioni impilate verticalmente in un unico blocco di testo.

    Mentre `merge_adjacent_regions` unisce parole sulla stessa riga
    orizzontale, questa funzione unisce righe di testo consecutive che
    appartengono allo stesso paragrafo/dialog box (es. visual novel,
    sottotitoli, descrizioni multi-riga).

    Criterio di merge:
      1. Le due regioni si sovrappongono orizzontalmente per almeno
         `min_horizontal_overlap_ratio` rispetto alla regione più stretta.
      2. Il gap verticale (fondo riga sopra → testa riga sotto) è inferiore
         a `max_line_gap_em` volte l'altezza media delle due righe.

    Il testo viene concatenato in ordine dall'alto verso il basso,
    separato da uno spazio (il tokenizer MarianMT gestisce bene i testi
    multi-frase come un'unica stringa).

    Args:
        regions: lista di TextRegion già post-merge orizzontale
        max_line_gap_em: gap verticale massimo (in em) tra righe adiacenti
        min_horizontal_overlap_ratio: sovrapposizione orizzontale minima
            perché due righe vengano considerate parte dello stesso blocco

    Returns:
        Nuova lista di TextRegion con blocchi multi-riga unificati.
    """
    if len(regions) <= 1:
        return regions

    # Ordina dall'alto verso il basso
    sorted_r = sorted(regions, key=lambda r: _region_bounds(r)[1])

    groups: list[list[TextRegion]] = [[sorted_r[0]]]

    for region in sorted_r[1:]:
        merged_into_group = False
        r_left, r_top, r_right, r_bottom = _region_bounds(region)
        r_width = max(r_right - r_left, 1.0)

        for group in reversed(groups):
            # Confronta con l'ultimo elemento del gruppo
            g_last = group[-1]
            g_left, g_top, g_right, g_bottom = _region_bounds(g_last)
            g_width = max(g_right - g_left, 1.0)

            # Gap verticale tra fondo del gruppo e cima della regione
            vert_gap = r_top - g_bottom
            avg_height = ((g_bottom - g_top) + (r_bottom - r_top)) / 2 or 1.0

            if vert_gap > max_line_gap_em * avg_height or vert_gap < -avg_height:
                # Troppo lontano o sovrapposto in modo anomalo: no merge
                continue

            # Controllo altezza (font size) simile per evitare di fondere dialoghi con pulsanti menu o tag nomi
            g_height = g_bottom - g_top
            r_height = r_bottom - r_top
            h_ratio = g_height / r_height if r_height > 0 else 0
            if h_ratio < 0.7 or h_ratio > 1.4:
                continue

            # Sovrapposizione orizzontale
            overlap_left = max(g_left, r_left)
            overlap_right = min(g_right, r_right)
            overlap = max(0.0, overlap_right - overlap_left)
            min_w = min(g_width, r_width)
            if overlap / min_w < min_horizontal_overlap_ratio:
                # Non abbastanza sovrapposti in X: elementi UI diversi
                continue

            group.append(region)
            merged_into_group = True
            break

        if not merged_into_group:
            groups.append([region])

    result: list[TextRegion] = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            # Ordina dall'alto verso il basso prima di concatenare il testo
            group_sorted = sorted(group, key=lambda r: _region_bounds(r)[1])
            result.append(_merge_region_group(group_sorted))

    return result


# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# OCR Engine
# ══════════════════════════════════════════════════════════════════

class RapidOCREngine:
    """
    Gestisce esclusivamente i modelli OCR (RapidOCR per il caso generale
    ed EasyOCR come fallback per font difficili).
    """

    def __init__(self):
        self._rapidocr_engine = None
        self._easy_reader = None
        self.easy_load_attempted: bool = False
        self.device: str = "cpu"

    @property
    def is_easyocr_loaded(self) -> bool:
        return self._easy_reader is not None

    def _resolve_gpu_device(self) -> str:
        try:
            from core.gpu_utils import detect_device, get_optimal_device
            if config.ocr_gpu_device == "cuda":
                info = detect_device()
                return "cuda" if info.backend == "cuda" else "cpu"
            elif config.ocr_gpu_device == "cpu":
                return "cpu"
            else:
                return get_optimal_device(vram_required_mb=300)
        except ImportError:
            return "cpu"

    def load_models(self) -> None:
        self.device = self._resolve_gpu_device()
        logger.info(
            "RapidOCREngine: device OCR → %s",
            self.device.upper()
        )

        self._rapidocr_engine = self._init_rapidocr(self.device)
        logger.info(
            "RapidOCREngine: RapidOCR (ONNX) caricato (lang=%s, backend=%s)",
            config.ocr_language, self.device
        )

        if config.ocr_lazy_load_easyocr:
            logger.info(
                "RapidOCREngine: EasyOCR in modalità lazy-load (sarà caricato al primo fallback)"
            )
            self._easy_reader = None
            self.easy_load_attempted = False
        else:
            self.load_easyocr()

    def load_easyocr(self) -> None:
        if self.easy_load_attempted:
            return
        self.easy_load_attempted = True
        try:
            import easyocr
            self._easy_reader = easyocr.Reader(
                [config.ocr_language], gpu=(self.device == "cuda"), verbose=False
            )
            logger.info("RapidOCREngine: EasyOCR caricato (fallback attivo)")
        except Exception:
            logger.warning(
                "RapidOCREngine: EasyOCR non disponibile — fallback disattivato",
                exc_info=True,
            )
            self._easy_reader = None

    def unload_models(self) -> None:
        self._rapidocr_engine = None
        self._easy_reader = None
        self.easy_load_attempted = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("RapidOCREngine: modelli scaricati dalla VRAM")

    def _init_rapidocr(self, device: str):
        from rapidocr_onnxruntime import RapidOCR
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "directml":
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        return RapidOCR(
            det_options={'providers': providers},
            rec_options={'providers': providers},
            cls_options={'providers': providers}
        )

    def run_rapidocr(self, frame: np.ndarray) -> list[tuple]:
        if self._rapidocr_engine is None:
            return []
        try:
            res, elapse = self._rapidocr_engine(frame)
            if not res:
                return []
            out = []
            for bbox, text, conf in res:
                out.append((bbox, text, float(conf)))
            return out
        except Exception:
            logger.exception("RapidOCREngine: errore durante run_rapidocr")
            return []

    def run_easyocr_fallback(
        self, frame: np.ndarray, bbox: tuple
    ) -> Optional[tuple[str, float]]:
        import cv2

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        h, w = frame.shape[:2]

        pad = 4
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            results = self._easy_reader.readtext(crop_rgb)
        except Exception:
            logger.debug("RapidOCREngine: EasyOCR fallback fallito", exc_info=True)
            return None

        if not results:
            return None

        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        return " ".join(texts), min(confs)

    def maybe_downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if not config.ocr_downscale:
            return frame, 1.0

        target_h = config.ocr_downscale_height

        # Phase 5: adaptive downscale basato su VRAM disponibile
        try:
            from core.gpu_utils import detect_device
            info = detect_device()
            if info.backend == "cuda" and info.vram_free_mb >= 6000:
                target_h = max(target_h, 1080)
        except ImportError:
            pass

        h, w = frame.shape[:2]
        if h <= target_h:
            return frame, 1.0

        import cv2
        scale = target_h / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(
            frame, (new_w, target_h), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    @staticmethod
    def rescale_bbox(bbox: tuple, scale: float) -> tuple:
        if scale == 1.0:
            return tuple((float(p[0]), float(p[1])) for p in bbox)
        return tuple((float(p[0]) / scale, float(p[1]) / scale) for p in bbox)


# ══════════════════════════════════════════════════════════════════
# OCR Worker
# ══════════════════════════════════════════════════════════════════

class OCRWorker:
    """
    Processo consumer che legge (frame, changed_cells) da `input_queue`
    (prodotta da CaptureWorker) ed emette list[TextRegion] su
    `output_queue` per il TranslationWorker.
    """

    def __init__(self, input_queue: Queue, output_queue: Queue) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self._stop_event = multiprocessing.Event()
        self._pause_event = multiprocessing.Event()
        self._model_unloaded: bool = False

        # Cache risultati OCR per zona della griglia.
        self._zone_cache: dict[int, list[TextRegion]] = {}

        # Conteggio ottimizzazioni zone
        self._n_zones_cached: int = 0
        self._n_zones_processed: int = 0

        # Stats
        self._n_processed: int = 0
        self._n_skipped_delta: int = 0
        self._n_regions: int = 0
        self._ocr_times: list[float] = []
        self._t_start: float = 0.0

        # Per la preview OpenCV
        self.preview_queue = multiprocessing.Queue(maxsize=4)
        self._stats_queue = multiprocessing.Queue(maxsize=2)
        self._last_stats_push: float = 0.0

        self._cached_stats = {
            "frames_processed": 0,
            "avg_ocr_ms": 0.0,
            "regions_emitted": 0,
            "frames_skipped_delta": 0,
            "easy_available": False,
            "easy_lazy": config.ocr_lazy_load_easyocr,
            "zones_processed": 0,
            "zones_cached": 0,
            "zone_cache_rate_pct": 0.0,
        }

        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=self._run_process, daemon=True, name="OCRWorker"
        )
        self._process.start()
        logger.info("OCRWorker avviato")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("OCRWorker arrestato")

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._process is not None:
            self._process.join(timeout)

    @property
    def stats(self) -> dict:
        while not self._stats_queue.empty():
            try:
                self._cached_stats = self._stats_queue.get_nowait()
            except queue.Empty:
                break
        return self._cached_stats

    def _run_process(self) -> None:
        self._set_process_priority_below_normal()
        
        # Inizializza il motore OCR nello spazio di indirizzamento del subprocesso
        self.engine = RapidOCREngine()
        self.engine.load_models()

        self._t_start = time.perf_counter()
        self._last_stats_push = self._t_start
        self._prev_text_set = frozenset()

        self._consume_loop()

    @staticmethod
    def _set_process_priority_below_normal() -> None:
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00004000
            )
            logger.debug("OCRWorker: priorità processo impostata a BELOW_NORMAL")
        except Exception:
            logger.debug(
                "OCRWorker: impossibile impostare priorità processo",
                exc_info=True,
            )

    def _consume_loop(self) -> None:
        last_run = 0.0

        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                if self._pause_event.is_set():
                    if not self._model_unloaded:
                        self.engine.unload_models()
                        self._model_unloaded = True
                else:
                    if self._model_unloaded:
                        self.engine.load_models()
                        self._model_unloaded = False
                continue

            while True:
                try:
                    item = self.input_queue.get_nowait()
                except queue.Empty:
                    break

            if len(item) == 3:
                frame, changed_cells, capture_time = item
            else:
                frame, changed_cells = item
                capture_time = time.perf_counter()

            if config.debug_capture_preview:
                try:
                    self.preview_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            n_cells = len(changed_cells)
            n_changed = sum(changed_cells)
            is_scene_change = (
                n_cells > 0
                and (n_changed / n_cells) >= config.ocr_scene_change_threshold
            )
            if is_scene_change:
                if self._prev_text_set:
                    logger.debug(
                        "OCRWorker: scene change rilevato (%d/%d celle cambiate) — reset text-delta e zone cache",
                        n_changed, n_cells,
                    )
                    self._prev_text_set = frozenset()
                self._zone_cache.clear()

            now = time.perf_counter()
            elapsed_ms = (now - last_run) * 1000
            if elapsed_ms < config.ocr_throttle_ms:
                time.sleep((config.ocr_throttle_ms - elapsed_ms) / 1000)
            last_run = time.perf_counter()

            t0 = time.perf_counter()
            logger.debug("OCRWorker: elaborazione frame in corso...")
            try:
                regions = self._process_frame_incremental(
                    frame, changed_cells, is_scene_change
                )
            except Exception:
                logger.exception("OCRWorker: errore durante _process_frame")
                continue
            dt_ms = (time.perf_counter() - t0) * 1000
            if dt_ms > 2000:
                logger.warning(
                    "OCRWorker: frame elaborato in %.0f ms (insolitamente lento)",
                    dt_ms,
                )

            self._ocr_times.append(dt_ms)
            if len(self._ocr_times) > _STATS_WINDOW:
                self._ocr_times.pop(0)

            if regions is None:
                continue

            self._n_processed += 1
            self._n_regions += len(regions)
            
            logger.info("OCRWorker: trovate %d regioni di testo nel frame", len(regions))

            payload = (regions, capture_time)
            try:
                self.output_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.output_queue.put_nowait(payload)
                except queue.Full:
                    pass

            if now - self._last_stats_push >= 1.0:
                try:
                    self._stats_queue.put_nowait(self._compute_stats())
                except queue.Full:
                    pass
                self._last_stats_push = now

    def _process_frame_incremental(
        self,
        frame: np.ndarray,
        changed_cells: list[bool],
        is_scene_change: bool,
    ) -> Optional[list[TextRegion]]:
        if is_scene_change or not self._zone_cache:
            return self._process_frame_full(frame, changed_cells)

        n_changed = sum(changed_cells)
        n_total = len(changed_cells)

        if n_changed > n_total * 0.5:
            return self._process_frame_full(frame, changed_cells)

        h, w = frame.shape[:2]
        rows = config.frame_diff_grid_rows
        cols = config.frame_diff_grid_cols
        cell_h = h // rows
        cell_w = w // cols

        target_h = config.ocr_downscale_height
        try:
            from core.gpu_utils import detect_device
            info = detect_device()
            if info.backend == "cuda" and info.vram_free_mb >= 6000:
                target_h = max(target_h, 1080)
        except ImportError:
            pass
        scale = target_h / h if config.ocr_downscale and h > target_h else 1.0

        for idx, changed in enumerate(changed_cells):
            if not changed:
                self._n_zones_cached += 1
                continue

            self._n_zones_processed += 1
            r_idx = idx // cols
            c_idx = idx % cols

            y1 = r_idx * cell_h
            x1 = c_idx * cell_w
            y2 = y1 + cell_h if r_idx < rows - 1 else h
            x2 = x1 + cell_w if c_idx < cols - 1 else w

            pad_y = max(1, int(cell_h * 0.1))
            pad_x = max(1, int(cell_w * 0.1))
            y1_pad = max(0, y1 - pad_y)
            x1_pad = max(0, x1 - pad_x)
            y2_pad = min(h, y2 + pad_y)
            x2_pad = min(w, x2 + pad_x)

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                self._zone_cache[idx] = []
                continue

            if scale != 1.0:
                import cv2
                crop_h, crop_w = crop.shape[:2]
                new_w = max(1, int(crop_w * scale))
                new_h = max(1, int(crop_h * scale))
                crop_scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                crop_scaled = crop

            raw = self.engine.run_rapidocr(crop_scaled)
            if not raw and not self.engine.is_easyocr_loaded:
                if config.ocr_lazy_load_easyocr and not self.engine.easy_load_attempted:
                    self.engine.load_easyocr()
            
            zone_regions: list[TextRegion] = []
            
            def _add_region(bbox_local, text, conf, source):
                bbox_local_rescaled = self.engine.rescale_bbox(bbox_local, scale)
                bbox_global = tuple(
                    (float(p[0]) + x1_pad, float(p[1]) + y1_pad)
                    for p in bbox_local_rescaled
                )
                if conf < config.ocr_confidence_threshold:
                    return
                if len(text.strip()) < config.ocr_min_text_length:
                    return
                zone_regions.append(
                    TextRegion(bbox=bbox_global, text=text.strip(), confidence=conf, source=source)
                )

            if raw:
                for bbox_local, text, conf in raw:
                    _add_region(bbox_local, text, conf, "rapidocr")
            elif self.engine.is_easyocr_loaded:
                full_bbox = ((float(x1_pad), float(y1_pad)), (float(x2_pad), float(y1_pad)), 
                             (float(x2_pad), float(y2_pad)), (float(x1_pad), float(y2_pad)))
                res = self.engine.run_easyocr_fallback(frame, full_bbox)
                if res:
                    _add_region(full_bbox, res[0], res[1], "easy")

            self._zone_cache[idx] = zone_regions

        all_regions: list[TextRegion] = []
        for idx in range(n_total):
            if idx in self._zone_cache:
                all_regions.extend(self._zone_cache[idx])

        return self._apply_merge_and_delta(all_regions)

    def _process_frame_full(
        self,
        frame: np.ndarray,
        changed_cells: list[bool],
    ) -> Optional[list[TextRegion]]:
        ocr_frame, scale = self.engine.maybe_downscale(frame)
        raw = self.engine.run_rapidocr(ocr_frame)
        
        if not raw and not self.engine.is_easyocr_loaded:
            if config.ocr_lazy_load_easyocr and not self.engine.easy_load_attempted:
                self.engine.load_easyocr()
        
        regions: list[TextRegion] = []
        if raw:
            for bbox_ds, text, conf in raw:
                bbox = self.engine.rescale_bbox(bbox_ds, scale)
                if conf < config.ocr_confidence_threshold or len(text.strip()) < config.ocr_min_text_length:
                    continue
                regions.append(
                    TextRegion(bbox=bbox, text=text.strip(), confidence=conf, source="rapidocr")
                )
        
        self._populate_zone_cache(frame.shape[:2], regions, changed_cells)

        return self._apply_merge_and_delta(regions)

    def _populate_zone_cache(
        self,
        frame_shape: tuple[int, int],
        regions: list[TextRegion],
        changed_cells: list[bool],
    ) -> None:
        h, w = frame_shape
        rows = config.frame_diff_grid_rows
        cols = config.frame_diff_grid_cols
        cell_h = h // rows
        cell_w = w // cols
        n_total = rows * cols

        for idx, changed in enumerate(changed_cells):
            if changed:
                self._zone_cache[idx] = []

        for region in regions:
            cx, cy = region.center
            c_idx = min(int(cx / cell_w), cols - 1) if cell_w > 0 else 0
            r_idx = min(int(cy / cell_h), rows - 1) if cell_h > 0 else 0
            idx = r_idx * cols + c_idx
            if 0 <= idx < n_total:
                if idx not in self._zone_cache:
                    self._zone_cache[idx] = []
                self._zone_cache[idx].append(region)

    def _apply_merge_and_delta(
        self, regions: list[TextRegion]
    ) -> Optional[list[TextRegion]]:
        pre_merge_count = len(regions)
        if config.ocr_merge_adjacent_boxes:
            regions = merge_adjacent_regions(
                regions,
                max_gap_em=config.ocr_merge_max_gap_em,
                max_vertical_offset_em=config.ocr_merge_max_vertical_offset_em,
            )
            logger.debug(
                "OCRWorker: %d regione/i dopo il merge orizzontale (da %d)",
                len(regions), pre_merge_count,
            )

        if config.ocr_merge_vertical_blocks:
            pre_vert = len(regions)
            regions = merge_vertical_blocks(
                regions,
                max_line_gap_em=config.ocr_merge_max_line_gap_em,
            )
            if len(regions) < pre_vert:
                logger.debug(
                    "OCRWorker: %d regione/i dopo il merge verticale (da %d) — dialog box multi-riga unificati",
                    len(regions), pre_vert,
                )

        current_text_set = frozenset(r.text for r in regions)
        logger.debug(
            "OCRWorker: %d regione/i dopo il filtro: %s",
            len(regions),
            [r.text[:30] for r in regions],
        )
        if current_text_set == self._prev_text_set:
            self._n_skipped_delta += 1
            logger.debug("OCRWorker: nessuna variazione rispetto al frame precedente — skip")
            return None

        self._prev_text_set = current_text_set
        return regions

    def _compute_stats(self) -> dict:
        avg_ms = sum(self._ocr_times) / len(self._ocr_times) if self._ocr_times else 0.0
        total_zones = self._n_zones_cached + self._n_zones_processed
        zone_cache_rate = (
            round(self._n_zones_cached / max(1, total_zones) * 100, 1)
            if total_zones > 0 else 0.0
        )
        return {
            "frames_processed": self._n_processed,
            "avg_ocr_ms": round(avg_ms, 1),
            "regions_emitted": self._n_regions,
            "frames_skipped_delta": self._n_skipped_delta,
            "easy_available": self.engine.is_easyocr_loaded if hasattr(self, "engine") else False,
            "easy_lazy": config.ocr_lazy_load_easyocr and not self.engine.easy_load_attempted,
            "zones_processed": self._n_zones_processed,
            "zones_cached": self._n_zones_cached,
            "zone_cache_rate_pct": zone_cache_rate,
        }