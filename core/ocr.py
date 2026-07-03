"""
core/ocr.py — Phase 2: OCR e Text Detection

Responsabilità:
  - Analizza l'intero frame (nessuna zona predefinita) con PaddleOCR
  - Fallback a EasyOCR sulle regioni a bassa confidenza (testo su texture,
    font irregolari, manoscritto)
  - Filtra i risultati per confidenza minima e lunghezza minima del testo
  - Text delta: non riemette nulla se il set di testi rilevati è identico
    al frame precedente (riduce il carico sul TranslationWorker)
  - Throttling: non esegue OCR più spesso di config.ocr_throttle_ms

Classi pubbliche:
  TextRegion  — dataclass immutabile che rappresenta una regione di testo
  OCRWorker   — thread consumer: frame_queue → OCR → output_queue

NON sostituire PaddleOCR con Tesseract (vedere KNOWLEDGE_BASE.md §13).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
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


# ══════════════════════════════════════════════════════════════════
# OCR Worker
# ══════════════════════════════════════════════════════════════════

class OCRWorker:
    """
    Thread consumer che legge (frame, changed_cells) da `input_queue`
    (prodotta da CaptureThread) ed emette list[TextRegion] su
    `output_queue` per il TranslationWorker.

    Uso tipico:
        capture_q = Queue(maxsize=4)
        ocr_q = Queue(maxsize=4)
        ocr = OCRWorker(capture_q, ocr_q)
        ocr.start()      # blocca finché i modelli non sono caricati
        regions = ocr_q.get()
        ocr.stop()
    """

    def __init__(self, input_queue: Queue, output_queue: Queue) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self._stop_event = threading.Event()

        # Modelli (caricati in _load_models)
        self._paddle_engine = None
        self._paddle_major: int = 2
        self._easy_reader = None  # None = fallback disattivato

        # Per la preview OpenCV in main.py
        self._latest_frame: Optional[np.ndarray] = None

        # Text delta: set di testi rilevati nell'ultimo frame emesso
        self._prev_text_set: frozenset = frozenset()

        # Stats
        self._n_processed: int = 0
        self._n_skipped_delta: int = 0
        self._n_regions: int = 0
        self._ocr_times: list[float] = []
        self._t_start: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Carica i modelli (bloccante) e avvia il thread consumer."""
        self._load_models()
        self._t_start = time.perf_counter()
        threading.Thread(
            target=self._consume_loop, daemon=True, name="OCRWorker"
        ).start()
        logger.info("OCRWorker avviato")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info(
            "OCRWorker fermato — processati=%d  regioni=%d  skip_delta=%d",
            self._n_processed, self._n_regions, self._n_skipped_delta,
        )

    # ── Model loading ─────────────────────────────────────────────

    @staticmethod
    def _detect_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _load_models(self) -> None:
        cuda_available = self._detect_cuda()
        logger.info(
            "OCRWorker: CUDA %s",
            "disponibile (GPU)" if cuda_available else "non disponibile (CPU)",
        )

        self._paddle_engine, self._paddle_major = self._init_paddle(cuda_available)
        if self._paddle_major >= 3 and config.ocr_language == "en":
            model_desc = "PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec (pin esplicito)"
        elif self._paddle_major >= 3:
            model_desc = "default libreria (nessun pin per lingue diverse da 'en')"
        else:
            model_desc = "default v2.x"
        logger.info(
            "OCRWorker: PaddleOCR caricato (API v%d, lang=%s, modello=%s, mkldnn=%s)",
            self._paddle_major, config.ocr_language, model_desc,
            "disattivato (workaround bug PaddlePaddle 3.3.x)"
            if (not cuda_available and self._paddle_mkldnn_broken())
            else "attivo",
        )

        try:
            import easyocr
            self._easy_reader = easyocr.Reader(
                [config.ocr_language], gpu=cuda_available, verbose=False
            )
            logger.info("OCRWorker: EasyOCR caricato (fallback attivo)")
        except Exception:
            logger.warning(
                "OCRWorker: EasyOCR non disponibile — fallback disattivato",
                exc_info=True,
            )
            self._easy_reader = None

    @staticmethod
    def _paddle_mkldnn_broken() -> bool:
        """
        True solo se la versione di PaddlePaddle installata è quella nota
        per la regressione oneDNN/PIR su CPU (PaddlePaddle/Paddle#77340),
        introdotta nella 3.3.0. Se non riusciamo a determinare la versione,
        assumiamo prudenzialmente che NON sia rotta (di default oneDNN
        resta attivo, che è il comportamento corretto per tutte le altre
        versioni).
        """
        try:
            import paddle
            parts = paddle.__version__.split(".")[:3]
            major, minor, patch = (int(p) for p in parts)
        except Exception:
            return False
        return (major, minor, patch) >= (3, 3, 0)

    def _init_paddle(self, use_gpu: bool) -> tuple[object, int]:
        """
        Inizializza PaddleOCR gestendo le differenze d'interfaccia tra
        le major version 2.x e 3.x (parametri rinominati in 3.x).
        """
        import paddleocr

        version = getattr(paddleocr, "__version__", "2.0.0")
        try:
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            major = 2

        if major >= 3:
            kwargs = dict(
                use_textline_orientation=config.ocr_use_angle_cls,
                # Moduli pensati per foto/scansioni di documenti (pagine
                # storte, distorsione geometrica). Per screenshot di un
                # gioco sono inutili e pesanti (UVDoc in particolare):
                # vanno disabilitati esplicitamente, altrimenti alcune
                # versioni di PaddleOCR 3.x li attivano di default.
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device="gpu" if use_gpu else "cpu",
            )
            # IMPORTANTE — leggere prima di modificare questo blocco:
            # PaddleOCR emette
            #   UserWarning: `lang` and `ocr_version` will be ignored
            #   when model names or model directories are not `None`.
            # Cioè: appena si specifica UN SOLO nome di modello esplicito
            # (es. solo text_detection_model_name), la selezione
            # automatica basata su `lang`/`ocr_version` viene disattivata
            # per L'INTERA pipeline, non solo per il modulo specificato.
            # Il risultato osservato: la detection passa al mobile
            # leggero come richiesto, ma la recognition ripiomba sul
            # default pesante (PP-OCRv6_medium_rec, ~ordini di grandezza
            # più lento del mobile) invece di restare quella scelta da
            # `lang="en"`. Per questo i due nomi vanno sempre fissati
            # insieme, mai uno solo.
            #
            # Al momento la mappatura nome-modello per lingua è
            # verificata solo per l'inglese (il caso d'uso primario).
            # Per altre lingue sorgente si lascia che `lang`/`ocr_version`
            # scelgano i modelli di default (verosimilmente più pesanti):
            # da rivedere quando si configurerà una lingua sorgente
            # diversa da "en" (vedi anche nota in memoria sui codici
            # lingua EasyOCR/PaddleOCR non sempre coincidenti).
            if config.ocr_language == "en":
                kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                kwargs["text_recognition_model_name"] = "en_PP-OCRv5_mobile_rec"
            else:
                kwargs["lang"] = config.ocr_language
                kwargs["ocr_version"] = "PP-OCRv5"
            if not use_gpu and self._paddle_mkldnn_broken():
                # PaddlePaddle 3.3.x ha una regressione nel backend oneDNN/PIR
                # su CPU: predict() solleva
                #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute
                #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
                # su alcune CPU (PaddlePaddle/Paddle#77340).
                #
                # ATTENZIONE: disabilitare oneDNN con enable_mkldnn=False su
                # PaddlePaddle 3.3.x innesca una SECONDA regressione, più
                # grave: allocazione di decine di GB di RAM / hang
                # apparentemente infinito durante predict() (vedi
                # PaddlePaddle/PaddleOCR#17955). Va quindi applicato solo
                # come ultima risorsa quando la versione installata è
                # esattamente quella difettosa; la soluzione corretta è
                # evitare del tutto quella versione (vedere log all'avvio).
                kwargs["enable_mkldnn"] = False
        else:
            kwargs = dict(
                lang=config.ocr_language,
                use_angle_cls=config.ocr_use_angle_cls,
                use_gpu=use_gpu,
                show_log=False,
            )

        try:
            engine = paddleocr.PaddleOCR(**kwargs)
        except TypeError:
            # Versione intermedia con firma diversa: ritenta con il set
            # minimo di parametri comuni a (quasi) tutte le versioni,
            # preservando comunque il workaround enable_mkldnn se presente.
            logger.warning(
                "OCRWorker: parametri PaddleOCR non riconosciuti per v%s, "
                "ritento con configurazione minima", version,
            )
            minimal: dict = {}
            for key in (
                "lang", "enable_mkldnn", "use_doc_orientation_classify",
                "use_doc_unwarping", "ocr_version",
                "text_detection_model_name", "text_recognition_model_name",
            ):
                if key in kwargs:
                    minimal[key] = kwargs[key]
            try:
                engine = paddleocr.PaddleOCR(**minimal)
            except TypeError:
                engine = paddleocr.PaddleOCR(lang=config.ocr_language)

        return engine, major

    # ── Consumer loop ─────────────────────────────────────────────

    def _consume_loop(self) -> None:
        last_run = 0.0

        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            # Scarta eventuali frame accumulati: ci interessa solo il più
            # recente (stessa filosofia "priorità alla latenza" di capture.py)
            while True:
                try:
                    item = self.input_queue.get_nowait()
                except Empty:
                    break

            frame, _changed_cells = item
            self._latest_frame = frame

            # Throttling: non eseguire OCR più spesso di ocr_throttle_ms
            now = time.perf_counter()
            elapsed_ms = (now - last_run) * 1000
            if elapsed_ms < config.ocr_throttle_ms:
                time.sleep((config.ocr_throttle_ms - elapsed_ms) / 1000)
            last_run = time.perf_counter()

            t0 = time.perf_counter()
            logger.debug("OCRWorker: elaborazione frame in corso...")
            try:
                regions = self._process_frame(frame)
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
                # Text delta: nessun cambiamento rispetto al frame precedente
                continue

            self._n_processed += 1
            self._n_regions += len(regions)

            try:
                self.output_queue.put_nowait(regions)
            except Full:
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.output_queue.put_nowait(regions)
                except Full:
                    pass

    # ── Frame processing ─────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> Optional[list[TextRegion]]:
        """
        Pipeline completa su un singolo frame:
          downscale (opzionale) → PaddleOCR → fallback EasyOCR sulle
          regioni a bassa confidenza → filtro → rescale bbox → text delta

        Returns:
            list[TextRegion] se il set di testi è cambiato rispetto al
            frame precedente, altrimenti None (= skip, nulla da emettere).
        """
        ocr_frame, scale = self._maybe_downscale(frame)
        raw = self._run_paddle(ocr_frame)
        logger.debug("OCRWorker: PaddleOCR ha rilevato %d elemento/i grezzo/i", len(raw))

        regions: list[TextRegion] = []
        for bbox_ds, text, conf in raw:
            bbox = self._rescale_bbox(bbox_ds, scale)
            text = text.strip()
            source = "paddle"

            if conf < config.ocr_confidence_threshold:
                if self._easy_reader is None or conf < _EASYOCR_FALLBACK_MIN_CONF:
                    continue
                fallback = self._run_easyocr_fallback(frame, bbox)
                if fallback is None:
                    continue
                text, conf = fallback
                text = text.strip()
                source = "easy"

            if conf < config.ocr_confidence_threshold:
                continue
            if len(text) < config.ocr_min_text_length:
                continue

            regions.append(
                TextRegion(bbox=bbox, text=text, confidence=conf, source=source)
            )

        pre_merge_count = len(regions)
        if config.ocr_merge_adjacent_boxes:
            regions = merge_adjacent_regions(
                regions,
                max_gap_em=config.ocr_merge_max_gap_em,
                max_vertical_offset_em=config.ocr_merge_max_vertical_offset_em,
            )
            logger.debug(
                "OCRWorker: %d regione/i dopo il merge box adiacenti (da %d)",
                len(regions), pre_merge_count,
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

    @staticmethod
    def _maybe_downscale(frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Ridimensiona il frame a ocr_downscale_height se più alto. Restituisce (frame, scale)."""
        if not config.ocr_downscale:
            return frame, 1.0

        h, w = frame.shape[:2]
        if h <= config.ocr_downscale_height:
            return frame, 1.0

        import cv2
        scale = config.ocr_downscale_height / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(
            frame, (new_w, config.ocr_downscale_height), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    @staticmethod
    def _rescale_bbox(bbox: tuple, scale: float) -> tuple:
        """Riporta una bbox dalle coordinate (eventualmente downscalate) a quelle originali."""
        if scale == 1.0:
            return tuple((float(p[0]), float(p[1])) for p in bbox)
        return tuple((float(p[0]) / scale, float(p[1]) / scale) for p in bbox)

    # ── PaddleOCR ─────────────────────────────────────────────────

    def _run_paddle(self, frame: np.ndarray) -> list[tuple]:
        """
        Esegue PaddleOCR su `frame` e normalizza l'output a:
            list[(bbox_4pts, text, confidence)]
        gestendo sia l'API 2.x (.ocr) sia la 3.x (.predict).
        """
        if self._paddle_major >= 3:
            return self._run_paddle_v3(frame)
        return self._run_paddle_v2(frame)

    def _run_paddle_v2(self, frame: np.ndarray) -> list[tuple]:
        out: list[tuple] = []
        try:
            raw = self._paddle_engine.ocr(frame, cls=config.ocr_use_angle_cls)
        except TypeError:
            raw = self._paddle_engine.ocr(frame)

        if not raw:
            return out

        # In 2.x raw è list[list[ [box, (text, score)] ]] — una lista per pagina.
        page = raw[0] if isinstance(raw[0], list) or raw[0] is None else raw
        for line in (page or []):
            try:
                box, (text, score) = line
                bbox = tuple((float(p[0]), float(p[1])) for p in box)
                out.append((bbox, text, float(score)))
            except (ValueError, TypeError):
                logger.debug("OCRWorker: riga PaddleOCR v2 non riconosciuta: %r", line)
        return out

    def _run_paddle_v3(self, frame: np.ndarray) -> list[tuple]:
        out: list[tuple] = []
        try:
            results = self._paddle_engine.predict(frame)
        except AttributeError:
            results = self._paddle_engine.ocr(frame)

        for res in results or []:
            try:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                polys = res.get("rec_polys", res.get("rec_boxes", []))
            except AttributeError:
                logger.debug("OCRWorker: risultato PaddleOCR v3 non riconosciuto: %r", res)
                continue

            for text, score, poly in zip(texts, scores, polys):
                bbox = tuple((float(p[0]), float(p[1])) for p in poly)
                out.append((bbox, text, float(score)))
        return out

    # ── EasyOCR fallback ──────────────────────────────────────────

    def _run_easyocr_fallback(
        self, frame: np.ndarray, bbox: tuple
    ) -> Optional[tuple[str, float]]:
        """
        Ritenta il riconoscimento su un crop della bbox PaddleOCR (in
        coordinate originali, non downscalate) usando EasyOCR. La
        posizione della bbox originale viene mantenuta — non si usa
        quella restituita da EasyOCR.
        """
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
            logger.debug("OCRWorker: EasyOCR fallback fallito", exc_info=True)
            return None

        if not results:
            return None

        # Più righe nel crop → le uniamo; la confidenza è la minima
        # (approccio prudente: il pezzo più debole determina la confidenza totale).
        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        return " ".join(texts), min(confs)

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        avg_ms = sum(self._ocr_times) / len(self._ocr_times) if self._ocr_times else 0.0
        return {
            "frames_processed": self._n_processed,
            "avg_ocr_ms": round(avg_ms, 1),
            "regions_emitted": self._n_regions,
            "frames_skipped_delta": self._n_skipped_delta,
            "easy_available": self._easy_reader is not None,
        }