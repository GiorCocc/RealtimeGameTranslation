"""
core/ocr.py — Phase 2: OCR e Text Detection

Responsabilità:
  - Riceve frame da CaptureThread via queue
  - Individua automaticamente tutte le regioni di testo (nessuna zona predefinita)
  - Filtra per confidenza e lunghezza minima
  - Text delta: invia al TranslationWorker solo le stringhe cambiate rispetto
    al frame precedente
  - Fallback EasyOCR per testi su texture / font irregolari

Classe pubblica principale:
  OCRWorker  — ThreadPoolExecutor consumer che produce TextRegion[]

Output atteso (Phase 2):
  Lista di TextRegion per ogni frame con testo cambiato.

ATTENZIONE: NON sostituire PaddleOCR con Tesseract.
  Tesseract non restituisce bounding box multi-orientate e ha prestazioni
  inferiori su font non-latini e pixel art. Vedere KNOWLEDGE_BASE.md §13.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TextRegion:
    """
    Una regione di testo rilevata dall'OCR in un frame.

    Attributes:
        bbox:       4 punti [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in pixel assoluti
                    (sistema di coordinate del frame originale pre-downscale).
        text:       Testo rilevato (lingua sorgente, non ancora tradotto).
        confidence: Score di confidenza OCR in [0, 1].
        source:     "paddle" | "easy"  — motore che ha rilevato la regione.
    """
    bbox: tuple[tuple[float, float], ...]  # 4 punti
    text: str
    confidence: float
    source: str = "paddle"

    @property
    def top_left(self) -> tuple[float, float]:
        return self.bbox[0]

    @property
    def width(self) -> float:
        return self.bbox[1][0] - self.bbox[0][0]

    @property
    def height(self) -> float:
        return self.bbox[2][1] - self.bbox[1][1]


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _bbox_from_paddle(raw_bbox: list) -> tuple[tuple[float, float], ...]:
    """Converte il formato bbox di PaddleOCR in tuple di punti."""
    return tuple((float(p[0]), float(p[1])) for p in raw_bbox)


def _scale_bbox(
    bbox: tuple[tuple[float, float], ...],
    scale_x: float,
    scale_y: float,
) -> tuple[tuple[float, float], ...]:
    """Riscala le coordinate bbox dal frame downscalato al frame originale."""
    return tuple((p[0] * scale_x, p[1] * scale_y) for p in bbox)


def _downscale_frame(
    frame: np.ndarray, target_height: int = 720
) -> tuple[np.ndarray, float, float]:
    """
    Riduce il frame se l'altezza supera target_height.

    Returns:
        (frame_ridotto, scale_x, scale_y)
        scale_x/y: fattori per riportare le coordinate bbox all'originale.
    """
    import cv2

    h, w = frame.shape[:2]
    if h <= target_height:
        return frame, 1.0, 1.0

    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale, 1.0 / scale


# ══════════════════════════════════════════════════════════════════
# OCR Worker — Phase 2 (TODO)
# ══════════════════════════════════════════════════════════════════

class OCRWorker:
    """
    Worker multi-thread per OCR.

    Consuma (frame, changed_cells) dalla `input_queue` e produce
    list[TextRegion] nella `output_queue`.

    Pipeline interna per ogni frame:
      1. Downscale opzionale a ocr_downscale_height (config)
      2. PaddleOCR su tutto il frame
      3. Filtro: scarta regioni con confidence < threshold o text < min_length
      4. Fallback EasyOCR su singole regioni con confidence bassa
      5. Text delta: confronto con frame precedente, scarta invariate
      6. Riscala bbox alle coordinate originali (se downscalato)
      7. Emette result in output_queue

    TODO Phase 2: implementare i metodi marcati con # TODO
    """

    def __init__(
        self,
        input_queue: Queue,   # riceve (frame_bgr, changed_cells) da CaptureThread
        output_queue: Queue,  # produce list[TextRegion] verso TranslationWorker
        max_workers: int = 2,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.max_workers = max_workers

        self._stop_event = threading.Event()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

        # PaddleOCR e EasyOCR vengono caricati una sola volta all'avvio
        self._paddle_ocr = None   # TODO Phase 2: inizializzare in _load_models()
        self._easy_ocr = None     # TODO Phase 2: inizializzare in _load_models()

        # Text delta: testo dell'ultimo frame per confronto
        self._prev_texts: set[str] = set()

        # Throttle: non eseguire OCR più spesso di ocr_throttle_ms
        self._last_ocr_ts: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Carica i modelli OCR e avvia il thread pool consumer."""
        logger.info("OCRWorker: caricamento modelli...")
        self._load_models()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="OCRWorker"
        )
        # Avvia il loop di consumo frame
        threading.Thread(
            target=self._consume_loop, daemon=True, name="OCRConsumer"
        ).start()
        logger.info("OCRWorker avviato (workers=%d)", self.max_workers)

    def stop(self) -> None:
        self._stop_event.set()
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("OCRWorker fermato")

    # ── Model loading ─────────────────────────────────────────────

    def _load_models(self) -> None:
        """
        TODO Phase 2 — Inizializza PaddleOCR ed EasyOCR.

        Esempio implementazione:
            from paddleocr import PaddleOCR
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=config.ocr_use_angle_cls,
                lang=config.ocr_language,
                show_log=False,
                use_gpu=False,  # TODO: rilevare CUDA disponibile
            )

            import easyocr
            self._easy_ocr = easyocr.Reader(
                [config.ocr_language],
                gpu=False,  # TODO: rilevare CUDA disponibile
                verbose=False,
            )
        """
        raise NotImplementedError("Phase 2: implementare _load_models()")

    # ── Consumer loop ─────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """Loop principale: prende frame dalla queue e schedula OCR."""
        while not self._stop_event.is_set():
            try:
                frame, cells = self.input_queue.get(timeout=0.1)
            except Exception:
                continue

            # Throttle: rispetta ocr_throttle_ms
            now = time.perf_counter()
            elapsed_ms = (now - self._last_ocr_ts) * 1000
            if elapsed_ms < config.ocr_throttle_ms:
                time.sleep((config.ocr_throttle_ms - elapsed_ms) / 1000)

            self._last_ocr_ts = time.perf_counter()
            if self._executor:
                self._executor.submit(self._process_frame, frame, cells)

    # ── OCR processing ────────────────────────────────────────────

    def _process_frame(
        self, frame: np.ndarray, changed_cells: list[bool]
    ) -> None:
        """
        TODO Phase 2 — Esegue OCR su un frame e aggiorna output_queue.

        Logica:
          1. Downscale se config.ocr_downscale
          2. Chiama _run_paddle(frame) → list[TextRegion]
          3. Per ogni regione con confidence < threshold: chiama _run_easy(crop)
          4. Filtra per lunghezza minima e confidence
          5. Text delta: confronta con self._prev_texts
          6. Se ci sono testi nuovi/cambiati: riscala bbox, emetti in output_queue
        """
        raise NotImplementedError("Phase 2: implementare _process_frame()")

    def _run_paddle(self, frame: np.ndarray) -> list[TextRegion]:
        """
        TODO Phase 2 — Esegue PaddleOCR sull'intero frame.

        Returns:
            Lista di TextRegion (con bbox in coordinate del frame passato).

        Esempio:
            result = self._paddle_ocr.ocr(frame, cls=config.ocr_use_angle_cls)
            regions = []
            for line in (result[0] or []):
                bbox_raw, (text, conf) = line
                if conf >= config.ocr_confidence_threshold:
                    regions.append(TextRegion(
                        bbox=_bbox_from_paddle(bbox_raw),
                        text=text,
                        confidence=conf,
                        source="paddle",
                    ))
            return regions
        """
        raise NotImplementedError("Phase 2: implementare _run_paddle()")

    def _run_easy(
        self, crop: np.ndarray, original_bbox: tuple
    ) -> Optional[TextRegion]:
        """
        TODO Phase 2 — Fallback EasyOCR su un crop di regione a bassa confidence.

        Args:
            crop:          Ritaglio numpy BGR della regione
            original_bbox: bbox originale da PaddleOCR (per mantenere coordinate)

        Returns:
            TextRegion con source="easy" o None se nulla rilevato.
        """
        raise NotImplementedError("Phase 2: implementare _run_easy()")

    @property
    def stats(self) -> dict:
        """TODO Phase 2: restituire statistiche (frame processati, tempo medio OCR, ecc.)"""
        return {}
