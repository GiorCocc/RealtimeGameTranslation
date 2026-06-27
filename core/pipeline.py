"""
core/pipeline.py — Phase 4: Orchestrazione della Pipeline

Responsabilità:
  - Crea e collega tutti e quattro gli stadi della pipeline:
      CaptureThread → OCRWorker → TranslationWorker → OverlayWindow
  - Gestisce le queue inter-thread
  - Espone metodi start/stop/pause/resume a main.py e al pannello Settings
  - Ponte tra thread Python e main Qt thread via QThread/signals

Threading model:
  CaptureThread    — thread daemon dedicato (produce frame)
  OCRWorker        — ThreadPoolExecutor consumer (consuma frame, produce TextRegion[])
  TranslationWorker — thread daemon (consuma TextRegion[], produce (bbox, str)[])
  OverlayWindow    — main Qt thread (riceve via Qt signal thread-safe)

TODO Phase 4: implementare dopo aver completato Phases 1-3.
"""

from __future__ import annotations

import logging
from queue import Queue
from typing import Optional

from config import config
from core.capture import CaptureThread
from core.ocr import OCRWorker
from core.translator import TranslationWorker

logger = logging.getLogger(__name__)

# Dimensioni queue inter-thread (buffer piccolo = bassa latenza)
_CAPTURE_QUEUE_SIZE = 4
_OCR_QUEUE_SIZE = 4
_TRANSLATION_QUEUE_SIZE = 8


class TranslationPipeline:
    """
    Orchestratore della pipeline completa.

    Uso:
        pipeline = TranslationPipeline(overlay_window)
        pipeline.start()
        ...
        pipeline.stop()

    TODO Phase 4:
      - Integrare con OverlayWindow tramite Qt signal/slot
      - Avviare tutti i worker in sequenza
      - Gestire errori nei sotto-thread (restart automatico)
      - Esporre statistiche aggregate
    """

    def __init__(self, overlay_window=None) -> None:
        """
        Args:
            overlay_window: istanza di OverlayWindow (Phase 4).
                            Per ora può essere None (Phase 1-3 headless).
        """
        self.overlay = overlay_window

        # ── Code inter-thread ──────────────────────────────────────
        self._capture_queue: Queue = Queue(maxsize=_CAPTURE_QUEUE_SIZE)
        self._ocr_queue: Queue = Queue(maxsize=_OCR_QUEUE_SIZE)
        self._translation_queue: Queue = Queue(maxsize=_TRANSLATION_QUEUE_SIZE)

        # ── Worker instances ──────────────────────────────────────
        self._capture: Optional[CaptureThread] = None
        self._ocr: Optional[OCRWorker] = None
        self._translator: Optional[TranslationWorker] = None

        self._running: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        TODO Phase 4 — Avvia tutti gli stadi della pipeline in ordine.

        Ordine avvio:
          1. TranslationWorker (carica modello ~secondi, avviare per primo)
          2. OCRWorker (carica PaddleOCR/EasyOCR)
          3. CaptureThread (inizia a produrre frame)

        Collegamento code:
          CaptureThread  → _capture_queue  → OCRWorker
          OCRWorker      → _ocr_queue      → TranslationWorker
          TranslationWorker → _translation_queue → OverlayWindow (via signal)
        """
        logger.info("Pipeline: avvio in corso...")

        # Phase 1 only: avvia solo la capture (OCR e Translation non ancora pronti)
        self._capture = CaptureThread(
            frame_queue=self._capture_queue,
            monitor_index=config.monitor_index,
            target_fps=config.capture_fps,
        )
        self._capture.start()

        # TODO Phase 4: aggiungere qui OCRWorker.start() e TranslationWorker.start()
        # TODO Phase 4: collegare _translation_queue → OverlayWindow.update_signal

        self._running = True
        logger.info("Pipeline avviata (Phase 1: solo capture)")

    def stop(self) -> None:
        """Ferma tutti gli stadi in ordine inverso."""
        logger.info("Pipeline: stop...")
        self._running = False

        if self._capture:
            self._capture.stop()
            self._capture.join(timeout=3)

        # TODO Phase 4: stop OCRWorker, TranslationWorker

        logger.info("Pipeline fermata")

    def pause(self) -> None:
        """Pausa la produzione di frame (es. overlay nascosto con F10)."""
        if self._capture:
            self._capture.pause()

    def resume(self) -> None:
        """Riprende dopo una pausa."""
        if self._capture:
            self._capture.resume()

    # ── Stats aggregate ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Statistiche aggregate da tutti gli stadi."""
        result: dict = {"running": self._running}
        if self._capture:
            result["capture"] = self._capture.stats
        # TODO Phase 4: aggiungere stats OCR e translation
        return result
