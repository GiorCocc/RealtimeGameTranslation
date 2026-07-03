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
  OCRWorker        — thread daemon consumer (consuma frame, produce TextRegion[])
  TranslationWorker — thread daemon (consuma TextRegion[], produce (bbox, str)[])
  OverlayWindow    — main Qt thread (riceve via Qt signal thread-safe)

Stato:
  - Capture (Phase 1), OCR (Phase 2) e Translation (Phase 3) sono
    completamente collegati e funzionanti in modalità headless.
  - OverlayWindow (Phase 4 UI) non è ancora implementato: se non viene
    passata un'istanza funzionante, la pipeline gira comunque fino alla
    _translation_queue, che può essere letta manualmente (vedere
    main.py --phase 3 per un esempio headless/preview).
"""

from __future__ import annotations

import logging
from queue import Empty, Queue
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

    Uso (headless, senza overlay — es. per test):
        pipeline = TranslationPipeline()
        pipeline.start()
        translated = pipeline.translation_queue.get()
        ...
        pipeline.stop()

    Uso (con overlay, quando OverlayWindow sarà implementato in Phase 4 UI):
        pipeline = TranslationPipeline(overlay_window)
        pipeline.start()   # collega automaticamente output → overlay.update_signal
        ...
        pipeline.stop()

    TODO Phase 4 (UI):
      - Gestire errori nei sotto-thread (restart automatico)
    """

    def __init__(self, overlay_window=None) -> None:
        """
        Args:
            overlay_window: istanza di OverlayWindow (Phase 4 UI), con un
                             attributo `update_signal` (pyqtSignal(list))
                             su cui viene emesso ogni list[(bbox, str)].
                             Se None, la pipeline resta headless: i
                             risultati vanno letti da `self.translation_queue`.
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

        # Thread che, se presente un overlay, inoltra _translation_queue
        # verso overlay.update_signal (ponte thread Python → main Qt thread).
        self._bridge_thread = None
        self._bridge_stop = None

        self._running: bool = False

    @property
    def translation_queue(self) -> Queue:
        """Accesso diretto alla queue di output, utile in modalità headless."""
        return self._translation_queue

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        Avvia tutti gli stadi della pipeline in ordine.

        Ordine avvio:
          1. TranslationWorker (carica il modello MarianMT — richiede
             qualche secondo, va avviato per primo così è pronto quando
             arrivano i primi risultati OCR)
          2. OCRWorker (carica PaddleOCR/EasyOCR)
          3. CaptureThread (inizia a produrre frame)

        Collegamento code:
          CaptureThread  → _capture_queue  → OCRWorker
          OCRWorker      → _ocr_queue      → TranslationWorker
          TranslationWorker → _translation_queue → OverlayWindow (via signal,
                                                     se `self.overlay` è impostato)
        """
        logger.info("Pipeline: avvio in corso...")

        # 1) TranslationWorker — per primo: il caricamento del modello
        #    MarianMT è l'operazione più lenta dell'avvio.
        self._translator = TranslationWorker(
            input_queue=self._ocr_queue,
            output_queue=self._translation_queue,
        )
        self._translator.start()

        # 2) OCRWorker — carica PaddleOCR/EasyOCR
        self._ocr = OCRWorker(
            input_queue=self._capture_queue,
            output_queue=self._ocr_queue,
        )
        self._ocr.start()

        # 3) CaptureThread — inizia subito a produrre frame
        self._capture = CaptureThread(
            frame_queue=self._capture_queue,
            monitor_index=config.monitor_index,
            target_fps=config.capture_fps,
        )
        self._capture.start()

        # Se è stato passato un overlay funzionante, avvia il bridge
        # thread che inoltra ogni risultato verso il suo Qt signal.
        if self.overlay is not None and hasattr(self.overlay, "update_signal"):
            self._start_overlay_bridge()

        self._running = True
        logger.info("Pipeline avviata (capture → OCR → translation)")

    def stop(self) -> None:
        """Ferma tutti gli stadi in ordine inverso rispetto all'avvio."""
        logger.info("Pipeline: stop...")
        self._running = False

        if self._bridge_stop is not None:
            self._bridge_stop.set()
        if self._bridge_thread is not None:
            self._bridge_thread.join(timeout=2)

        if self._capture:
            self._capture.stop()
            self._capture.join(timeout=3)

        if self._ocr:
            self._ocr.stop()

        if self._translator:
            self._translator.stop()

        logger.info("Pipeline fermata")

    def pause(self) -> None:
        """Pausa la produzione di frame (es. overlay nascosto con F10)."""
        if self._capture:
            self._capture.pause()

    def resume(self) -> None:
        """Riprende dopo una pausa."""
        if self._capture:
            self._capture.resume()

    # ── Overlay bridge (thread → Qt signal) ─────────────────────────

    def _start_overlay_bridge(self) -> None:
        """
        Avvia un thread daemon che legge da `_translation_queue` e
        chiama `overlay.update_signal.emit(...)`, così l'aggiornamento
        della UI Qt avviene sempre nel main thread tramite il
        meccanismo signal/slot (thread-safe), mai da un worker thread.
        """
        import threading

        self._bridge_stop = threading.Event()

        def _bridge_loop() -> None:
            while not self._bridge_stop.is_set():
                try:
                    output = self._translation_queue.get(timeout=0.5)
                except Empty:
                    continue
                try:
                    self.overlay.update_signal.emit(output)
                except Exception:
                    logger.exception("Pipeline: errore nell'inoltro a OverlayWindow")

        self._bridge_thread = threading.Thread(
            target=_bridge_loop, daemon=True, name="OverlayBridge"
        )
        self._bridge_thread.start()

    # ── Stats aggregate ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Statistiche aggregate da tutti gli stadi."""
        result: dict = {"running": self._running}
        if self._capture:
            result["capture"] = self._capture.stats
        if self._ocr:
            result["ocr"] = self._ocr.stats
        if self._translator:
            result["translation"] = self._translator.stats
        return result