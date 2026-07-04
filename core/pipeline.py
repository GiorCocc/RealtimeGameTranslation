"""
core/pipeline.py — Phase 4: Orchestrazione della Pipeline
                   + Phase 6: supporto restart a caldo dopo il pannello Settings

Responsabilità:
  - Crea e collega tutti e quattro gli stadi della pipeline:
      CaptureThread → OCRWorker → TranslationWorker → OverlayWindow
  - Gestisce le queue inter-thread
  - Espone metodi start/stop/pause/resume/restart a main.py e al pannello Settings
  - Ponte tra thread Python e main Qt thread via QThread/signals

Threading model:
  CaptureThread    — thread daemon dedicato (produce frame)
  OCRWorker        — thread daemon consumer (consuma frame, produce TextRegion[])
  TranslationWorker — thread daemon (consuma TextRegion[], produce (bbox, str)[])
  OverlayWindow    — main Qt thread (riceve via Qt signal thread-safe)

Stato:
  - Capture (Phase 1), OCR (Phase 2), Translation (Phase 3) e Overlay
    (Phase 4) sono completi e collegati.
  - Il pannello Settings (Phase 6) può modificare a runtime i valori in
    `config` (lingua, monitor/finestra da catturare, soglie OCR, ecc.):
    `restart()` ferma e riavvia tutti gli stadi rileggendo i valori
    aggiornati, senza dover ricreare l'overlay.
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

    Uso (con overlay):
        pipeline = TranslationPipeline(overlay_window)
        pipeline.start()   # collega automaticamente output → overlay.update_signal
        ...
        pipeline.stop()

    Uso (dopo modifiche da SettingsDialog — Phase 6):
        # SettingsDialog ha già scritto i nuovi valori nel singleton config
        pipeline.restart()   # ferma e riavvia leggendo i valori aggiornati
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

        # Phase 5: Latenza end-to-end
        self.last_e2e_latency_ms: float = 0.0
        self._e2e_latencies: list[float] = []

    @property
    def translation_queue(self) -> Queue:
        """Accesso diretto alla queue di output, utile in modalità headless."""
        return self._translation_queue

    @property
    def is_running(self) -> bool:
        return self._running

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

        Tutti i parametri (lingua, monitor/finestra, soglie, ecc.) vengono
        letti da `config` al momento della chiamata: se il pannello
        Settings li ha modificati prima di start()/restart(), la pipeline
        parte già con i valori aggiornati.

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

        # Warm-up sequence silenziosa (Phase 5)
        logger.info("Pipeline: esecuzione warm-up silenzioso...")
        try:
            # Warm-up translator (carica pesi / riscalda CUDA kernels)
            self._translator.translate_batch(["Warm up."])
            # Warm-up OCR (riscalda modelli PaddleOCR con un frame fittizio vuoto)
            import numpy as np
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Chiamiamo _process_frame_full per riscaldare la pipeline completa
            self._ocr._process_frame_full(dummy_frame, [True] * (config.frame_diff_grid_rows * config.frame_diff_grid_cols))
            logger.info("Pipeline: warm-up completato con successo")
        except Exception:
            logger.warning("Pipeline: errore durante il warm-up (proseguo comunque)", exc_info=True)

        # 3) CaptureThread — inizia subito a produrre frame.
        #    mode/window_title/region (Phase 6): vedere config.capture_mode.
        self._capture = CaptureThread(
            frame_queue=self._capture_queue,
            monitor_index=config.monitor_index,
            target_fps=config.capture_fps,
            mode=config.capture_mode,
            window_title=config.capture_window_title,
            region=config.capture_region,
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
        self._bridge_thread = None
        self._bridge_stop = None

        if self._capture:
            self._capture.stop()
            self._capture.join(timeout=3)

        if self._ocr:
            self._ocr.stop()

        if self._translator:
            self._translator.stop()

        # Rimuove eventuali code residue: un restart non deve mostrare
        # per qualche istante traduzioni "vecchie" della sessione precedente
        # (es. lingua appena cambiata dal pannello Settings).
        for q in (self._capture_queue, self._ocr_queue, self._translation_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

        if self.overlay is not None and hasattr(self.overlay, "clear_all_labels"):
            try:
                self.overlay.clear_all_labels()
            except Exception:
                logger.exception("Pipeline: errore nella pulizia delle label overlay")

        logger.info("Pipeline fermata")

    def restart(self) -> None:
        """
        Ferma e riavvia l'intera pipeline, rileggendo i valori correnti
        di `config`.

        Pensato per essere chiamato dopo che il pannello Settings (Phase 6)
        ha scritto nuovi valori nel singleton `config` — tipicamente in
        seguito a un cambio di: coppia di lingue (ricarica il modello
        MarianMT), monitor/finestra da catturare, FPS, o soglie OCR che
        richiedono di re-inizializzare i worker.

        Non è necessario ricreare l'OverlayWindow: viene riutilizzata
        l'istanza già passata al costruttore.
        """
        logger.info("Pipeline: restart richiesto (nuove impostazioni da applicare)")
        was_running = self._running
        self.stop()
        if was_running:
            self.start()

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
                    item = self._translation_queue.get(timeout=0.5)
                except Empty:
                    continue

                if isinstance(item, tuple) and len(item) == 2:
                    output, capture_time = item
                else:
                    output = item
                    capture_time = None

                # Calcola latenza end-to-end (Phase 5)
                if capture_time is not None:
                    latency_ms = (time.perf_counter() - capture_time) * 1000
                    self.last_e2e_latency_ms = latency_ms
                    self._e2e_latencies.append(latency_ms)
                    if len(self._e2e_latencies) > 50:
                        self._e2e_latencies.pop(0)

                # Se siamo in modalità finestra (o regione limitata), ricaviamo l'offset
                # (left, top) relativo al monitor e trasliamo la bbox di conseguenza,
                # dato che l'OCR ha rilevato il testo a coordinate relative al frame
                # della finestra, ma l'overlay è a schermo intero sul monitor.
                offset_x, offset_y = 0, 0
                if self._capture and self._capture._active_region:
                    offset_x = self._capture._active_region[0]
                    offset_y = self._capture._active_region[1]

                if offset_x != 0 or offset_y != 0:
                    adjusted_output = []
                    for bbox, text in output:
                        # Sposta ogni punto della bbox dell'offset (offset_x, offset_y)
                        adjusted_bbox = tuple(
                            (p[0] + offset_x, p[1] + offset_y) for p in bbox
                        )
                        adjusted_output.append((adjusted_bbox, text))
                    output = adjusted_output

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
        avg_e2e = sum(self._e2e_latencies) / len(self._e2e_latencies) if self._e2e_latencies else 0.0
        result: dict = {
            "running": self._running,
            "e2e_latency_ms": round(self.last_e2e_latency_ms, 1),
            "avg_e2e_latency_ms": round(avg_e2e, 1),
            "queue_sizes": {
                "capture_queue": self._capture_queue.qsize(),
                "ocr_queue": self._ocr_queue.qsize(),
                "translation_queue": self._translation_queue.qsize(),
            }
        }
        if self._capture:
            result["capture"] = self._capture.stats
        if self._ocr:
            result["ocr"] = self._ocr.stats
        if self._translator:
            result["translation"] = self._translator.stats
        return result