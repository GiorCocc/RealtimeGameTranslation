"""
core/capture.py — Phase 1: Screen Capture Pipeline

Responsabilità:
  - Cattura continua dello schermo tramite dxcam (DXGI Desktop Duplication API)
  - Frame differencing a griglia per saltare frame identici
  - Produzione frame in una queue thread-safe per il consumer OCR

Classi pubbliche:
  FrameDifferencer  — confronto frame su griglia NxM con hash MD5
  CaptureThread     — thread daemon che produce (frame, changed_cells) nella queue
  list_monitors()   — utility per enumerare i monitor disponibili

NON modificare la scelta di dxcam: PIL.ImageGrab e mss sono troppo lenti
per uso real-time (vedere KNOWLEDGE_BASE.md §13).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from queue import Full, Queue
from typing import Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Frame Differencing
# ══════════════════════════════════════════════════════════════════

class FrameDifferencer:
    """
    Divide il frame in una griglia rows×cols e calcola l'MD5 di ogni cella.
    Confrontando con il frame precedente, identifica quali celle sono cambiate.

    Usato per:
      1. Decidere se passare il frame all'OCR (skip se nulla è cambiato)
      2. In Phase 5: far processare all'OCR solo le zone della griglia cambiate

    Esempio:
        diff = FrameDifferencer(rows=4, cols=4)
        any_changed, cells = diff.compute(frame)
        if any_changed:
            ocr_queue.put(frame)
    """

    def __init__(self, rows: int = 4, cols: int = 4) -> None:
        self.rows = rows
        self.cols = cols
        self._prev_hashes: list[str] = []

    def compute(self, frame: np.ndarray) -> tuple[bool, list[bool]]:
        """
        Confronta `frame` con il precedente.

        Returns:
            (any_changed, changed_cells)
            - any_changed: True se almeno una cella è cambiata
            - changed_cells: lista bool di lunghezza rows*cols

        Al primo frame restituisce (True, [True, True, ...]).
        """
        h, w = frame.shape[:2]
        cell_h = h // self.rows
        cell_w = w // self.cols
        n = self.rows * self.cols

        current: list[str] = []
        for r in range(self.rows):
            for c in range(self.cols):
                y1 = r * cell_h
                x1 = c * cell_w
                # Ultima riga/colonna prende i pixel rimanenti (evita buchi)
                y2 = y1 + cell_h if r < self.rows - 1 else h
                x2 = x1 + cell_w if c < self.cols - 1 else w
                cell = frame[y1:y2, x1:x2]
                current.append(hashlib.md5(cell.tobytes()).hexdigest())

        if not self._prev_hashes:
            # Primo frame: tutto "cambiato" per forzare un run OCR iniziale
            self._prev_hashes = current
            return True, [True] * n

        changed = [a != b for a, b in zip(current, self._prev_hashes)]
        self._prev_hashes = current
        return any(changed), changed

    def reset(self) -> None:
        """Invalida lo stato: il prossimo frame verrà sempre considerato cambiato."""
        self._prev_hashes = []


# ══════════════════════════════════════════════════════════════════
# Capture Thread
# ══════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    """
    Thread daemon che cattura lo schermo in continuo tramite dxcam e
    produce tuple (frame_bgr, changed_cells) nella `frame_queue`.

    Comportamenti chiave:
      - Frame identici al precedente vengono scartati (zero OCR su freeze)
      - Se la queue è piena, l'elemento più vecchio viene rimosso per far spazio
        al frame più recente (priorità alla latenza, non alla completezza)
      - Supporta pause/resume per disattivare la cattura senza fermare il thread
      - Espone statistiche in tempo reale via `.stats`

    Uso tipico:
        q = Queue(maxsize=4)
        t = CaptureThread(q, monitor_index=0, target_fps=30)
        t.start()
        frame, cells = q.get()
        t.stop()
        t.join(timeout=3)
    """

    def __init__(
        self,
        frame_queue: Queue,
        monitor_index: int = 0,
        target_fps: int = 30,
    ) -> None:
        super().__init__(daemon=True, name="CaptureThread")
        self.frame_queue = frame_queue
        self.monitor_index = monitor_index
        self.target_fps = target_fps

        self._stop_event = threading.Event()
        # _running: quando clear() è in pausa (frame_queue smette di ricevere)
        self._running = threading.Event()
        self._running.set()  # attivo per default

        self._differencer = FrameDifferencer(
            rows=config.frame_diff_grid_rows,
            cols=config.frame_diff_grid_cols,
        )

        # Contatori per .stats
        self._n_captured: int = 0
        self._n_skipped: int = 0
        self._n_dropped: int = 0
        self._t_start: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop: dxcam → differencing → queue."""
        # Import qui: il modulo resta importabile anche su macchine non-Windows
        try:
            import dxcam
        except ImportError as exc:
            logger.error("dxcam non disponibile: %s", exc)
            return

        self._t_start = time.perf_counter()
        logger.info(
            "CaptureThread avviato — monitor=%d  target_fps=%d",
            self.monitor_index,
            self.target_fps,
        )

        camera = None
        try:
            camera = dxcam.create(
                device_idx=0,
                output_idx=self.monitor_index,
                output_color="BGR",
            )
            camera.start(target_fps=self.target_fps, video_mode=True)
            logger.info("dxcam camera avviata")

            while not self._stop_event.is_set():

                # Pausa: aspetta finché _running torna Set
                if not self._running.is_set():
                    time.sleep(0.05)
                    continue

                frame: Optional[np.ndarray] = camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                any_changed, cells = self._differencer.compute(frame)

                if not any_changed:
                    self._n_skipped += 1
                    continue

                # Copia il frame prima di metterlo in queue (dxcam riusa il buffer)
                payload = (frame.copy(), cells)

                try:
                    self.frame_queue.put_nowait(payload)
                    self._n_captured += 1
                except Full:
                    # Queue piena: scarta il più vecchio, inserisci il più recente
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.frame_queue.put_nowait(payload)
                        self._n_captured += 1
                        self._n_dropped += 1
                    except Full:
                        pass

        except Exception:
            logger.exception("CaptureThread: errore fatale")
        finally:
            if camera is not None:
                try:
                    camera.stop()
                    del camera
                except Exception:
                    logger.warning("Errore chiusura dxcam camera", exc_info=True)
            logger.info(
                "CaptureThread fermato — catturati=%d  saltati=%d  droppati=%d",
                self._n_captured,
                self._n_skipped,
                self._n_dropped,
            )

    def stop(self) -> None:
        """Segnala la terminazione del thread (non bloccante)."""
        self._stop_event.set()
        self._running.set()  # sblocca eventuale pausa

    def pause(self) -> None:
        """Sospende la produzione di frame (overlay nascosto, CPU a riposo)."""
        self._running.clear()
        logger.debug("CaptureThread in pausa")

    def resume(self) -> None:
        """Riprende la produzione di frame."""
        self._differencer.reset()  # forza un run OCR al resume
        self._running.set()
        logger.debug("CaptureThread ripreso")

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Dizionario con statistiche correnti del thread."""
        elapsed = time.perf_counter() - self._t_start if self._t_start else 1e-9
        total = self._n_captured + self._n_skipped
        return {
            "frames_captured": self._n_captured,
            "frames_skipped": self._n_skipped,
            "frames_dropped": self._n_dropped,
            "uptime_s": round(elapsed, 1),
            "effective_fps": round(self._n_captured / elapsed, 1),
            "skip_rate_pct": round(self._n_skipped / max(1, total) * 100, 1),
        }


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def list_monitors() -> list[dict]:
    """
    Restituisce la lista dei monitor disponibili con le loro proprietà.

    Returns:
        Lista di dict con chiavi: index, name, width, height, left, top.
        Lista vuota se nessun monitor rilevato.

    Usa win32api se disponibile (più accurato), altrimenti prova a sondare
    gli output dxcam sequenzialmente.
    """
    monitors: list[dict] = []

    # ── Strategia 1: win32api (accurato, disponibile con pywin32) ─
    try:
        import win32api  # type: ignore

        for i, hmon in enumerate(win32api.EnumDisplayMonitors()):
            try:
                info = win32api.GetMonitorInfo(hmon[0])
                rect = info["Monitor"]
                monitors.append(
                    {
                        "index": i,
                        "name": info.get("Device", f"Monitor {i}"),
                        "width": rect[2] - rect[0],
                        "height": rect[3] - rect[1],
                        "left": rect[0],
                        "top": rect[1],
                    }
                )
            except Exception:
                pass
        if monitors:
            return monitors
    except ImportError:
        pass

    # ── Strategia 2: sonda dxcam output indices ───────────────────
    try:
        import dxcam  # type: ignore

        for i in range(8):
            try:
                cam = dxcam.create(device_idx=0, output_idx=i)
                frame = cam.grab()
                if frame is not None:
                    h, w = frame.shape[:2]
                    monitors.append(
                        {"index": i, "name": f"Monitor {i}", "width": w, "height": h,
                         "left": 0, "top": 0}
                    )
                del cam
            except Exception:
                break  # indice non valido → stop
    except ImportError:
        logger.error("dxcam non installato — eseguire: pip install dxcam")

    return monitors
