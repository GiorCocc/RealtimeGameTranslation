"""
main.py — Game Translation Overlay — Entry Point

Modalità di avvio:
  Phase 1 (attuale):
    python main.py --phase 1                  # Test capture con preview OpenCV
    python main.py --phase 1 --no-preview     # Solo stats, niente finestra
    python main.py --phase 1 --monitor 1      # Usa monitor secondario

  Futura (Phase 4+):
    python main.py                            # Avvio completo con overlay Qt

Argomenti principali:
  --monitor  INT    Indice monitor da catturare (default: 0)
  --fps      INT    Target FPS cattura (default: 30)
  --phase    INT    Fase da testare in sviluppo (default: produzione completa)
  --no-preview      Nessuna finestra OpenCV preview (Phase 1 headless)
  --debug           Abilita output di debug e overlay diagnostico
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from queue import Queue

from config import config

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Phase 1: Capture Test
# ══════════════════════════════════════════════════════════════════

def run_phase1(monitor_index: int, preview: bool) -> None:
    """
    Test completo della capture pipeline (Phase 1).

    Avvia CaptureThread e mostra:
      - Lista monitor disponibili
      - Preview frame con griglia differencing sovrapposta (se --preview)
      - Statistiche aggiornate ogni 2 secondi
      - Report finale a Ctrl+C
    """
    from core.capture import CaptureThread, list_monitors

    _print_banner("Phase 1: Capture Pipeline")

    # ── Lista monitor ─────────────────────────────────────────────
    monitors = list_monitors()
    if not monitors:
        logger.error(
            "Nessun monitor rilevato. Assicurarsi di essere su Windows con "
            "dxcam installato (pip install dxcam)."
        )
        sys.exit(1)

    print("Monitor disponibili:")
    for m in monitors:
        w = m.get("width", "?")
        h = m.get("height", "?")
        mark = " ◀ selezionato" if m["index"] == monitor_index else ""
        print(f"  [{m['index']}] {m.get('name', 'Monitor')}  {w}×{h}{mark}")
    print()

    if monitor_index >= len(monitors):
        logger.error("Monitor %d non trovato (disponibili: %d).", monitor_index, len(monitors))
        sys.exit(1)

    if preview:
        print("Preview attiva — premi 'q' nella finestra o Ctrl+C nel terminale per uscire.")
    else:
        print("Modalità headless — premi Ctrl+C per uscire.")
    print()

    # ── Setup ─────────────────────────────────────────────────────
    frame_queue: Queue = Queue(maxsize=4)
    capture = CaptureThread(
        frame_queue=frame_queue,
        monitor_index=monitor_index,
        target_fps=config.capture_fps,
    )

    running = True

    def _on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── Avvio ─────────────────────────────────────────────────────
    capture.start()
    t_last_stats = time.perf_counter()

    try:
        while running:
            now = time.perf_counter()

            # Stats ogni 2 secondi
            if now - t_last_stats >= 2.0:
                s = capture.stats
                print(
                    f"  captured={s['frames_captured']:>5}  "
                    f"skipped={s['frames_skipped']:>5}  "
                    f"dropped={s['frames_dropped']:>3}  "
                    f"fps_eff={s['effective_fps']:>5.1f}  "
                    f"skip_rate={s['skip_rate_pct']:>4.1f}%  "
                    f"up={s['uptime_s']}s"
                )
                t_last_stats = now

            if preview:
                _run_preview_loop(frame_queue, capture, running)
                # Se 'q' premuto nella finestra preview → esce
                # (cv2.waitKey gestito dentro _run_preview_loop)
                try:
                    import cv2
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        running = False
                except ImportError:
                    pass
            else:
                # Headless: svuota la queue per evitare accumulo
                _drain_queue(frame_queue)
                time.sleep(0.05)

    finally:
        running = False
        capture.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass

        # ── Report finale ─────────────────────────────────────────
        s = capture.stats
        _print_banner("Report Finale")
        print(f"  Frame catturati  : {s['frames_captured']}")
        print(f"  Frame saltati    : {s['frames_skipped']}  (identici al precedente)")
        print(f"  Frame droppati   : {s['frames_dropped']}  (queue piena, frame rimpiazzato)")
        print(f"  FPS effettivi    : {s['effective_fps']}")
        print(f"  Skip rate        : {s['skip_rate_pct']}%  (atteso ~60-80% su schermo statico)")
        print(f"  Uptime           : {s['uptime_s']}s")
        print()
        print("Phase 1 completata. Prossimo step: implementare core/ocr.py (Phase 2).")
        print()


def _run_preview_loop(frame_queue: Queue, capture: "CaptureThread", running_ref) -> None:
    """Gestisce una singola iterazione del loop preview OpenCV."""
    try:
        import cv2
        import numpy as np
        from config import config as cfg
    except ImportError:
        logger.warning("opencv-python non installato — preview non disponibile")
        time.sleep(0.05)
        return

    try:
        frame, changed_cells = frame_queue.get(timeout=0.05)
    except Exception:
        return

    # Ridimensiona per display (max 1280px wide)
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))
        h, w = frame.shape[:2]

    # Sovrapponi griglia differencing
    _draw_diff_grid(frame, changed_cells, cfg.frame_diff_grid_rows, cfg.frame_diff_grid_cols)

    # Overlay stats
    s = capture.stats
    label = (
        f"FPS: {s['effective_fps']:.1f}  |  "
        f"skip: {s['skip_rate_pct']:.0f}%  |  "
        f"dropped: {s['frames_dropped']}"
    )
    cv2.rectangle(frame, (0, 0), (len(label) * 8 + 10, 22), (0, 0, 0), -1)
    cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imshow("Game Translator — Phase 1 Capture Preview  [q = esci]", frame)


def _draw_diff_grid(
    frame: "np.ndarray",
    changed_cells: list[bool],
    rows: int,
    cols: int,
) -> None:
    """
    Disegna la griglia differencing sul frame (in-place).
    Celle cambiate → verde semi-trasparente.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    h, w = frame.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    overlay = frame.copy()

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(changed_cells) and changed_cells[idx]:
                y1, x1 = r * cell_h, c * cell_w
                y2 = y1 + cell_h if r < rows - 1 else h
                x2 = x1 + cell_w if c < cols - 1 else w
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 80), -1)

    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Linee griglia
    for r in range(1, rows):
        cv2.line(frame, (0, r * cell_h), (w, r * cell_h), (80, 80, 80), 1)
    for c in range(1, cols):
        cv2.line(frame, (c * cell_w, 0), (c * cell_w, h), (80, 80, 80), 1)


def _drain_queue(q: Queue) -> None:
    """Svuota la queue senza bloccare."""
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break


# ══════════════════════════════════════════════════════════════════
# Full Application (Phase 4+)
# ══════════════════════════════════════════════════════════════════

def run_full_app() -> None:
    """
    TODO Phase 4 — Avvio completo dell'applicazione con overlay Qt.

    Sequenza:
      1. Carica settings da settings.json
      2. Avvia QApplication
      3. Crea OverlayWindow
      4. Avvia TranslationPipeline (collega a OverlayWindow via signal)
      5. Registra hotkey globali (keyboard)
      6. Avvia TrayIcon
      7. QApplication.exec() — event loop Qt (blocca qui)
      8. Al quit: pipeline.stop(), tray.stop()
    """
    raise NotImplementedError(
        "Phase 4: implementare run_full_app() dopo aver completato "
        "OverlayWindow, OCRWorker e TranslationWorker."
    )


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def _print_banner(title: str) -> None:
    width = max(60, len(title) + 6)
    print(f"\n{'═' * width}")
    print(f"  Game Translator — {title}")
    print(f"{'═' * width}\n")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Game Translation Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase", "-p",
        type=int,
        default=None,
        metavar="N",
        help="Fase da testare (1=capture, 2=ocr, 3=translation). "
             "Omettere per avvio completo (Phase 4+).",
    )
    p.add_argument(
        "--monitor", "-m",
        type=int,
        default=0,
        metavar="INDEX",
        help="Indice monitor da catturare (default: 0)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="FPS",
        help="Target FPS cattura (default: 30)",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Nessuna finestra preview OpenCV (solo stats nel terminale)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Abilita logging DEBUG e overlay diagnostico",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Aggiorna config da CLI
    config.monitor_index = args.monitor
    config.capture_fps = args.fps
    config.debug_capture_preview = not args.no_preview

    # ── Dispatch ──────────────────────────────────────────────────
    if args.phase == 1 or args.phase is None:
        # Phase 1 è l'unica implementata; in futuro None → run_full_app()
        run_phase1(
            monitor_index=args.monitor,
            preview=not args.no_preview,
        )
    else:
        print(f"Phase {args.phase} non ancora implementata.")
        print("Implementare i metodi TODO nelle fasi successive.")
        sys.exit(1)
