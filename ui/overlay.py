"""
ui/overlay.py — Phase 4: Overlay Trasparente PyQt6

Responsabilità:
  - Finestra frameless, trasparente, always-on-top, a schermo intero
  - Click-through totale via WS_EX_TRANSPARENT (win32api)
  - Riceve list[(bbox, translated_text)] via Qt signal thread-safe
  - Posiziona QLabel sopra ogni bounding box con sfondo semi-trasparente
  - Animazione fade in/out su cambio testo (QPropertyAnimation)
  - Smoothing coordinate bbox (media mobile su N frame) per evitare flickering
  - Toggle semi-trasparente / opaco (F9)

ATTENZIONE: NON usare tkinter (no trasparenza reale) né pygame (non overlay).
Vedere KNOWLEDGE_BASE.md §13.

TODO Phase 4: implementare dopo aver completato Phases 1-3.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Bbox Smoother
# ══════════════════════════════════════════════════════════════════

class BBoxSmoother:
    """
    Applica una media mobile sulle coordinate di una bounding box
    per eliminare il micro-flickering causato da variazioni OCR frame-to-frame.

    Usato da OverlayWindow per smorzare le posizioni delle QLabel.

    TODO Phase 4: integrare in OverlayWindow._update_labels()
    """

    def __init__(self, window: int = 4) -> None:
        """
        Args:
            window: numero di frame su cui mediare (config.overlay_smoothing_frames)
        """
        self._window = window
        self._history: deque = deque(maxlen=window)

    def update(self, bbox: tuple) -> tuple:
        """
        Aggiorna con la nuova bbox e restituisce la bbox smussata.

        Args:
            bbox: 4 punti ((x1,y1),(x2,y2),(x3,y3),(x4,y4))

        Returns:
            bbox mediata sugli ultimi N frame.
        """
        self._history.append(bbox)
        n = len(self._history)
        smoothed = tuple(
            (
                sum(f[i][0] for f in self._history) / n,
                sum(f[i][1] for f in self._history) / n,
            )
            for i in range(4)
        )
        return smoothed

    def reset(self) -> None:
        self._history.clear()


# ══════════════════════════════════════════════════════════════════
# Overlay Window — Phase 4 (TODO)
# ══════════════════════════════════════════════════════════════════

class OverlayWindow:
    """
    Finestra overlay PyQt6 trasparente e click-through.

    Caratteristiche tecniche:
      - Qt.FramelessWindowHint + Qt.WindowStaysOnTopHint
      - WA_TranslucentBackground per trasparenza totale
      - win32api WS_EX_TRANSPARENT per click-through (vedi _apply_click_through)
      - QLabel per ogni TextRegion, posizionate con move() sulle coordinate bbox
      - QPropertyAnimation per fade in/out su cambio testo
      - Ricezione aggiornamenti via Qt signal (thread-safe da worker thread)

    TODO Phase 4: implementare tutti i metodi.

    Struttura suggerita:
        class OverlayWindow(QWidget):
            # Signal ricevuto da TranslationWorker (thread diverso)
            update_signal = pyqtSignal(list)  # list[(bbox, str)]

            def __init__(self):
                super().__init__()
                self._labels: dict[str, TranslationLabel] = {}
                self._smoothers: dict[str, BBoxSmoother] = {}
                self._opaque_mode: bool = False
                self._setup_window()
                self._apply_click_through()
                self.update_signal.connect(self._on_update)

            def _setup_window(self): ...
            def _apply_click_through(self): ...  # win32api
            def _on_update(self, regions): ...   # aggiorna labels
            def toggle_overlay(self): ...        # F10
            def toggle_opaque(self): ...         # F9
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Phase 4: implementare OverlayWindow come QWidget. "
            "Vedere docstring per la struttura completa."
        )

    def _setup_window(self) -> None:
        """
        TODO Phase 4 — Configura flags Qt della finestra.

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QWidget

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # non appare in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        """
        raise NotImplementedError("Phase 4")

    def _apply_click_through(self) -> None:
        """
        TODO Phase 4 — Imposta WS_EX_TRANSPARENT tramite win32api.

        import win32api
        import win32con
        hwnd = int(self.winId())
        ex_style = win32api.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32api.SetWindowLong(
            hwnd, win32con.GWL_EXSTYLE,
            ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
        )
        """
        raise NotImplementedError("Phase 4")

    def _on_update(self, regions: list) -> None:
        """
        TODO Phase 4 — Riceve list[(bbox, translated_text)] e aggiorna le QLabel.

        Per ogni regione:
          - Se testo già presente: aggiorna posizione (con smoother) e testo
          - Se testo nuovo: crea TranslationLabel con fade-in
          - Rimuove label di testi non più presenti (fade-out)
        """
        raise NotImplementedError("Phase 4")

    def toggle_overlay(self) -> None:
        """TODO Phase 4 — Mostra/nasconde tutto l'overlay (hotkey F10)."""
        raise NotImplementedError("Phase 4")

    def toggle_opaque(self) -> None:
        """TODO Phase 4 — Toggle sfondo label semi-trasparente/opaco (hotkey F9)."""
        raise NotImplementedError("Phase 4")
