"""
ui/tray.py — Phase 6: System Tray Icon

Responsabilità:
  - Icona nella system tray con menu contestuale
  - Accesso rapido a: toggle overlay, settings, statistiche, quit
  - Notifiche balloon (es. "Modello caricato", "Errore OCR")

TODO Phase 6: implementare con pystray.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TrayIcon:
    """
    TODO Phase 6 — Icona system tray con pystray.

    Menu contestuale suggerito:
      ✓ Overlay attivo      (toggle, checkmark)
      ─────────────────────
      ⚙ Impostazioni...     (apre SettingsDialog)
      📊 Statistiche        (popup con stats pipeline)
      ─────────────────────
      ✕ Esci

    Esempio struttura:
        import pystray
        from PIL import Image, ImageDraw

        class TrayIcon:
            def __init__(self, on_toggle, on_settings, on_quit):
                self._icon = pystray.Icon(
                    name="game-translator",
                    icon=self._create_icon(),
                    title="Game Translator",
                    menu=pystray.Menu(...)
                )

            def _create_icon(self) -> Image.Image:
                # Icona 64×64 disegnata con PIL
                ...

            def run(self):
                self._icon.run()  # blocca il thread corrente

            def stop(self):
                self._icon.stop()
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_settings: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        raise NotImplementedError("Phase 6: implementare con pystray")

    def run_detached(self) -> None:
        """TODO Phase 6 — Avvia la tray icon in un thread separato."""
        raise NotImplementedError("Phase 6")

    def stop(self) -> None:
        """TODO Phase 6 — Rimuove l'icona dalla tray."""
        raise NotImplementedError("Phase 6")

    def notify(self, title: str, message: str) -> None:
        """TODO Phase 6 — Mostra una notifica balloon."""
        raise NotImplementedError("Phase 6")
