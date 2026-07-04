"""
ui/tray.py — Phase 6: System Tray Icon

Responsabilità:
  - Icona nella system tray con menu contestuale
  - Accesso rapido a: toggle overlay, impostazioni, statistiche, quit
  - Notifiche balloon (es. "Modello caricato", "Errore OCR")

ATTENZIONE — thread-safety:
  pystray gestisce la propria icona in un thread separato dal main thread
  Qt. Le callback passate al costruttore (`on_toggle`, `on_settings`,
  `on_quit`, `on_stats`) vengono quindi invocate DA QUEL THREAD, non dal
  thread Qt. Se devono toccare oggetti Qt (OverlayWindow, QDialog, ecc.),
  il CHIAMANTE (main.py) è responsabile di avvolgerle con
  `QTimer.singleShot(0, funzione_reale)` prima di passarle a TrayIcon —
  esattamente come già avviene per gli hotkey globali in ui/overlay.py
  (`keyboard` callback → thread non-Qt → QTimer.singleShot per rientrare
  nel thread Qt).

Uso tipico (da main.py, dopo aver creato overlay e pipeline):
    def _tray_toggle():
        QTimer.singleShot(0, overlay.toggle_overlay)

    def _tray_settings():
        QTimer.singleShot(0, lambda: _open_settings_dialog(overlay, pipeline))

    def _tray_quit():
        QTimer.singleShot(0, app.quit)

    tray = TrayIcon(on_toggle=_tray_toggle, on_settings=_tray_settings, on_quit=_tray_quit)
    tray.run_detached()
    ...
    tray.stop()   # alla chiusura dell'app
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_ICON_SIZE = 64


class TrayIcon:
    """
    Icona system tray con pystray.

    Menu contestuale:
      ✓ Overlay attivo      (checkbox, toggle overlay)
      ─────────────────────
      ⚙ Impostazioni...     (apre SettingsDialog)
      📊 Statistiche        (mostra stats pipeline)
      ─────────────────────
      ✕ Esci
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_settings: Callable[[], None],
        on_quit: Callable[[], None],
        on_stats: Optional[Callable[[], None]] = None,
        overlay_active: bool = True,
    ) -> None:
        """
        Args:
            on_toggle: invocata quando l'utente clicca "Overlay attivo"
                (toggle mostra/nascondi overlay — equivalente hotkey F10)
            on_settings: invocata per aprire il pannello impostazioni
            on_quit: invocata per chiudere l'applicazione
            on_stats: invocata per mostrare le statistiche correnti della
                pipeline (opzionale — se assente la voce di menu è disabilitata)
            overlay_active: stato iniziale della checkbox "Overlay attivo",
                da sincronizzare con lo stato reale di OverlayWindow all'avvio
        """
        import pystray

        self._on_toggle = on_toggle
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._on_stats = on_stats
        self._overlay_active = overlay_active

        self._icon = pystray.Icon(
            name="game-translator",
            icon=self._create_icon_image(),
            title="Game Translator — Realtime Overlay",
            menu=self._build_menu(pystray),
        )

    # ── Costruzione icona/menu ────────────────────────────────────

    def _create_icon_image(self):
        """Disegna un'icona 64×64 semplice con PIL (nessun asset esterno richiesto)."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Sfondo: rettangolo arrotondato blu scuro (richiama il tema dell'overlay)
        draw.rounded_rectangle(
            (2, 2, _ICON_SIZE - 2, _ICON_SIZE - 2),
            radius=14,
            fill=(28, 32, 48, 255),
            outline=(90, 140, 255, 255),
            width=2,
        )

        # Glifo "GT" (Game Translator) centrato
        text = "GT"
        try:
            font = ImageFont.truetype("arial.ttf", 26)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((_ICON_SIZE - tw) / 2 - bbox[0], (_ICON_SIZE - th) / 2 - bbox[1]),
            text,
            font=font,
            fill=(235, 240, 250, 255),
        )
        return img

    def _build_menu(self, pystray):
        items = [
            pystray.MenuItem(
                "Overlay attivo",
                self._handle_toggle,
                checked=lambda _item: self._overlay_active,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Impostazioni...", self._handle_settings),
            pystray.MenuItem(
                "Statistiche",
                self._handle_stats,
                enabled=self._on_stats is not None,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Esci", self._handle_quit),
        ]
        return pystray.Menu(*items)

    # ── Handler interni (eseguiti nel thread di pystray) ──────────

    def _handle_toggle(self, icon=None, item=None) -> None:
        self._overlay_active = not self._overlay_active
        try:
            self._on_toggle()
        except Exception:
            logger.exception("TrayIcon: errore nella callback on_toggle")

    def _handle_settings(self, icon=None, item=None) -> None:
        try:
            self._on_settings()
        except Exception:
            logger.exception("TrayIcon: errore nella callback on_settings")

    def _handle_stats(self, icon=None, item=None) -> None:
        if self._on_stats is None:
            return
        try:
            self._on_stats()
        except Exception:
            logger.exception("TrayIcon: errore nella callback on_stats")

    def _handle_quit(self, icon=None, item=None) -> None:
        try:
            self._on_quit()
        except Exception:
            logger.exception("TrayIcon: errore nella callback on_quit")
        finally:
            self.stop()

    # ── Ciclo di vita ─────────────────────────────────────────────

    def run_detached(self) -> None:
        """Avvia la tray icon in un thread daemon separato (non bloccante)."""
        threading.Thread(target=self._icon.run, daemon=True, name="TrayIconThread").start()
        logger.info("TrayIcon avviata")

    def stop(self) -> None:
        """Rimuove l'icona dalla tray."""
        try:
            self._icon.stop()
        except Exception:
            logger.exception("TrayIcon: errore durante lo stop")

    def sync_overlay_state(self, active: bool) -> None:
        """
        Sincronizza la checkbox del menu con lo stato reale dell'overlay,
        utile quando il toggle avviene da un'altra fonte (es. hotkey F10
        premuto direttamente, senza passare dal menu tray).
        """
        self._overlay_active = active
        try:
            self._icon.update_menu()
        except Exception:
            logger.debug("TrayIcon: update_menu non disponibile su questo backend")

    def notify(self, title: str, message: str) -> None:
        """
        Mostra una notifica balloon, se supportata dal backend pystray
        in uso sul sistema corrente (su Windows generalmente sì).
        """
        try:
            self._icon.notify(message, title)
        except NotImplementedError:
            logger.debug("TrayIcon: notify() non supportata su questo backend")
        except Exception:
            logger.exception("TrayIcon: errore durante notify()")