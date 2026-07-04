"""
ui/overlay.py — Phase 4: Overlay Trasparente PyQt6

Responsabilità:
  - Finestra frameless, trasparente, always-on-top, a schermo intero
  - Click-through totale via WS_EX_TRANSPARENT + WA_TransparentForMouseEvents
  - DPI-awareness: gestisce correttamente scale factor ≠ 100%
  - Riceve list[(bbox, translated_text)] via Qt signal thread-safe
  - Posiziona QLabel sopra ogni bounding box con sfondo semi-trasparente
  - Animazione fade in/out su cambio testo (QPropertyAnimation)
  - Smoothing coordinate bbox (media mobile su N frame) per evitare flickering
  - Le label rimangono a schermo finché il frame sottostante non cambia:
    vengono rimosse (con fade-out) solo quando l'OCR emette un nuovo
    aggiornamento che non le include più — nessun timeout temporale
  - Toggle overlay visibilità (F10) e opacità sfondo label (F9)

ATTENZIONE: NON usare tkinter (no trasparenza reale) né pygame (non overlay).
Vedere KNOWLEDGE_BASE.md §13.
"""

from __future__ import annotations

import ctypes
import logging
from collections import deque
from typing import Optional

import keyboard
import win32api
import win32con
from PyQt6.QtCore import (
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QLabel,
    QWidget,
)

from config import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DPI Awareness
# ══════════════════════════════════════════════════════════════════

def enable_dpi_awareness() -> None:
    """
    Abilita la DPI-awareness per-monitor (V2) sul processo corrente.

    Deve essere chiamata PRIMA di creare la QApplication.
    Se SetProcessDpiAwarenessContext (Win10 1703+) non è disponibile,
    usa il fallback SetProcessDpiAwareness (Win 8.1+).
    """
    try:
        # Per-Monitor Aware V2 (Windows 10 1703+)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        )
        logger.info("DPI awareness: Per-Monitor V2 abilitata")
        return
    except (AttributeError, OSError):
        pass

    try:
        # Fallback: Per-Monitor Aware V1 (Win 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        logger.info("DPI awareness: Per-Monitor V1 abilitata")
        return
    except (AttributeError, OSError):
        pass

    logger.warning("DPI awareness: nessuna API disponibile — "
                   "le coordinate overlay potrebbero non corrispondere "
                   "alla risoluzione reale del monitor")


def _get_screen_dpi_scale(screen) -> float:
    """
    Restituisce il fattore di scala DPI dello schermo corrente.

    Args:
        screen: QScreen di riferimento.

    Returns:
        Scale factor (es. 1.0 = 100%, 1.25 = 125%, 1.5 = 150%).
    """
    if screen is None:
        return 1.0
    # devicePixelRatio() restituisce il fattore di scala effettivo
    # dopo che la DPI-awareness è stata abilitata.
    return screen.devicePixelRatio()


# ══════════════════════════════════════════════════════════════════
# Bbox Smoother
# ══════════════════════════════════════════════════════════════════

class BBoxSmoother:
    """
    Applica una media mobile sulle coordinate di una bounding box
    per eliminare il micro-flickering causato da variazioni OCR frame-to-frame.

    Usato da OverlayWindow per smorzare le posizioni delle TranslationLabel.
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
# Translation Label
# ══════════════════════════════════════════════════════════════════

_LABEL_PADDING_H = 6   # px di padding orizzontale interno alla label
_LABEL_PADDING_V = 2   # px di padding verticale interno alla label


class TranslationLabel(QLabel):
    """
    Label singola che mostra il testo tradotto sopra la posizione originale.

    Caratteristiche:
      - Sfondo semi-trasparente (overlay_bg_alpha) con angoli arrotondati
      - Animazione fade-in alla creazione, fade-out alla rimozione
      - Font e dimensioni configurabili da config
      - Toggle opacità sfondo (F9)
    """

    def __init__(
        self,
        text: str,
        parent: QWidget,
        bg_alpha: int = 160,
        opaque_mode: bool = False,
    ) -> None:
        super().__init__(text, parent)

        self._bg_alpha = bg_alpha
        self._opaque_mode = opaque_mode

        # ── Opacity effect per animazioni ──────────────────────────
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        # ── Stile ──────────────────────────────────────────────────
        self._apply_style()
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.show()

    def _apply_style(self) -> None:
        """Applica lo stile CSS alla label in base alla modalità corrente."""
        self._bg_alpha = config.overlay_bg_alpha
        alpha = 255 if self._opaque_mode else self._bg_alpha
        bg_color = QColor(20, 20, 30, alpha)
        text_color = QColor(240, 240, 245)

        font = QFont(config.overlay_font_family, config.overlay_font_size)
        font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        self.setFont(font)

        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: rgba({bg_color.red()}, {bg_color.green()}, "
            f"    {bg_color.blue()}, {bg_color.alpha()});"
            f"  color: rgba({text_color.red()}, {text_color.green()}, "
            f"    {text_color.blue()}, {text_color.alpha()});"
            f"  border-radius: 4px;"
            f"  padding: {_LABEL_PADDING_V}px {_LABEL_PADDING_H}px;"
            f"}}"
        )

    def fade_in(self, duration_ms: int = 150) -> None:
        """Animazione fade-in da opacità 0 a 1."""
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start()
        # Mantieni riferimento per evitare garbage collection
        self._fade_anim = anim

    def fade_out(self, duration_ms: int = 150, on_finished=None) -> None:
        """
        Animazione fade-out da opacità corrente a 0.

        Args:
            duration_ms: durata dell'animazione.
            on_finished: callback invocata al termine (es. deleteLater).
        """
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(self._opacity_effect.opacity())
        anim.setEndValue(0.0)
        if on_finished is not None:
            anim.finished.connect(on_finished)
        anim.start()
        self._fade_anim = anim

    def set_opaque_mode(self, opaque: bool) -> None:
        """Alterna tra sfondo semi-trasparente e opaco."""
        self._opaque_mode = opaque
        self._apply_style()

    def update_position(self, x: int, y: int, w: int, h: int) -> None:
        """Muove e ridimensiona la label sulla posizione indicata."""
        self.setGeometry(x, y, w, h)

    def update_text(self, text: str) -> None:
        """Aggiorna il testo della label."""
        if self.text() != text:
            self.setText(text)


# ══════════════════════════════════════════════════════════════════
# Overlay Window
# ══════════════════════════════════════════════════════════════════

# Griglia di arrotondamento per generare la chiave di matching delle label
# basata sul centroide della bbox. Un valore più alto rende il matching
# più aggressivo (tollera spostamenti maggiori tra frame).
_CENTROID_GRID_PX = 50



def _bbox_centroid(bbox: tuple) -> tuple[float, float]:
    """Calcola il centroide di una bbox a 4 punti."""
    cx = sum(p[0] for p in bbox) / 4.0
    cy = sum(p[1] for p in bbox) / 4.0
    return cx, cy


def _bbox_key(bbox: tuple) -> tuple[int, int]:
    """
    Genera una chiave di matching arrotondando il centroide della bbox
    a una griglia discreta. Due bbox con centroide vicino producono
    la stessa chiave, consentendo il re-use della label anche se
    l'OCR oscilla leggermente tra frame.
    """
    cx, cy = _bbox_centroid(bbox)
    return (
        round(cx / _CENTROID_GRID_PX),
        round(cy / _CENTROID_GRID_PX),
    )


def _bbox_rect(bbox: tuple) -> tuple[int, int, int, int]:
    """
    Converte una bbox a 4 punti in un rettangolo (x, y, w, h)
    basato sul bounding rect esterno.
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x = int(min(xs))
    y = int(min(ys))
    w = int(max(xs) - min(xs))
    h = int(max(ys) - min(ys))
    # Aggiungi spazio per il padding CSS
    return x - _LABEL_PADDING_H, y - _LABEL_PADDING_V, w + 2 * _LABEL_PADDING_H, h + 2 * _LABEL_PADDING_V


class _LabelEntry:
    """Voce interna per tracciare ogni TranslationLabel attiva e il suo stato."""

    __slots__ = ("label", "smoother", "fading_out")

    def __init__(self, label: TranslationLabel, smoother: BBoxSmoother) -> None:
        self.label = label
        self.smoother = smoother
        self.fading_out: bool = False


class OverlayWindow(QWidget):
    """
    Finestra overlay PyQt6 trasparente e click-through.

    Caratteristiche tecniche:
      - Qt.FramelessWindowHint + Qt.WindowStaysOnTopHint + Qt.Tool
      - WA_TranslucentBackground per trasparenza totale
      - win32api WS_EX_TRANSPARENT per click-through
      - DPI-awareness: le coordinate vengono compensate per il fattore
        di scala DPI del monitor corrente
      - TranslationLabel per ogni TextRegion, con fade in/out
      - BBoxSmoother per stabilizzare le coordinate frame-to-frame
      - Hotkey globali F9 (toggle opacità) e F10 (toggle visibilità)
    """

    # Signal ricevuto dal bridge thread in pipeline.py
    # Trasporta list[(bbox, translated_text)]
    update_signal = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()

        # ── Stato interno ──────────────────────────────────────────
        # Chiave: _bbox_key(smoothed_bbox) → _LabelEntry
        self._entries: dict[tuple[int, int], _LabelEntry] = {}
        self._opaque_mode: bool = False
        self._overlay_visible: bool = True
        self._dpi_scale: float = 1.0

        # ── Setup ──────────────────────────────────────────────────
        self._setup_window()
        self._apply_click_through()
        self._register_hotkeys()

        # ── Connessione signal → slot ──────────────────────────────
        self.update_signal.connect(self._on_update)

        logger.info("OverlayWindow inizializzata (DPI scale: %.2f)", self._dpi_scale)

    # ── Window setup ─────────────────────────────────────────────

    def _setup_window(self) -> None:
        """
        Configura i flag Qt della finestra overlay:
        frameless, always-on-top, trasparente, fullscreen.
        """
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Non appare nella taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Doppia protezione: sia WS_EX_TRANSPARENT (win32) sia il flag Qt.
        # Garantisce che nessun evento mouse raggiunga mai la finestra overlay.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Rileva il fattore di scala DPI dello schermo corrente
        screen = QApplication.primaryScreen()
        self._dpi_scale = _get_screen_dpi_scale(screen)

        self.showFullScreen()

    def _apply_click_through(self) -> None:
        """
        Imposta WS_EX_TRANSPARENT sul HWND della finestra tramite win32api,
        garantendo che tutti gli input (mouse, tastiera) passino direttamente
        al gioco sottostante senza essere intercettati dall'overlay.
        Esclude inoltre l'overlay dalla cattura dello schermo per evitare feedback loop.
        """
        hwnd = int(self.winId())
        ex_style = win32api.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32api.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED,
        )

        # Esclude la finestra dell'overlay dalla cattura (dxcam / DXGI Desktop Duplication)
        # 0x00000011 = WDA_EXCLUDEFROMCAPTURE (Windows 10 2004+)
        try:
            ret = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
            if ret:
                logger.info("OverlayWindow: finestra esclusa dalla cattura (SetWindowDisplayAffinity)")
            else:
                logger.warning("OverlayWindow: impossibile impostare SetWindowDisplayAffinity (ritorno 0)")
        except Exception:
            logger.exception("OverlayWindow: errore durante l'impostazione di SetWindowDisplayAffinity")

        logger.debug("Click-through applicato (HWND=%d)", hwnd)

    # ── Hotkey ───────────────────────────────────────────────────

    def _register_hotkeys(self) -> None:
        """Registra gli hotkey globali F9 e F10 tramite la libreria `keyboard`."""
        try:
            keyboard.add_hotkey(
                config.hotkey_toggle_overlay,
                self._on_hotkey_toggle_overlay,
                suppress=False,
            )
            keyboard.add_hotkey(
                config.hotkey_toggle_opaque,
                self._on_hotkey_toggle_opaque,
                suppress=False,
            )
            logger.info(
                "Hotkey registrati: %s = toggle overlay, %s = toggle opacità",
                config.hotkey_toggle_overlay.upper(),
                config.hotkey_toggle_opaque.upper(),
            )
        except Exception:
            logger.exception("Errore nella registrazione degli hotkey globali")

    def _unregister_hotkeys(self) -> None:
        """Rimuove tutti gli hotkey registrati dalla libreria `keyboard`."""
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            logger.exception("Errore nella rimozione degli hotkey")

    def apply_hotkey_changes(self) -> None:
        """
        Phase 6 — Ri-registra gli hotkey globali con i valori correnti di
        `config` (hotkey_toggle_overlay / hotkey_toggle_opaque).

        Da chiamare dopo che il pannello Settings ha scritto nuovi valori
        nel singleton config: `keyboard.add_hotkey` cattura la stringa
        dell'hotkey al momento della registrazione, quindi un cambio a
        runtime richiede sempre un unhook + re-register esplicito.
        """
        self._unregister_hotkeys()
        self._register_hotkeys()
        logger.info(
            "OverlayWindow: hotkey aggiornati (%s = toggle overlay, %s = toggle opacità)",
            config.hotkey_toggle_overlay.upper(),
            config.hotkey_toggle_opaque.upper(),
        )

    def apply_style_changes(self) -> None:
        """
        Phase 6 — Applica le modifiche di stile (font family, font size, bg alpha)
        a tutte le label correntemente attive.
        """
        logger.info(
            "OverlayWindow: applicazione modifiche di stile (font=%s, size=%d, alpha=%d)",
            config.overlay_font_family,
            config.overlay_font_size,
            config.overlay_bg_alpha,
        )
        for entry in self._entries.values():
            entry.label._apply_style()

    def _on_hotkey_toggle_overlay(self) -> None:
        """Callback hotkey F10: mostra/nasconde l'overlay."""
        # QTimer.singleShot assicura che l'operazione avvenga nel thread Qt
        QTimer.singleShot(0, self.toggle_overlay)

    def _on_hotkey_toggle_opaque(self) -> None:
        """Callback hotkey F9: toggle opacità sfondo label."""
        QTimer.singleShot(0, self.toggle_opaque)

    # ── Toggle ───────────────────────────────────────────────────

    def toggle_overlay(self) -> None:
        """Mostra/nasconde tutto l'overlay (hotkey F10)."""
        self._overlay_visible = not self._overlay_visible
        if self._overlay_visible:
            self.show()
            logger.info("Overlay: visibile")
        else:
            self.hide()
            logger.info("Overlay: nascosto")

    def toggle_opaque(self) -> None:
        """Toggle sfondo label semi-trasparente/opaco (hotkey F9)."""
        self._opaque_mode = not self._opaque_mode
        for entry in self._entries.values():
            entry.label.set_opaque_mode(self._opaque_mode)
        logger.info(
            "Overlay: sfondo label %s",
            "opaco" if self._opaque_mode else "semi-trasparente",
        )

    # ── Update da pipeline ────────────────────────────────────────

    def _on_update(self, regions: list) -> None:
        """
        Riceve list[(bbox, translated_text)] dal bridge thread della
        pipeline e aggiorna le TranslationLabel.

        Comportamento:
          - Le label rimangono a schermo finché il frame non cambia.
          - Una label viene rimossa (con fade-out) solo quando arriva
            un nuovo aggiornamento che non la include: questo avviene
            perché l'OCR ha rilevato che il testo è scomparso dallo schermo
            (il text-delta in OCRWorker emette solo quando il set cambia).
          - Nessun timeout temporale: la durata della label è determinata
            esclusivamente dai dati della pipeline, non dal clock.
        """
        if not self._overlay_visible:
            return

        seen_keys: set[tuple[int, int]] = set()

        for bbox, translated_text in regions:
            key = _bbox_key(bbox)
            seen_keys.add(key)

            if key in self._entries:
                # ── Label esistente → aggiorna ─────────────────────
                entry = self._entries[key]
                entry.fading_out = False

                smoothed = entry.smoother.update(bbox)
                x, y, w, h = _bbox_rect(smoothed)
                x, y, w, h = self._apply_dpi(x, y, w, h)
                entry.label.update_position(x, y, w, h)
                entry.label.update_text(translated_text)

                # Se stava facendo fade-out, riportala a piena visibilità
                if entry.label._opacity_effect.opacity() < 0.95:
                    entry.label.fade_in(config.overlay_fade_duration_ms)
            else:
                # ── Nuova label ────────────────────────────────────
                smoother = BBoxSmoother(window=config.overlay_smoothing_frames)
                smoothed = smoother.update(bbox)
                x, y, w, h = _bbox_rect(smoothed)
                x, y, w, h = self._apply_dpi(x, y, w, h)

                label = TranslationLabel(
                    text=translated_text,
                    parent=self,
                    bg_alpha=config.overlay_bg_alpha,
                    opaque_mode=self._opaque_mode,
                )
                label.update_position(x, y, w, h)
                label.fade_in(config.overlay_fade_duration_ms)

                self._entries[key] = _LabelEntry(label=label, smoother=smoother)

        # ── Rimuovi label non più presenti nell'aggiornamento corrente ──
        # Questo avviene solo se l'OCR ha rilevato un cambiamento nel frame
        # (text-delta). Le label assenti iniziano il fade-out ora.
        for key in list(self._entries):
            if key not in seen_keys:
                entry = self._entries[key]
                if not entry.fading_out:
                    entry.fading_out = True
                    entry.label.fade_out(
                        config.overlay_fade_duration_ms,
                        on_finished=lambda k=key: self._remove_entry(k),
                    )

    # ── Rimozione label ───────────────────────────────────────────

    def _remove_entry(self, key: tuple[int, int]) -> None:
        """Rimuove definitivamente una label dal pool dopo il fade-out."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            entry.label.deleteLater()
            entry.smoother.reset()

    # ── DPI compensation ──────────────────────────────────────────

    def _apply_dpi(self, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        """
        Compensa le coordinate per il fattore di scala DPI del monitor.

        Le coordinate bbox dall'OCR sono in pixel fisici dello schermo,
        ma Qt lavora in coordinate logiche quando DPI-awareness è attiva.
        Dividiamo per il scale factor per riallinearle.
        """
        if self._dpi_scale == 1.0:
            return x, y, w, h
        s = self._dpi_scale
        return (
            int(x / s),
            int(y / s),
            int(w / s),
            int(h / s),
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Pulizia alla chiusura della finestra."""
        self._unregister_hotkeys()

        # Rimuovi tutte le label
        for entry in self._entries.values():
            entry.label.deleteLater()
        self._entries.clear()

        logger.info("OverlayWindow chiusa")
        super().closeEvent(event)

    def clear_all_labels(self) -> None:
        """Rimuove immediatamente tutte le label (utile su stop pipeline)."""
        for entry in self._entries.values():
            entry.label.deleteLater()
        self._entries.clear()