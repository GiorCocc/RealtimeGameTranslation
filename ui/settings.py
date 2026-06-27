"""
ui/settings.py — Phase 6: Pannello Impostazioni

Responsabilità:
  - Dialog PyQt6 per configurare l'applicazione senza toccare il codice
  - Modifica il singleton `config` a runtime
  - Selezione lingua sorgente/destinazione (con caricamento modello)
  - Selezione monitor
  - Configurazione soglie OCR, FPS, hotkey
  - Salvataggio/caricamento configurazione su JSON

TODO Phase 6: implementare dopo che il sistema è funzionante end-to-end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path("settings.json")

# Coppie di lingue supportate (modelli MarianMT disponibili su HuggingFace)
SUPPORTED_LANGUAGE_PAIRS: list[tuple[str, str, str]] = [
    # (source, target, display_label)
    ("en", "it", "Inglese → Italiano"),
    ("ja", "en", "Giapponese → Inglese"),
    ("zh", "en", "Cinese → Inglese"),
    ("ko", "en", "Coreano → Inglese"),
    ("fr", "en", "Francese → Inglese"),
    ("de", "en", "Tedesco → Inglese"),
    ("es", "en", "Spagnolo → Inglese"),
    ("ru", "en", "Russo → Inglese"),
]


class SettingsDialog:
    """
    TODO Phase 6 — Dialog modale per le impostazioni dell'applicazione.

    Struttura suggerita (QDialog con QTabWidget):

    Tab 1 — Traduzione:
      - ComboBox coppia di lingue
      - Bottone "Scarica modello" con progress bar
      - Label stato modello (presente/assente/in download)

    Tab 2 — Cattura:
      - ComboBox selezione monitor
      - Slider FPS (10-60)
      - Checkbox modalità debug (preview capture)

    Tab 3 — OCR:
      - Slider soglia confidenza (0.5-0.95)
      - Spinbox throttle ms (50-500)
      - Spinbox lunghezza minima testo (1-10)
      - Checkbox angle classification

    Tab 4 — Overlay:
      - Slider opacità sfondo label
      - Spinbox font size
      - Spinbox smoothing frames
      - Hotkey editor (F9/F10 configurabili)

    Bottoni: OK (salva + applica), Annulla, Ripristina default
    """

    def __init__(self, parent=None) -> None:
        raise NotImplementedError("Phase 6: implementare come QDialog")

    def exec(self) -> bool:
        """TODO Phase 6 — Mostra il dialog e restituisce True se l'utente ha cliccato OK."""
        raise NotImplementedError("Phase 6")

    def _apply_settings(self) -> None:
        """TODO Phase 6 — Applica le impostazioni al singleton config."""
        raise NotImplementedError("Phase 6")


# ── Persistenza settings ──────────────────────────────────────────

def save_settings() -> None:
    """
    TODO Phase 6 — Salva la configurazione corrente su settings.json.

    Esempio:
        data = {
            "source_language": config.source_language,
            "target_language": config.target_language,
            "monitor_index": config.monitor_index,
            "capture_fps": config.capture_fps,
            "ocr_confidence_threshold": config.ocr_confidence_threshold,
            ...
        }
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    """
    raise NotImplementedError("Phase 6")


def load_settings() -> None:
    """
    TODO Phase 6 — Carica settings.json e aggiorna il singleton config.

    Ignorare chiavi sconosciute per compatibilità con versioni future.
    """
    if not SETTINGS_FILE.exists():
        return
    raise NotImplementedError("Phase 6")
