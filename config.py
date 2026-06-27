"""
config.py — Configurazione globale dell'applicazione.

Tutti i moduli importano il singleton `config` da qui.
I valori possono essere sovrascritti a runtime dal pannello Settings (Phase 6)
o da argomenti CLI (main.py).

Esempio:
    from config import config
    print(config.capture_fps)
    config.target_language = "de"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Root del progetto (cartella che contiene questo file)
ROOT_DIR = Path(__file__).parent


@dataclass
class Config:
    # ── Capture ───────────────────────────────────────────────────
    monitor_index: int = 0          # Indice monitor da catturare (0 = primario)
    capture_fps: int = 30           # Target FPS per dxcam

    # ── Frame differencing ────────────────────────────────────────
    # Lo schermo viene diviso in griglia rows×cols.
    # Solo le celle cambiate vengono passate all'OCR (ottimizzazione Phase 5).
    frame_diff_grid_rows: int = 4
    frame_diff_grid_cols: int = 4

    # ── OCR ───────────────────────────────────────────────────────
    ocr_confidence_threshold: float = 0.7   # Scarta rilevamenti sotto soglia
    ocr_min_text_length: int = 3            # Ignora testi < 3 caratteri (falsi pos.)
    ocr_throttle_ms: int = 150              # Intervallo minimo tra run OCR
    ocr_use_angle_cls: bool = True          # Gestione testo ruotato (PaddleOCR)
    ocr_language: str = "en"               # Lingua sorgente per PaddleOCR
    # Downscale: riduce il frame a max ocr_downscale_height prima dell'OCR
    # dimezza il carico senza perdere leggibilità per i modelli OCR
    ocr_downscale: bool = True
    ocr_downscale_height: int = 720

    # ── Traduzione ────────────────────────────────────────────────
    source_language: str = "en"     # Lingua sorgente (codice ISO 639-1)
    target_language: str = "it"     # Lingua destinazione
    translation_cache_maxsize: int = 1024   # Dimensione LRU cache frasi tradotte
    models_dir: Path = field(default_factory=lambda: ROOT_DIR / "models")

    # ── Overlay ───────────────────────────────────────────────────
    overlay_bg_alpha: int = 160             # Opacità sfondo label (0–255)
    overlay_font_size: int = 14             # pt
    overlay_font_family: str = "Arial"
    overlay_smoothing_frames: int = 4       # Media mobile coordinate bbox
    overlay_fade_duration_ms: int = 150     # Durata animazione fade in/out

    # ── Hotkey ────────────────────────────────────────────────────
    hotkey_toggle_overlay: str = "f10"      # On/off overlay completo
    hotkey_toggle_opaque: str = "f9"        # Toggle label semi-trasparente/opaca

    # ── Debug / sviluppo ──────────────────────────────────────────
    debug_show_fps: bool = False            # Mostra FPS nell'overlay
    debug_show_bboxes: bool = False         # Disegna bounding box OCR
    debug_capture_preview: bool = False     # Finestra OpenCV con preview cattura

    def __post_init__(self) -> None:
        # Assicura che la cartella modelli esista
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def marian_model_name(self) -> str:
        """Nome HuggingFace del modello MarianMT per la coppia di lingue attuale."""
        return f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"

    @property
    def marian_model_local_path(self) -> Path:
        """Percorso locale dove il modello viene salvato/caricato."""
        return self.models_dir / f"opus-mt-{self.source_language}-{self.target_language}"


# ── Singleton ─────────────────────────────────────────────────────
# Unica istanza usata da tutta l'applicazione.
config = Config()
