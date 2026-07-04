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
from typing import Optional

# Root del progetto (cartella che contiene questo file)
ROOT_DIR = Path(__file__).parent


@dataclass
class Config:
    # ── Capture ───────────────────────────────────────────────────
    monitor_index: int = 0          # Indice monitor da catturare (0 = primario)
    capture_fps: int = 30           # Target FPS per dxcam

    # ── Capture: modalità (Phase 6) ────────────────────────────────
    # "monitor" → cattura l'intero monitor indicato da `monitor_index`
    # "window"  → cattura solo il rettangolo della finestra il cui titolo
    #             contiene `capture_window_title` (stile OBS "Window Capture").
    #             Il monitor su cui si trova la finestra viene rilevato
    #             automaticamente ad ogni avvio/riavvio della pipeline;
    #             `monitor_index` viene ignorato in questa modalità.
    capture_mode: str = "monitor"          # "monitor" | "window"
    capture_window_title: str = ""         # Sottostringa (case-insensitive) del titolo finestra
    # Regione esplicita (left, top, right, bottom) in coordinate assolute
    # dello schermo. Se impostata ha priorità sulla modalità "window"/"monitor"
    # per catturare una porzione custom (es. solo l'area di gioco in una
    # finestra con bordi/HUD dell'OS). None = nessuna regione custom.
    capture_region: Optional[tuple] = None

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

    # Merge bounding box adiacenti: PaddleOCR isola spesso singole parole
    # in box separate (es. "Press" e "Start" invece di "Press Start"),
    # rompendo il contesto per MarianMT e producendo traduzioni prive di
    # senso. Queste soglie sono espresse in multipli dell'altezza media
    # delle due box confrontate ("em"), non in pixel assoluti, così da
    # restare coerenti indipendentemente dalla risoluzione/downscale.
    ocr_merge_adjacent_boxes: bool = True
    ocr_merge_max_gap_em: float = 0.6              # Gap orizzontale massimo per unire due box sulla stessa riga
    ocr_merge_max_vertical_offset_em: float = 0.4  # Tolleranza verticale per considerare due box sulla stessa riga
    # Merge verticale (multi-riga): unisce righe impilate verticalmente con
    # simile estensione orizzontale. Fondamentale per le visual novel, dove
    # il dialog box ha 2-3 righe di testo che vanno tradotte come un blocco.
    ocr_merge_vertical_blocks: bool = True
    ocr_merge_max_line_gap_em: float = 1.2         # Gap verticale massimo tra righe da unire (in em)
    # Soglia scene change: se almeno questa frazione delle celle della griglia
    # MD5 è cambiata, si assume un cambio scena completo e il text-delta
    # dell'OCR viene resettato per forzare un nuovo emit (evita label
    # "fantasma" della schermata precedente).
    ocr_scene_change_threshold: float = 0.75       # Frazione celle cambiate per rilevare un scene change

    # ── Traduzione ────────────────────────────────────────────────
    source_language: str = "en"     # Lingua sorgente (codice ISO 639-1)
    target_language: str = "it"     # Lingua destinazione
    translation_cache_maxsize: int = 1024   # Dimensione LRU cache frasi tradotte
    models_dir: Path = field(default_factory=lambda: ROOT_DIR / "models")
    # Parametri di generazione MarianMT. I modelli Helsinki-NLP/opus-mt-*
    # di default possono usare beam search (num_beams del model config,
    # spesso 4-6): su CPU questo può moltiplicare i tempi di inferenza
    # di diverse volte rispetto a una ricerca greedy. Per testo di gioco
    # (dialoghi/HUD/menu, tipicamente brevi) la perdita di qualità della
    # ricerca greedy è generalmente trascurabile rispetto al guadagno
    # di latenza. Vedere KNOWLEDGE_BASE.md §5 (throttling/latenza).
    translation_num_beams: int = 1          # 1 = greedy (veloce), >1 = beam search
    translation_max_new_tokens: int = 128   # Limite lunghezza traduzione generata

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