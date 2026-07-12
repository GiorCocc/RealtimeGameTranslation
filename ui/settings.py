"""
ui/settings.py — Phase 6: Pannello Impostazioni

Responsabilità:
  - Dialog PyQt6 per configurare l'applicazione senza toccare il codice
  - Modifica il singleton `config` a runtime
  - Selezione lingua sorgente/destinazione (con verifica/download modello)
  - Selezione monitor OPPURE finestra specifica (stile OBS "Window Capture")
  - Configurazione soglie OCR, FPS, hotkey, aspetto overlay
  - Pannello risorse: CPU/RAM correnti (psutil) + stima impatto impostazioni
  - Salvataggio/caricamento configurazione su JSON (settings.json)

Uso tipico (da main.py, sempre nel thread Qt):
    from ui.settings import SettingsDialog, load_settings, save_settings
    from PyQt6.QtWidgets import QDialog

    load_settings()   # applica eventuale settings.json salvato in precedenza

    dialog = SettingsDialog(parent=None)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        if dialog.requires_restart:
            pipeline.restart()   # nuova lingua / monitor / finestra / fps
        if dialog.hotkeys_changed:
            overlay.apply_hotkey_changes()
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import config

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path("settings.json")


class HotkeyInput(QLineEdit):
    """
    Widget personalizzato per l'acquisizione sicura degli hotkey.
    Intercetta le combinazioni di tasti senza permettere la normale digitazione.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Premi combinazione...")

    def keyPressEvent(self, event) -> None:
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return

        if key == Qt.Key.Key_Tab:
            super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Backspace or key == Qt.Key.Key_Delete:
            self.clear()
            return

        # QKeySequence(modifiers | key) a volte ha problemi con i modificatori speciali, 
        # costruiamo la stringa manualmente per compatibilità con il modulo `keyboard`
        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append("win")

        key_name = ""
        # Mappa per i tasti principali
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            key_name = chr(key).lower()
        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            key_name = chr(key)
        elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
            key_name = f"f{key - Qt.Key.Key_F1 + 1}"
        else:
            # Fallback a stringa Qt (a volte potrebbe non combaciare col modulo keyboard)
            key_seq_str = QKeySequence(key).toString().lower()
            if key_seq_str:
                key_name = key_seq_str

        if key_name:
            parts.append(key_name)
            self.setText("+".join(parts))


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


# ══════════════════════════════════════════════════════════════════
# Settings Dialog
# ══════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    """
    Dialog modale per le impostazioni dell'applicazione (QDialog + QTabWidget).

    Tab:
      1. Traduzione — coppia di lingue, stato/download modello MarianMT
      2. Cattura    — monitor intero o finestra specifica (stile OBS), FPS
      3. OCR        — soglia confidenza, throttle, lunghezza minima, downscale
      4. Overlay    — opacità, font, smoothing, fade, hotkey
      5. Risorse    — CPU/RAM correnti + stima impatto delle impostazioni

    Dopo `exec() == QDialog.DialogCode.Accepted`, i nuovi valori sono già
    stati scritti nel singleton `config` e salvati su disco. Il chiamante
    deve solo controllare:
      - `requires_restart`  → chiamare pipeline.restart()
      - `hotkeys_changed`   → chiamare overlay.apply_hotkey_changes()
    """

    # Emesso dal thread di download del modello, ricevuto nel thread Qt
    # (le connessioni cross-thread di PyQt sono automaticamente accodate
    # nell'event loop del thread proprietario del ricevente — qui il thread Qt).
    model_status_signal = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Impostazioni — Game Translator")
        self.resize(560, 480)

        self.requires_restart: bool = False
        self.hotkeys_changed: bool = False

        self.model_status_signal.connect(self._on_model_status_update)

        self._build_ui()
        self._load_from_config()

    # ── Costruzione UI ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_translation_tab(), "Traduzione")
        self.tabs.addTab(self._build_capture_tab(), "Cattura")
        self.tabs.addTab(self._build_ocr_tab(), "OCR")
        self.tabs.addTab(self._build_overlay_tab(), "Overlay")
        self.tabs.addTab(self._build_resources_tab(), "Risorse")

        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Ripristina default")
        self.btn_reset.clicked.connect(self._on_reset_defaults)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

    # ── Tab 1: Traduzione ────────────────────────────────────────────

    def _build_translation_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.combo_lang_pair = QComboBox()
        for src, tgt, label in SUPPORTED_LANGUAGE_PAIRS:
            self.combo_lang_pair.addItem(label, (src, tgt))
        self.combo_lang_pair.currentIndexChanged.connect(self._refresh_model_status)
        form.addRow("Coppia di lingue:", self.combo_lang_pair)

        self.label_model_status = QLabel("—")
        self.label_model_status.setWordWrap(True)
        form.addRow("Stato modello:", self.label_model_status)

        self.btn_check_model = QPushButton("Verifica / Scarica modello")
        self.btn_check_model.clicked.connect(self._on_check_model)
        form.addRow("", self.btn_check_model)

        note = QLabel(
            "Il download richiede una connessione a Internet solo la prima "
            "volta per ogni coppia di lingue. Una volta scaricato, il "
            "modello (~300MB) resta in models/ e l'app funziona interamente "
            "offline (vedere KNOWLEDGE_BASE.md §13)."
        )
        note.setWordWrap(True)
        form.addRow(note)

        return w

    def _refresh_model_status(self) -> None:
        src, tgt = self.combo_lang_pair.currentData()
        local_path = config.models_dir / f"opus-mt-{src}-{tgt}-ct2"
        if local_path.exists():
            self.label_model_status.setText(f"Presente in locale ({local_path})")
        else:
            self.label_model_status.setText(
                "Non ancora scaricato — verrà scaricato al primo avvio "
                "della pipeline, oppure subito con il pulsante qui sotto."
            )

    def _on_check_model(self) -> None:
        src, tgt = self.combo_lang_pair.currentData()
        self.btn_check_model.setEnabled(False)
        self.label_model_status.setText("Verifica/download in corso...")
        threading.Thread(
            target=self._download_model_worker, args=(src, tgt), daemon=True
        ).start()

    def _download_model_worker(self, src: str, tgt: str) -> None:
        """Eseguito in un thread separato: scarica/verifica e converte il modello MarianMT per CTranslate2."""
        try:
            local_path = config.models_dir / f"opus-mt-{src}-{tgt}-ct2"
            if local_path.exists():
                self.model_status_signal.emit(f"Già presente in locale: {local_path}")
                return

            from transformers import MarianTokenizer
            import ctranslate2

            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            tokenizer = MarianTokenizer.from_pretrained(model_name)

            self.model_status_signal.emit("Scaricamento e conversione a CTranslate2 in corso...")
            
            # Converte il modello da HuggingFace al formato CT2
            converter = ctranslate2.converters.TransformersConverter(model_name)
            local_path.mkdir(parents=True, exist_ok=True)
            converter.convert(output_dir=str(local_path), force=True)

            # Salva il tokenizer per l'uso offline
            tokenizer.save_pretrained(str(local_path))
            
            self.model_status_signal.emit(f"Scaricato e convertito in: {local_path}")
        except Exception as exc:  # noqa: BLE001 — mostrare comunque un feedback
            logger.exception("Errore durante il download/conversione del modello %s-%s", src, tgt)
            self.model_status_signal.emit(f"Errore durante il download/conversione: {exc}")

    def _on_model_status_update(self, text: str) -> None:
        self.label_model_status.setText(text)
        self.btn_check_model.setEnabled(True)

    # ── Tab 2: Cattura ───────────────────────────────────────────────

    def _build_capture_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        mode_row = QHBoxLayout()
        self.radio_mode_monitor = QRadioButton("Monitor intero")
        self.radio_mode_window = QRadioButton("Finestra specifica (stile OBS)")
        self.mode_group = QButtonGroup(w)
        self.mode_group.addButton(self.radio_mode_monitor)
        self.mode_group.addButton(self.radio_mode_window)
        self.radio_mode_monitor.toggled.connect(self._on_mode_toggled)
        mode_row.addWidget(self.radio_mode_monitor)
        mode_row.addWidget(self.radio_mode_window)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        form = QFormLayout()

        self.combo_monitor = QComboBox()
        self._populate_monitor_combo()
        form.addRow("Monitor:", self.combo_monitor)

        window_row = QHBoxLayout()
        self.combo_window = QComboBox()
        self.combo_window.setEditable(True)
        self.combo_window.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.btn_refresh_windows = QPushButton("Aggiorna elenco")
        self.btn_refresh_windows.clicked.connect(self._refresh_window_list)
        window_row.addWidget(self.combo_window, 1)
        window_row.addWidget(self.btn_refresh_windows)
        form.addRow("Finestra:", window_row)

        window_note = QLabel(
            "In modalità finestra la cattura segue automaticamente la "
            "posizione/il monitor della finestra scelta a ogni avvio. Se la "
            "finestra non viene trovata (chiusa/minimizzata) si ricade "
            "temporaneamente sul monitor intero."
        )
        window_note.setWordWrap(True)
        form.addRow(window_note)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(10, 60)
        self.spin_fps.valueChanged.connect(self._update_impact_estimate)
        form.addRow("FPS cattura:", self.spin_fps)

        self.check_preview = QCheckBox("Mostra preview di debug (OpenCV)")
        form.addRow("", self.check_preview)

        layout.addLayout(form)
        self._refresh_window_list()
        return w

    def _populate_monitor_combo(self) -> None:
        from core.capture import list_monitors

        self.combo_monitor.clear()
        monitors = list_monitors()
        if not monitors:
            self.combo_monitor.addItem("Nessun monitor rilevato", 0)
            return
        for m in monitors:
            label = f"[{m['index']}] {m.get('name', 'Monitor')}  {m['width']}×{m['height']}"
            self.combo_monitor.addItem(label, m["index"])

    def _refresh_window_list(self) -> None:
        from core.capture import list_windows

        current_text = self.combo_window.currentText()
        self.combo_window.clear()
        for win in list_windows():
            self.combo_window.addItem(win["title"])
        if current_text:
            idx = self.combo_window.findText(current_text)
            if idx >= 0:
                self.combo_window.setCurrentIndex(idx)
            else:
                self.combo_window.setEditText(current_text)

    def _on_mode_toggled(self, *_args) -> None:
        is_window = self.radio_mode_window.isChecked()
        self.combo_monitor.setEnabled(not is_window)
        self.combo_window.setEnabled(is_window)
        self.btn_refresh_windows.setEnabled(is_window)

    # ── Tab 3: OCR ───────────────────────────────────────────────────

    def _build_ocr_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.slider_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_confidence.setRange(50, 95)  # rappresenta 0.50 – 0.95
        self.label_confidence_value = QLabel()
        self.slider_confidence.valueChanged.connect(
            lambda v: self.label_confidence_value.setText(f"{v / 100:.2f}")
        )
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.slider_confidence, 1)
        conf_row.addWidget(self.label_confidence_value)
        form.addRow("Soglia confidenza OCR:", conf_row)

        self.spin_throttle = QSpinBox()
        self.spin_throttle.setRange(50, 500)
        self.spin_throttle.setSingleStep(10)
        self.spin_throttle.setSuffix(" ms")
        self.spin_throttle.valueChanged.connect(self._update_impact_estimate)
        form.addRow("Throttle OCR:", self.spin_throttle)

        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(1, 10)
        form.addRow("Lunghezza minima testo:", self.spin_min_len)

        self.check_angle_cls = QCheckBox("Gestione testo ruotato (angle classification)")
        form.addRow("", self.check_angle_cls)

        self.check_downscale = QCheckBox("Downscale prima dell'OCR")
        self.check_downscale.toggled.connect(self._update_impact_estimate)
        form.addRow("", self.check_downscale)

        self.spin_downscale_h = QSpinBox()
        self.spin_downscale_h.setRange(360, 2160)
        self.spin_downscale_h.setSingleStep(60)
        self.spin_downscale_h.setSuffix(" px")
        form.addRow("Altezza downscale:", self.spin_downscale_h)

        return w

    # ── Tab 4: Overlay ───────────────────────────────────────────────

    def _build_overlay_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 255)
        self.label_alpha_value = QLabel()
        self.slider_alpha.valueChanged.connect(
            lambda v: self.label_alpha_value.setText(str(v))
        )
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self.slider_alpha, 1)
        alpha_row.addWidget(self.label_alpha_value)
        form.addRow("Opacità sfondo label:", alpha_row)

        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(8, 36)
        self.spin_font_size.setSuffix(" pt")
        form.addRow("Dimensione font:", self.spin_font_size)

        self.combo_font_family = QComboBox()
        self.combo_font_family.setEditable(True)
        self.combo_font_family.addItems(
            ["Arial", "Segoe UI", "Verdana", "Tahoma", "Calibri", "Consolas"]
        )
        form.addRow("Font:", self.combo_font_family)

        self.spin_smoothing = QSpinBox()
        self.spin_smoothing.setRange(1, 10)
        self.spin_smoothing.setSuffix(" frame")
        form.addRow("Smoothing bounding box:", self.spin_smoothing)

        self.spin_fade = QSpinBox()
        self.spin_fade.setRange(0, 1000)
        self.spin_fade.setSingleStep(50)
        self.spin_fade.setSuffix(" ms")
        form.addRow("Durata fade in/out:", self.spin_fade)

        self.edit_hotkey_toggle = HotkeyInput()
        form.addRow("Hotkey toggle overlay (F10):", self.edit_hotkey_toggle)

        self.edit_hotkey_opaque = HotkeyInput()
        form.addRow("Hotkey toggle opacità (F9):", self.edit_hotkey_opaque)

        return w

    # ── Tab 5: Risorse ───────────────────────────────────────────────

    def _build_resources_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.label_cpu = QLabel("CPU: —")
        self.label_ram = QLabel("RAM: —")
        self.label_gpu = QLabel("GPU: —")
        for lbl in (self.label_cpu, self.label_ram, self.label_gpu):
            layout.addWidget(lbl)

        layout.addSpacing(16)
        self.label_impact = QLabel("Impatto CPU stimato: —")
        self.label_impact.setWordWrap(True)
        layout.addWidget(self.label_impact)
        layout.addStretch(1)

        # Aggiorna CPU/RAM ogni secondo finché il dialog è aperto.
        self._resource_timer = QTimer(self)
        self._resource_timer.timeout.connect(self._update_resources)
        self._resource_timer.start(1000)
        self._update_resources()

        return w

    def _update_resources(self) -> None:
        try:
            import psutil

            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            self.label_cpu.setText(f"CPU: {cpu_pct:.0f}%")
            self.label_ram.setText(
                f"RAM: {vm.used / (1024 ** 3):.1f} / {vm.total / (1024 ** 3):.1f} GB "
                f"({vm.percent:.0f}%)"
            )
        except ImportError:
            self.label_cpu.setText("CPU: psutil non installato (pip install psutil)")
            self.label_ram.setText("RAM: psutil non installato")

        self.label_gpu.setText(self._detect_gpu_status())
        self._update_impact_estimate()

    @staticmethod
    def _detect_gpu_status() -> str:
        """
        A differenza di NVIDIA (nvidia-smi/GPUtil/pynvml), le GPU AMD non
        hanno un equivalente semplice per il monitoring da Python puro senza
        driver/SDK dedicati (ROCm-smi è Linux-only). Qui si rileva solo la
        presenza della scheda via WMI, senza percentuale di utilizzo in
        tempo reale — coerente con l'operatività CPU-only del progetto
        (vedere memoria di progetto: "AMD GPU, no CUDA").
        """
        try:
            import wmi  # type: ignore

            c = wmi.WMI()
            names = [g.Name for g in c.Win32_VideoController() if g.Name]
            if names:
                return "GPU: " + ", ".join(names) + "  (% utilizzo non disponibile su AMD)"
        except Exception:
            pass
        return "GPU: rilevamento non disponibile (nessun modulo WMI installato)"

    def _update_impact_estimate(self) -> None:
        """
        Euristica indicativa (NON una misura reale) per dare un'idea
        dell'impatto CPU delle impostazioni correnti prima di applicarle,
        combinando FPS di cattura, throttle OCR e downscale. Utile per
        capire in anticipo se un cambio di configurazione (es. FPS più
        alti, throttle più basso) rischia di appesantire troppo la CPU
        già occupata da OCR e traduzione.
        """
        fps = self.spin_fps.value() if hasattr(self, "spin_fps") else config.capture_fps
        throttle = (
            self.spin_throttle.value() if hasattr(self, "spin_throttle") else config.ocr_throttle_ms
        )
        downscale_on = (
            self.check_downscale.isChecked() if hasattr(self, "check_downscale") else config.ocr_downscale
        )

        score = 0.0
        score += fps / 30.0
        score += 150.0 / max(throttle, 50)
        score += 0.5 if downscale_on else 1.3

        if score < 1.8:
            level, color = "Basso", "#2e7d32"
        elif score < 2.8:
            level, color = "Medio", "#e08000"
        else:
            level, color = "Alto", "#c62828"

        if not hasattr(self, "label_impact"):
            return  # tab Risorse non ancora costruita

        self.label_impact.setText(
            f"<b>Impatto CPU stimato: <span style='color:{color}'>{level}</span></b><br>"
            f"Stima indicativa basata su FPS cattura, throttle OCR e downscale — "
            f"non una misura reale (dipende da CPU e gioco in esecuzione)."
        )

    # ── Caricamento valori da config ─────────────────────────────────

    def _load_from_config(self) -> None:
        # Traduzione
        idx = self.combo_lang_pair.findData((config.source_language, config.target_language))
        self.combo_lang_pair.setCurrentIndex(idx if idx >= 0 else 0)
        self._refresh_model_status()

        # Cattura
        self.radio_mode_window.setChecked(config.capture_mode == "window")
        self.radio_mode_monitor.setChecked(config.capture_mode != "window")
        m_idx = self.combo_monitor.findData(config.monitor_index)
        if m_idx >= 0:
            self.combo_monitor.setCurrentIndex(m_idx)
        if config.capture_window_title:
            w_idx = self.combo_window.findText(config.capture_window_title)
            if w_idx >= 0:
                self.combo_window.setCurrentIndex(w_idx)
            else:
                self.combo_window.setEditText(config.capture_window_title)
        self.spin_fps.setValue(config.capture_fps)
        self.check_preview.setChecked(config.debug_capture_preview)
        self._on_mode_toggled()

        # OCR
        self.slider_confidence.setValue(int(round(config.ocr_confidence_threshold * 100)))
        self.spin_throttle.setValue(config.ocr_throttle_ms)
        self.spin_min_len.setValue(config.ocr_min_text_length)
        self.check_angle_cls.setChecked(config.ocr_use_angle_cls)
        self.check_downscale.setChecked(config.ocr_downscale)
        self.spin_downscale_h.setValue(config.ocr_downscale_height)

        # Overlay
        self.slider_alpha.setValue(config.overlay_bg_alpha)
        self.spin_font_size.setValue(config.overlay_font_size)
        self.combo_font_family.setCurrentText(config.overlay_font_family)
        self.spin_smoothing.setValue(config.overlay_smoothing_frames)
        self.spin_fade.setValue(config.overlay_fade_duration_ms)
        self.edit_hotkey_toggle.setText(config.hotkey_toggle_overlay)
        self.edit_hotkey_opaque.setText(config.hotkey_toggle_opaque)

        self._update_impact_estimate()

    # ── Ripristino default ────────────────────────────────────────────

    def _on_reset_defaults(self) -> None:
        """
        Ripopola i controlli con i valori di default di `Config()`.
        Non tocca ancora il singleton `config`: l'utente deve comunque
        premere OK per confermare e salvare.
        """
        from config import Config

        defaults = Config()

        idx = self.combo_lang_pair.findData((defaults.source_language, defaults.target_language))
        if idx >= 0:
            self.combo_lang_pair.setCurrentIndex(idx)

        self.radio_mode_monitor.setChecked(defaults.capture_mode != "window")
        self.radio_mode_window.setChecked(defaults.capture_mode == "window")
        m_idx = self.combo_monitor.findData(defaults.monitor_index)
        if m_idx >= 0:
            self.combo_monitor.setCurrentIndex(m_idx)
        self.combo_window.setEditText("")
        self.spin_fps.setValue(defaults.capture_fps)
        self.check_preview.setChecked(defaults.debug_capture_preview)

        self.slider_confidence.setValue(int(round(defaults.ocr_confidence_threshold * 100)))
        self.spin_throttle.setValue(defaults.ocr_throttle_ms)
        self.spin_min_len.setValue(defaults.ocr_min_text_length)
        self.check_angle_cls.setChecked(defaults.ocr_use_angle_cls)
        self.check_downscale.setChecked(defaults.ocr_downscale)
        self.spin_downscale_h.setValue(defaults.ocr_downscale_height)

        self.slider_alpha.setValue(defaults.overlay_bg_alpha)
        self.spin_font_size.setValue(defaults.overlay_font_size)
        self.combo_font_family.setCurrentText(defaults.overlay_font_family)
        self.spin_smoothing.setValue(defaults.overlay_smoothing_frames)
        self.spin_fade.setValue(defaults.overlay_fade_duration_ms)
        self.edit_hotkey_toggle.setText(defaults.hotkey_toggle_overlay)
        self.edit_hotkey_opaque.setText(defaults.hotkey_toggle_opaque)

        self._update_impact_estimate()

    # ── Conferma / applicazione ───────────────────────────────────────

    def _on_ok(self) -> None:
        self.requires_restart = self._apply_settings()
        save_settings()
        self.accept()

    def _apply_settings(self) -> bool:
        """
        Scrive i valori correnti dei widget nel singleton `config`.

        Returns:
            True se è cambiata un'impostazione che richiede il riavvio
            della pipeline (lingua, monitor/finestra/modalità di cattura,
            FPS) — vedere TranslationPipeline.restart(). Le soglie OCR e
            l'aspetto dell'overlay sono invece lette "live" dai rispettivi
            worker a ogni frame, quindi non richiedono restart.
        """
        restart_needed = False

        # ── Traduzione ────────────────────────────────────────────
        src, tgt = self.combo_lang_pair.currentData()
        if (src, tgt) != (config.source_language, config.target_language):
            restart_needed = True
        config.source_language, config.target_language = src, tgt

        # ── Cattura ───────────────────────────────────────────────
        new_mode = "window" if self.radio_mode_window.isChecked() else "monitor"
        new_monitor = self.combo_monitor.currentData()
        new_window_title = self.combo_window.currentText().strip()
        new_fps = self.spin_fps.value()

        if (new_mode, new_monitor, new_window_title, new_fps) != (
            config.capture_mode,
            config.monitor_index,
            config.capture_window_title,
            config.capture_fps,
        ):
            restart_needed = True

        config.capture_mode = new_mode
        if new_monitor is not None:
            config.monitor_index = int(new_monitor)
        config.capture_window_title = new_window_title
        config.capture_fps = new_fps
        config.debug_capture_preview = self.check_preview.isChecked()

        # ── OCR (lette live dai worker: nessun restart necessario) ──
        config.ocr_confidence_threshold = self.slider_confidence.value() / 100.0
        config.ocr_throttle_ms = self.spin_throttle.value()
        config.ocr_min_text_length = self.spin_min_len.value()
        config.ocr_use_angle_cls = self.check_angle_cls.isChecked()
        config.ocr_downscale = self.check_downscale.isChecked()
        config.ocr_downscale_height = self.spin_downscale_h.value()

        # ── Overlay ──────────────────────────────────────────────
        config.overlay_bg_alpha = self.slider_alpha.value()
        config.overlay_font_size = self.spin_font_size.value()
        config.overlay_font_family = self.combo_font_family.currentText().strip() or "Arial"
        config.overlay_smoothing_frames = self.spin_smoothing.value()
        config.overlay_fade_duration_ms = self.spin_fade.value()

        new_hotkey_toggle = self.edit_hotkey_toggle.text().strip() or "f10"
        new_hotkey_opaque = self.edit_hotkey_opaque.text().strip() or "f9"
        if (new_hotkey_toggle, new_hotkey_opaque) != (
            config.hotkey_toggle_overlay,
            config.hotkey_toggle_opaque,
        ):
            self.hotkeys_changed = True
        config.hotkey_toggle_overlay = new_hotkey_toggle
        config.hotkey_toggle_opaque = new_hotkey_opaque

        return restart_needed


# ── Persistenza settings ──────────────────────────────────────────

# Whitelist esplicita dei campi persistiti: le chiavi sconosciute nel JSON
# vengono ignorate in caricamento (compatibilità con versioni future),
# ed eventuali campi "avanzati"/derivati di config non vengono mai scritti
# su disco (es. models_dir, che dipende da ROOT_DIR e non ha senso serializzare).
_PERSISTED_FIELDS: list[str] = [
    "source_language", "target_language",
    "capture_mode", "capture_window_title", "monitor_index", "capture_fps",
    "debug_capture_preview",
    "ocr_confidence_threshold", "ocr_throttle_ms", "ocr_min_text_length",
    "ocr_use_angle_cls", "ocr_downscale", "ocr_downscale_height",
    "overlay_bg_alpha", "overlay_font_size", "overlay_font_family",
    "overlay_smoothing_frames", "overlay_fade_duration_ms",
    "hotkey_toggle_overlay", "hotkey_toggle_opaque",
]


def save_settings() -> None:
    """Salva la configurazione corrente su settings.json."""
    data = {name: getattr(config, name) for name in _PERSISTED_FIELDS}
    if config.capture_region is not None:
        data["capture_region"] = list(config.capture_region)

    try:
        SETTINGS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Settings salvati in %s", SETTINGS_FILE)
    except OSError:
        logger.exception("Errore durante il salvataggio di %s", SETTINGS_FILE)


def load_settings() -> None:
    """
    Carica settings.json e aggiorna il singleton config.

    Ignora chiavi sconosciute per compatibilità con versioni future
    (whitelist esplicita in _PERSISTED_FIELDS).
    """
    if not SETTINGS_FILE.exists():
        return

    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Errore durante la lettura di %s — ignorato", SETTINGS_FILE)
        return

    for name in _PERSISTED_FIELDS:
        if name in data:
            setattr(config, name, data[name])

    if data.get("capture_region"):
        config.capture_region = tuple(data["capture_region"])

    logger.info("Settings caricati da %s", SETTINGS_FILE)