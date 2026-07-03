# AGENTS.md

## Comandi

- `python main.py --phase 1` - Test della pipeline di cattura con preview
- `python main.py --phase 1 --monitor 1 --no-preview` - Monitor secondario senza preview
- `python main.py --phase 2` - Test OCR con preview bounding box
- `python main.py --phase 3` - Test traduzione end-to-end
- `python main.py` - Applicazione completa con overlay (Phase 4)
- `python main.py --phase 4` - Alias esplicito per avvio completo
- `python main.py --debug` - Avvio completo con logging DEBUG

## Struttura progetto

- `main.py` - Entry point + test runner per tutte le fasi
- `config.py` - Configurazione globale (singleton)
- `core/capture.py` - ✅ dxcam wrapper + frame differencing
- `core/ocr.py` - ✅ PaddleOCR + EasyOCR fallback (Phase 2)
- `core/translator.py` - ✅ MarianMT + LRU cache (Phase 3)
- `core/pipeline.py` - ✅ Orchestrazione pipeline multi-thread (Phase 4)
- `ui/overlay.py` - ✅ Finestra trasparente PyQt6 click-through (Phase 4)
- `ui/settings.py`, `ui/tray.py` - ⬜ Pannello impostazioni e System tray icon (Phase 6)

## Note tecniche

- **OCR:** PaddleOCR analizza l'intero schermo senza zone predefinite. EasyOCR è usato come fallback per font irregolari e testi su texture.
- **Traduzione:** Helsinki-NLP MarianMT (~300MB per coppia di lingue), caricato all'avvio e mantenuto in RAM durante la sessione.
- **Frame differencing:** griglia 4×4, hash MD5 per cella. Frame identici al precedente vengono scartati senza passare all'OCR.
- **Click-through:** `WS_EX_TRANSPARENT` via win32api — l'overlay non intercetta mai mouse né tastiera.
- **DPI-awareness:** Per-Monitor V2 abilitata automaticamente all'avvio; le coordinate bbox vengono compensate per il fattore di scala DPI.
- **Hotkey globali:** F10 = toggle overlay, F9 = toggle sfondo opaco/trasparente (via libreria `keyboard`).