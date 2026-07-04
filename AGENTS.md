# AGENTS.md

## Comandi

- `python main.py --phase 1` - Test della pipeline di cattura con preview
- `python main.py --phase 1 --monitor 1 --no-preview` - Monitor secondario senza preview
- `python main.py --phase 2` - Test OCR con preview bounding box
- `python main.py --phase 3` - Test traduzione end-to-end
- `python main.py --phase 5` - Test benchmark e profiling delle prestazioni (Phase 5)
- `python main.py` - Applicazione completa con overlay (Phase 4)
- `python main.py --phase 4` - Alias esplicito per avvio completo
- `python main.py --debug` - Avvio completo con logging DEBUG

## Struttura progetto

- `main.py` - Entry point + test runner per tutte le fasi
- `config.py` - Configurazione globale (singleton)
- `core/gpu_utils.py` - ✅ Gestione GPU, detection CUDA e monitoraggio VRAM
- `core/capture.py` - ✅ dxcam wrapper + fast MAD differencing (Phase 1 e 5)
- `core/ocr.py` - ✅ PaddleOCR + lazy EasyOCR fallback con ROI incrementale (Phase 2 e 5)
- `core/translator.py` - ✅ MarianMT + FP16 + OrderedDict LRU cache (Phase 3 e 5)
- `core/pipeline.py` - ✅ Orchestrazione pipeline + warm-up silenzioso (Phase 4 e 5)
- `ui/overlay.py` - ✅ Finestra trasparente PyQt6 click-through (Phase 4)
- `ui/settings.py`, `ui/tray.py` - ✅ Pannello impostazioni, risorse e System tray icon (Phase 6)

## Note tecniche

- **OCR:** PaddleOCR analizza solo le celle della griglia che sono cambiate rispetto al frame precedente (ROI incrementale). EasyOCR viene caricato in modalità lazy (on-demand) come fallback.
- **Traduzione:** Helsinki-NLP MarianMT (~300MB per coppia di lingue), caricato in modalità FP16 su GPU CUDA per raddoppiare la velocità e dimezzare l'impatto VRAM. Supporta `torch.compile()` opt-in.
- **Cache:** LRU cache implementata tramite `collections.OrderedDict` per inserimento ed espulsione O(1).
- **Frame differencing:** Griglia 4×4 con confronto MAD (Mean Absolute Difference) numpy 10x più veloce su CPU rispetto ad MD5.
- **Process & Thread Priority:** Priorità di processo Windows e del thread OCR impostate a `BELOW_NORMAL` per non interferire con giochi e OBS. GC thresholds ottimizzati.
- **Click-through:** `WS_EX_TRANSPARENT` via win32api — l'overlay non intercetta mai mouse né tastiera.
- **DPI-awareness:** Per-Monitor V2 abilitata automaticamente all'avvio; le coordinate bbox vengono compensate per il fattore di scala DPI.
- **Hotkey globali:** F10 = toggle overlay, F9 = toggle sfondo opaco/trasparente (via libreria `keyboard`).