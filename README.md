# Game Translation Overlay

Overlay in tempo reale per tradurre automaticamente qualsiasi testo su schermo
durante il gaming — completamente offline, senza configurazione per-gioco.

```
[Schermo di gioco]
       │
       ▼
  CaptureThread  (dxcam, ~30fps)
       │
       ▼
   OCRWorker     (PaddleOCR + EasyOCR fallback)
       │
       ▼
 TranslationWorker  (MarianMT locale, LRU cache)
       │
       ▼
  OverlayWindow  (PyQt6, trasparente, click-through)
```

## Requisiti

- Windows 10/11
- Python 3.11+
- (Raccomandato) GPU NVIDIA con CUDA per latenza ridotta

## Installazione

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU CUDA:** sostituire `paddlepaddle` con `paddlepaddle-gpu`
> e installare il torch CUDA-enabled da [pytorch.org](https://pytorch.org).

## Avvio

```bash
# Phase 1: test capture pipeline con preview
python main.py --phase 1

# Monitor secondario, senza preview
python main.py --phase 1 --monitor 1 --no-preview

# Applicazione completa (Phase 4+)
python main.py
```

## Stato sviluppo

| Fase | Modulo | Stato |
|------|--------|-------|
| 1 — Capture Pipeline | `core/capture.py` | ✅ Implementata |
| 2 — OCR | `core/ocr.py` | ⬜ TODO |
| 3 — Traduzione locale | `core/translator.py` | ⬜ TODO |
| 4 — Overlay PyQt6 | `ui/overlay.py` | ⬜ TODO |
| 5 — Ottimizzazione | vari | ⬜ TODO |
| 6 — Config Panel & UX | `ui/settings.py`, `ui/tray.py` | ⬜ TODO |

## Hotkey (Phase 4+)

| Tasto | Azione |
|-------|--------|
| `F10` | Toggle overlay on/off |
| `F9` | Toggle sfondo label semi-trasparente/opaco |

## Struttura progetto

```
game-translator/
├── main.py          # Entry point + Phase 1 test runner
├── config.py        # Configurazione globale (singleton)
├── core/
│   ├── capture.py   # ✅ dxcam wrapper + frame differencing
│   ├── ocr.py       # ⬜ PaddleOCR + EasyOCR fallback (Phase 2)
│   ├── translator.py# ⬜ MarianMT + LRU cache (Phase 3)
│   └── pipeline.py  # ⬜ Orchestratore thread (Phase 4)
├── ui/
│   ├── overlay.py   # ⬜ Finestra trasparente PyQt6 (Phase 4)
│   ├── settings.py  # ⬜ Pannello impostazioni (Phase 6)
│   └── tray.py      # ⬜ System tray icon (Phase 6)
├── models/          # Modelli MarianMT (scaricati al primo avvio)
└── requirements.txt
```

## Note tecniche

- **OCR:** PaddleOCR analizza l'intero schermo senza zone predefinite.
  EasyOCR è usato come fallback per font irregolari e testi su texture.
- **Traduzione:** Helsinki-NLP MarianMT (~300MB per coppia di lingue),
  caricato all'avvio e mantenuto in RAM durante la sessione.
- **Frame differencing:** griglia 4×4, hash MD5 per cella.
  Frame identici al precedente vengono scartati senza passare all'OCR.
- **Click-through:** `WS_EX_TRANSPARENT` via win32api — l'overlay
  non intercetta mai mouse né tastiera.
