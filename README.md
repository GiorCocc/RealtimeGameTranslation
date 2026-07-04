# RealtimeGameTranslation

Real-time OCR + translation overlay for games on Windows.  
The application captures the screen, detects text, translates it, and renders translated text in a transparent click-through overlay.

## ✨ Core Features

- **Real-time screen capture** via `dxcam` wrapper
- **Smart frame differencing** (4×4 grid + MD5 hashes) to skip unchanged frames
- **OCR pipeline**:
  - Primary: **PaddleOCR**
  - Fallback: **EasyOCR** for irregular fonts/textured backgrounds
- **On-device translation** with **Helsinki-NLP MarianMT**
- **Translation cache** with LRU strategy for lower latency
- **Multi-threaded pipeline orchestration**
- **Transparent click-through overlay** (PyQt6 + `WS_EX_TRANSPARENT`)
- **DPI-aware coordinate handling** (Per-Monitor V2)
- **Global hotkeys**:
  - `F10` → toggle overlay
  - `F9` → toggle opaque/transparent background

---

## 🧠 How it Works (Pipeline)

1. **Capture** current frame from selected monitor
2. **Compare** with previous frame using cell hashes
3. If changed, run **OCR** on full frame
4. Send detected text to **translator**
5. Render translated bounding boxes in **overlay**

This architecture minimizes unnecessary OCR work and improves end-to-end responsiveness.

---

## 📁 Project Structure

```text
RealtimeGameTranslation/
├─ main.py                 # Entry point + phase test runner
├─ config.py               # Global configuration (singleton)
├─ core/
│  ├─ capture.py           # ✅ dxcam wrapper + frame differencing
│  ├─ ocr.py               # ✅ PaddleOCR + EasyOCR fallback (Phase 2)
│  ├─ translator.py        # ✅ MarianMT + LRU cache (Phase 3)
│  └─ pipeline.py          # ✅ Multi-threaded pipeline orchestration (Phase 4)
└─ ui/
   ├─ overlay.py           # ✅ Transparent click-through PyQt6 overlay (Phase 4)
   ├─ settings.py          # ⬜ Settings panel (planned / Phase 6)
   └─ tray.py              # ⬜ System tray integration (planned / Phase 6)
```

---

## 🚀 Run Commands

### Phase-based tests

- `python main.py --phase 1`  
  Capture pipeline test with preview

- `python main.py --phase 1 --monitor 1 --no-preview`  
  Secondary monitor capture test without preview

- `python main.py --phase 2`  
  OCR test with bounding-box preview

- `python main.py --phase 3`  
  End-to-end translation test

### Full application

- `python main.py`  
  Start complete app with overlay (Phase 4)

- `python main.py --phase 4`  
  Explicit alias for full app start

- `python main.py --debug`  
  Full app with DEBUG logging enabled

---

## ⚙️ Technical Notes

### OCR behavior
- OCR scans the **entire screen** (no predefined zones).
- EasyOCR fallback is used for difficult visual contexts (e.g., stylized fonts, textured UI).

### Translation model
- Uses **Helsinki-NLP MarianMT** models.
- Approx. **~300MB per language pair**.
- Model is loaded at startup and kept in RAM during the session.

### Frame differencing
- Frame is partitioned into a **4×4 grid**.
- Each cell is hashed using **MD5**.
- If all hashes match previous frame, OCR is skipped.

### Overlay input behavior
- Click-through mode enabled via `WS_EX_TRANSPARENT` (`win32api`).
- Overlay does not intercept mouse/keyboard input intended for the game.

### DPI handling
- **Per-Monitor V2 DPI awareness** is enabled at startup.
- Bounding-box coordinates are compensated according to monitor scale factor.

### Hotkeys
- `F10`: Toggle overlay visibility
- `F9`: Toggle opaque/transparent overlay background

---

## 🗺️ Roadmap Snapshot

- ✅ Capture and frame differencing
- ✅ OCR with fallback strategy
- ✅ Translation with caching
- ✅ End-to-end threaded pipeline + overlay
- ⬜ Settings panel
- ⬜ System tray controls

---

## 📌 Scope

Current implementation is focused on **Python + Windows desktop overlay workflow** for game text translation in real time.

---

## 🧪 Suggested Workflow for Development

1. Validate capture with `--phase 1`
2. Validate OCR boxes with `--phase 2`
3. Validate translation with `--phase 3`
4. Run full integration via `--phase 4` (or no phase argument)

---

## 📝 License

Add your preferred license (e.g., MIT) to clarify usage and contributions.
