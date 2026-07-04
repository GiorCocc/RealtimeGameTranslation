# RealtimeGameTranslation

Real-time OCR + translation overlay for games on Windows.  
The application captures the screen, detects text, translates it, and renders translated text in a transparent click-through overlay.

## ✨ Core Features

- **Real-time screen capture** via `dxcam` wrapper (DXGI Desktop Duplication API)
- **Performance Optimized Pipeline** (Phase 5):
  - **Fast Frame Differencing**: MAD (Mean Absolute Difference) numpy-based cell comparison (10x faster than MD5).
  - **Incremental ROI OCR**: Only processes changed grid cells, reducing OCR load by 60-80%.
  - **Lazy Loading**: Fallback engines (EasyOCR) are loaded on-demand, saving ~500MB RAM.
  - **FP16 Inference**: MarianMT on GPU CUDA runs in half-precision, doubling speed and halving VRAM.
  - **Adaptive FPS & Downscale**: Auto-throttling under heavy CPU/GPU load (OBS + gaming).
- **OCR pipeline**:
  - Primary: **PaddleOCR** (optimized mobile-det + en-rec models pinned)
  - Fallback: **EasyOCR** for irregular fonts/textured backgrounds
- **On-device translation** with **Helsinki-NLP MarianMT** (100% offline, local models)
- **Translation cache** with O(1) OrderedDict LRU strategy
- **Multi-threaded pipeline orchestration**
- **Transparent click-through overlay** (PyQt6 + `WS_EX_TRANSPARENT`)
- **DPI-aware coordinate handling** (Per-Monitor V2)
- **Settings Panel & System Tray Controls** (Phase 6)
- **Global hotkeys**:
  - `F10` → toggle overlay
  - `F9` → toggle opaque/transparent background

---

## 🧠 How it Works (Pipeline)

1. **Capture** current frame from selected monitor / window.
2. **Compare** with previous frame using fast numpy-based differencing.
3. If changed, run **incremental OCR** only on the changed regions (grid cells).
4. Send newly detected text to **translator** (checking the LRU cache first).
5. Render translated bounding boxes in **overlay** with smooth fade transitions.

---

## 📁 Project Structure

```text
RealtimeGameTranslation/
├─ main.py                 # Entry point + phase test runner
├─ config.py               # Global configuration (singleton)
├─ core/
│  ├─ capture.py           # ✅ dxcam wrapper + fast MAD differencing
│  ├─ ocr.py               # ✅ PaddleOCR + lazy EasyOCR fallback (ROI-based)
│  ├─ translator.py        # ✅ MarianMT + FP16 + OrderedDict LRU cache
│  ├─ gpu_utils.py         # ✅ GPU detection & VRAM monitoring
│  └─ pipeline.py          # ✅ Multi-threaded pipeline + warm-up sequence
└─ ui/
   ├─ overlay.py           # ✅ Transparent click-through PyQt6 overlay
   ├─ settings.py          # ✅ Settings panel & resource monitor
   └─ tray.py              # ✅ System tray integration
```

---

## 🚀 Run Commands

### Phase-based tests

- `python main.py --phase 1`  
  Capture pipeline test with preview

- `python main.py --phase 2`  
  OCR test with bounding-box preview

- `python main.py --phase 3`  
  End-to-end translation test (headless)

- `python main.py --phase 5`  
  Performance profiling benchmark runner (runs headless for 30s, generates report)

### Full application

- `python main.py` / `python main.py --phase 4`  
  Start complete app with overlay

- `python main.py --debug`  
  Full app with DEBUG logging enabled

---

## ⚙️ Technical Notes & Performance Tuning

- **Windows Process Priority**: Automatically set to `BELOW_NORMAL` to avoid competing with your games and OBS Studio.
- **Garbage Collector Tuning**: GC thresholds adjusted to reduce micro-stuttering during gameplay.
- **Warm-up Sequence**: The pipeline runs a silent compilation and warm-up batch at startup, so there is zero initial lag during gaming.

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 10/11 |
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | CPU Fallback | NVIDIA RTX (4060 to 4090) with CUDA |
| Disk | ~1.5 GB | ~5 GB (for multiple local language models) |

---

## 🗺️ Roadmap & Future AMD GPU Support

- [x] DXGI Capture and MAD frame differencing
- [x] Incremental OCR with lazy fallback strategy
- [x] Offline Translation with FP16 and OrderedDict caching
- [x] End-to-End threaded pipeline + transparent overlay
- [x] Settings panel & System tray controls
- [ ] **Future Update: AMD GPU Acceleration**
  - AMD ROCm support is currently experimental/Linux-focused for PyTorch on Windows.
  - Planned support via **DirectML** (`torch-directml` extension) or **ONNX Runtime Vulkan** to enable hardware acceleration on all non-NVIDIA GPUs (AMD Radeon, Intel Arc).
  - Currently, AMD GPU systems automatically fallback to our highly optimized multi-threaded CPU pipeline.

---

## 📝 License

This project is licensed under the MIT License.

