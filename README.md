# Realtime Game Translation Overlay

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)

Un'applicazione desktop Windows per la traduzione in tempo reale dello schermo durante le sessioni di gioco. Il sistema acquisisce lo schermo a bassissima latenza, rileva automaticamente qualsiasi testo visibile tramite modelli OCR locali, lo traduce offline in tempo reale e proietta la traduzione direttamente sopra il testo originale tramite un overlay grafico trasparente e click-through.

---

## 📌 Indice
1. [Panoramica e Contesto](#-panoramica-e-contesto)
2. [Funzionalità Principali](#-funzionalità-principali)
3. [Demo](#-demo)
4. [Tech Stack](#-tech-stack)
5. [Requisiti di Sistema](#-requisiti-di-sistema)
6. [Installazione e Setup](#-installazione-e-setup)
7. [Configurazione](#-configurazione)
8. [Utilizzo e Quick Start](#-utilizzo-e-quick-start)
9. [Struttura del Progetto](#-struttura-del-progetto)
10. [Architettura e Pipeline](#-architettura-e-pipeline)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [Licenza](#-licenza)
14. [Contatti e Credits](#-contatti-e-credits)

---

## 📖 Panoramica e Contesto

**Realtime Game Translation Overlay** nasce per rendere accessibili i giochi che non supportano la propria lingua (in particolare RPG, avventure testuali e giochi ricchi di dialoghi), senza rompere l'immersione del giocatore. 

A differenza di altri strumenti, l'applicazione:
- **Funziona al 100% offline**: Nessuna chiamata API esterna (DeepL, Google Translate, OpenAI, ecc.), garantendo privacy totale e nessun costo di utilizzo.
- **È universale**: Non richiede profili o mod specifiche per ogni gioco; l'OCR rileva dinamicamente il testo su qualsiasi schermata.
- **Non è invasiva**: L'overlay grafico non intercetta gli input di mouse e tastiera, consentendo di giocare normalmente mentre le traduzioni appaiono sovrapposte in tempo reale.

---

## ✨ Funzionalità Principali

- **Screen Capture DXGI**: Cattura ad alte prestazioni tramite `dxcam` (DXGI Desktop Duplication API), garantendo frame rate elevati con un impatto minimo sulla CPU/GPU.
- **Frame Differencing con Numpy**: Algoritmo basato su griglia 4×4 e confronto MAD (Mean Absolute Difference) per elaborare solo le porzioni di schermo effettivamente cambiate (10x più veloce rispetto a controlli MD5).
- **OCR Incrementale a due livelli**:
  - **PaddleOCR** come motore principale ad altissima precisione.
  - **EasyOCR** (caricato in modalità lazy/on-demand) come fallback per testi irregolari o texture complesse.
- **Traduzione locale con MarianMT**: Inferenza offline tramite modelli Helsinki-NLP MarianMT ottimizzati in FP16 su GPU CUDA per massimizzare la velocità e dimezzare l'impatto VRAM.
- **Cache LRU integrata**: Memorizzazione O(1) delle traduzioni tramite `OrderedDict` per evitare di tradurre testi ripetitivi.
- **Finestra di Overlay click-through**: Interfaccia frameless PyQt6 configurata con attributi di sistema `WS_EX_TRANSPARENT` per rendersi trasparente all'input del giocatore.
- **Supporto DPI-Aware**: Gestione nativa del fattore di scala DPI (Per-Monitor V2) per mantenere le bounding box allineate su monitor di qualsiasi risoluzione.
- **Pannello Impostazioni e System Tray**: Gestione dei parametri, monitoraggio risorse CPU/GPU/VRAM ed icona nella tray di Windows per l'avvio e spegnimento rapido.
- **Hotkey Globali**:
  - `F10`: Mostra/nasconde l'overlay grafico.
  - `F9`: Attiva/disattiva lo sfondo opaco dietro le traduzioni per migliorare la leggibilità.

---

## 📷 Demo

> [!NOTE]
> **Placeholder per la Demo**: Attualmente non è presente una demo video o GIF pubblica sul repository. 
> 
> *TODO: Aggiungere qui una GIF animata dell'overlay in azione su un videogioco di esempio.*

---

## 🛠️ Tech Stack

- **Linguaggio**: Python 3.12+
- **Cattura Schermo**: `dxcam`
- **Computer Vision & Elaborazione**: `numpy`, `opencv-python`, `pillow`
- **OCR Engines**: `paddleocr` (PaddlePaddle), `easyocr` (PyTorch)
- **Traduzione Locale**: `transformers` (MarianMT), `sacremoses`, `sentencepiece`
- **Framework GUI & Integrazione OS**: `PyQt6`, `pywin32` (Win32 API)
- **Gestione UX**: `keyboard` (hotkey di sistema), `pystray` (system tray icon)
- **Performance & System Utilities**: `psutil`, `torch` (con supporto CUDA)

---

## 💻 Requisiti di Sistema

| Componente | Requisito Minimo | Consigliato |
| :--- | :--- | :--- |
| **Sistema Operativo** | Windows 10 / 11 (64-bit) | Windows 10 / 11 (64-bit) |
| **CPU** | Intel i5 / AMD Ryzen 5 (4 core) | Intel i7 / AMD Ryzen 7 (6+ core) |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Qualsiasi (Esecuzione su CPU) | NVIDIA RTX con supporto CUDA (per inferenza ultra-rapida) |
| **Spazio su Disco** | ~1.5 GB | ~5 GB (in base ai modelli linguistici installati) |

---

## 🚀 Installazione e Setup

### 1. Clona il repository
```bash
git clone https://github.com/GiorCocc/RealtimeGameTranslation.git
cd RealtimeGameTranslation
```

### 2. Configura l'ambiente virtuale
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installa le dipendenze
Installa le librerie base specificate nel file `requirements.txt`:
```powershell
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Accelerazione GPU (Consigliata)**: 
> Per sfruttare al massimo la GPU NVIDIA tramite CUDA:
> 1. Disinstalla la versione CPU di `paddlepaddle` e installa `paddlepaddle-gpu`.
> 2. Installa la build di `PyTorch` abilitata a CUDA seguendo le istruzioni ufficiali sul sito di [PyTorch.org](https://pytorch.org/).

---

## ⚙️ Configurazione

Tutte le impostazioni vengono caricate e salvate in un file locale chiamato `settings.json` generato al primo avvio. I parametri includono:
- Lingua sorgente e di destinazione per OCR e traduzione.
- Selezione del monitor da monitorare.
- Soglia di confidenza OCR e filtri sui testi brevi.
- Impostazioni di rendering delle label dell'overlay (font, dimensioni, opacità dello sfondo).

I parametri possono essere modificati sia a runtime tramite il **Pannello Impostazioni** accessibile dall'icona nella barra di sistema (System Tray), sia modificando direttamente il file `settings.json`.

---

## 🎮 Utilizzo e Quick Start

L'applicazione supporta diversi parametri da riga di comando per testare singolarmente le varie fasi del progetto.

### Avvio dell'Applicazione Completa
Per avviare l'applicazione con overlay e icona nella tray:
```powershell
python main.py
# Oppure esplicitamente
python main.py --phase 4
```

Per avviare l'applicazione abilitando i log di debug in console:
```powershell
python main.py --debug
```

### Comandi per i Test di Fase (Debugging)

- **Fase 1: Test della Pipeline di Cattura**
  Avvia il feed video di cattura con overlay della griglia di differencing:
  ```powershell
  python main.py --phase 1
  ```
  Per testare la cattura su un monitor secondario senza mostrare la preview video (solo statistiche a terminale):
  ```powershell
  python main.py --phase 1 --monitor 1 --no-preview
  ```

- **Fase 2: Test del motore OCR**
  Rileva il testo a schermo e mostra a video le bounding box individuate:
  ```powershell
  python main.py --phase 2
  ```

- **Fase 3: Test della Traduzione End-to-End**
  Avvia la pipeline di cattura, OCR e traduzione stampando i risultati testuali direttamente in console (senza overlay grafico):
  ```powershell
  python main.py --phase 3
  ```

- **Fase 5: Test delle Prestazioni e Benchmark**
  Avvia un benchmark automatico senza interfaccia grafica per 30 secondi e stampa un report dettagliato di latenza e consumo di risorse:
  ```powershell
  python main.py --phase 5
  ```

*Nota: Nelle modalità con preview OpenCV (`--phase 1`, `--phase 2`), è possibile chiudere la finestra premendo il tasto **`q`**.*

---

## 📁 Struttura del Progetto

```text
RealtimeGameTranslation/
├── main.py                  # Entry point del programma e runner per i test di fase
├── config.py                # Gestore di configurazione globale (Singleton)
├── requirements.txt         # Elenco delle dipendenze di progetto
├── core/                    # Logica di backend e pipeline dati
│   ├── capture.py           # Gestione dxcam e differencing MAD numpy (Fase 1)
│   ├── ocr.py               # Rilevamento testo con PaddleOCR + EasyOCR fallback (Fase 2)
│   ├── translator.py        # Traduzione offline con MarianMT (Fase 3)
│   ├── gpu_utils.py         # Utility per il monitoraggio hardware di GPU e VRAM
│   └── pipeline.py          # Coordinatore multithreaded e code di sincronizzazione (Fase 4)
└── ui/                      # Moduli dell'interfaccia grafica
    ├── overlay.py           # Finestra di overlay PyQt6 click-through e bounding box smoothing
    ├── settings.py          # Pannello delle impostazioni e visualizzazione risorse
    └── tray.py              # Icona nella system tray e gestione del ciclo di vita dei thread
```

---

## 🧠 Architettura e Pipeline

Il flusso di elaborazione dati è multithreaded per non ostacolare il frame rate di gioco e la fluidità dell'overlay grafico:

```
[Schermo di Gioco]
       │
       ▼
  CaptureThread  ──► [ _capture_queue ] ──►  OCRWorker  ──► [ _ocr_queue ] ──►  TranslationWorker  ──► [ _translation_queue ] ──►  OverlayWindow
 (dxcam, ~30fps)                          (PaddleOCR)                           (MarianMT locale)                                (PyQt6 UI Thread)
```

1. **Priorità di Processo**: Il processo principale e i thread dell'OCR/Traduzione sono impostati a priorità ridotta (`BELOW_NORMAL`) per non impattare sulle performance dei videogiochi eseguiti contemporaneamente.
2. **Warm-up**: All'avvio dell'applicazione viene eseguita una traduzione fittizia silente per compilare i grafi dei modelli ed eliminare il lag alla prima visualizzazione di testo reale.

---

## 🗺️ Roadmap

- [x] Fase 1: Pipeline di acquisizione DXGI a bassa latenza e calcolo differenze frame.
- [x] Fase 2: Rilevamento del testo tramite PaddleOCR e integrazione EasyOCR.
- [x] Fase 3: Traduzione locale basata su MarianMT con ottimizzazione FP16.
- [x] Fase 4: Overlay trasparente PyQt6 con logica di click-through nativa Windows.
- [x] Fase 5: Ottimizzazione prestazionale generale e benchmark di latenza.
- [x] Fase 6: UI di configurazione e integrazione dell'icona nella barra delle applicazioni.
- [ ] Supporto GPU AMD via DirectML / ONNX Runtime (Attualmente sperimentale).

---

## 🤝 Contributing

I contributi sono sempre benvenuti! Se desideri proporre miglioramenti o segnalare bug:
1. Effettua un fork del repository.
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`).
3. Effettua il commit dei tuoi cambiamenti (`git commit -m 'Add some AmazingFeature'`).
4. Esegui il push sul branch (`git push origin feature/AmazingFeature`).
5. Apri una Pull Request descrivendo accuratamente le modifiche apportate.

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. 