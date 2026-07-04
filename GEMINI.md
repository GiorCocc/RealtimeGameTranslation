# Gemini CLI Project Context: Game Translation Overlay

Benvenuto nel contesto di sviluppo di **Game Translation Overlay**. Questo file (`GEMINI.md`) serve come fonte di verità assoluta ed istruzione per gli agenti AI e gli sviluppatori che lavorano su questa codebase. Contiene la struttura del progetto, lo stato attuale, le decisioni architetturali non negoziabili, i comandi di avvio e le linee guida per completare le fasi di sviluppo rimanenti.

---

## 1. Panoramica del Progetto

**Game Translation Overlay** è un'applicazione desktop per Windows che acquisisce lo schermo in tempo reale durante le sessioni di gioco, individua automaticamente qualsiasi area di testo visibile, la traduce offline tramite modelli di traduzione locale e proietta la traduzione direttamente sopra il testo originale tramite un overlay trasparente e click-through.

### Obiettivi Chiave
- **Nessuna API Esterna / Connessione Cloud:** Tutte le operazioni di OCR e traduzione avvengono in locale (100% offline).
- **Universalità:** Nessun profilo specifico per-gioco: l'OCR rileva il testo autonomamente su tutto lo schermo.
- **Esperienza Non Invasiva:** L'overlay grafico non intercetta alcun input (mouse/tastiera) grazie alle API win32 (`WS_EX_TRANSPARENT`), garantendo prestazioni fluide ed ininterrotte durante il gaming.

---

## 2. Architettura & Pipeline Multi-Thread

Il sistema è strutturato come una pipeline sequenziale guidata da code thread-safe (`queue.Queue`) per mantenere la latenza minima ed evitare di bloccare il thread grafico principale di PyQt6:

```
[Schermo di gioco]
       │
       ▼
  CaptureThread  ──► [ _capture_queue ] ──►  OCRWorker  ──► [ _ocr_queue ] ──►  TranslationWorker  ──► [ _translation_queue ] ──►  OverlayWindow
 (dxcam, ~30fps)                          (PaddleOCR)                           (MarianMT locale)                                (PyQt6 UI Thread)
```

1. **CaptureThread (Produttore Frame):** Usa `dxcam` per catturare lo schermo ad alta efficienza. Implementa un sistema di **Frame Differencing** dividendo il frame in una griglia 4×4 e calcolando l'hash MD5 per ogni cella. Se nessun blocco cambia, il frame viene scartato per risparmiare risorse CPU/GPU.
2. **OCRWorker (Rilevatore Testo):** Riceve i frame cambiati, applica un downscale opzionale (a 720p) per migliorare la velocità, ed esegue **PaddleOCR** per estrarre le bounding box (`bbox`), il testo originale e lo score di confidenza. Supporta la classificazione dell'angolo (`use_angle_cls`) e un fallback opzionale su **EasyOCR** per testi irregolari. Filtra i testi corti (< 3 caratteri) e quelli a bassa confidenza (< 0.7).
3. **TranslationWorker (Traduttore):** Riceve le regioni di testo identificate, applica un filtro "text delta" per non tradurre stringhe già elaborate, consulta una cache LRU interna, e invia in batch le stringhe sconosciute al modello **Helsinki-NLP MarianMT** (caricato localmente). Emette le tuple `[(bbox, testo_tradotto)]`.
4. **OverlayWindow (UI PyQt6):** Gestisce una finestra trasparente fullscreen, stays-on-top e click-through. Riceve le traduzioni via segnali Qt e disegna le label sopra le aree di gioco originali con transizioni animate di fade (in/out) e un filtro di **Smoothing temporale** (media mobile su N frame delle coordinate) per evitare flickering derivante dalle inevitabili imperfezioni dell'OCR.

---

## 3. Stato del Progetto & Roadmap delle Fasi

Il progetto è suddiviso in 6 fasi di sviluppo incrementali. Di seguito è riportato lo stato corrente di ciascun modulo:

| Fase | Modulo / File | Stato | Descrizione / Ruolo |
| :--- | :--- | :--- | :--- |
| **Fase 1** | `core/capture.py` | ✅ **Completata** | Acquisizione DXGI con `dxcam`, griglia differenziale, gestione dei thread e statistiche aggregate. |
| **Fase 2** | `core/ocr.py` | ✅ **Completata** | Integrazione PaddleOCR + EasyOCR fallback, downscale del frame, logica di text delta e filtro di confidenza. |
| **Fase 3** | `core/translator.py` | ✅ **Completata** | Caricamento offline di MarianMT da `models/`, inferenza in batch, gestione della cache LRU interna. |
| **Fase 4** | `core/pipeline.py`<br>`ui/overlay.py` | ✅ **Completata** | Orchestrazione multithread con segnali Qt; creazione della finestra trasparente click-through con PyQt6, smoother per bounding box e animazione label. |
| **Fase 5** | Vari | ✅ **Completata** | Ottimizzazione delle performance (fast differencing numpy, OCR incrementale per zone, lazy loading, FP16, process priority e benchmark). |
| **Fase 6** | `ui/settings.py`<br>`ui/tray.py` | ✅ **Completata** | Pannello impostazioni PyQt6 (selezione monitor, lingue, risorse), persistenza in `settings.json` ed icona di tray con `pystray`. |

---

## 4. Requisiti & Comandi di Setup / Avvio

### Prerequisiti
- **OS:** Windows 10/11 (obbligatorio per `dxcam` e le API win32 click-through)
- **Python:** Versione 3.11+
- **Hardware:** Raccomandata una GPU NVIDIA con CUDA per un'esecuzione OCR e di traduzione fluida (~50-80ms vs ~250-300ms su CPU).

### Setup dell'Ambiente
```bash
# Creazione e attivazione del virtual environment
python -m venv .venv
.venv\Scripts\activate

# Installazione delle dipendenze principali
pip install -r requirements.txt
```

> **Abilitazione GPU CUDA:** Per sfruttare l'accelerazione GPU, commentare `paddlepaddle` nel file `requirements.txt` e installare `paddlepaddle-gpu`. Inoltre, installare la build di PyTorch abilitata a CUDA seguendo le istruzioni ufficiali di [pytorch.org](https://pytorch.org).

### Esecuzione ed Avvio dei Test

Al momento, solo la **Fase 1 (Capture Pipeline)** è completamente operativa ed eseguibile per testare la latenza e l'accuratezza dell'acquisizione:

```bash
# Avvio Fase 1: Test capture pipeline con preview video OpenCV e sovrapposizione griglia
python main.py --phase 1

# Avvio Fase 1 su monitor secondario, senza preview grafica (solo statistiche nel terminale)
python main.py --phase 1 --monitor 1 --no-preview

# Avvio in modalità debug (mostra log dettagliati)
python main.py --phase 1 --debug
```

*Istruzioni per uscire dalla preview:* premere il tasto **`q`** sulla finestra della preview di OpenCV o premere **`Ctrl+C`** nella console.

---

## 5. Vincoli Architetturali e Decisioni Non Negoziabili

Durante lo sviluppo di qualsiasi nuova funzionalità o refactoring, gli agenti AI e i programmatori **devono** rispettare i seguenti principi e scelte tecnologiche (vedere `KNOWLEDGE_BASE.md` per dettagli estesi):

1. **Nessun Servizio Cloud:** È tassativamente vietato introdurre librerie o chiamate HTTP esterne a servizi come DeepL, Google Translate, OpenAI, Azure o AWS. L'intera pipeline deve funzionare off-grid.
2. **Uso Esclusivo di `dxcam` per lo Screen Capture:** Non utilizzare `PIL.ImageGrab`, `mss` o `pyautogui`. Queste alternative introducono un overhead eccessivo (>100ms) non compatibile con il gameplay in tempo reale.
3. **PaddleOCR come Motore Principale:** Non sostituire PaddleOCR con Tesseract. Tesseract non è adatto per estrarre coordinate complesse e ruotate, ed è notevolmente più lento su font non standard o ambientazioni pixel-art.
4. **PyQt6 per la GUI:** PyQt6 è l'unico framework UI ammesso. Non utilizzare `tkinter` (manca il supporto nativo per la trasparenza alfa reale con click-through) né `pygame` (non è un tool adatto per overlay trasparenti integrati con Windows).
5. **Click-Through Mandatorio:** L'overlay PyQt6 deve sempre applicare lo stile esteso `WS_EX_TRANSPARENT` tramite `win32api`/`pywin32` per delegare gli input del mouse e della tastiera direttamente al gioco sottostante.
6. **Nessuna Whitelist Rigida:** Non limitare l'OCR a coordinate preimpostate dello schermo; l'intero schermo deve essere scansionato per identificare dinamicamente dialoghi, interfacce utente, note testuali 3D e HUD.

---

## 6. Struttura dei Moduli e Interfacce Attese

### `config.py` (Configurazione)
Contiene la classe singleton `config` (basata su `@dataclass`). Tutti i moduli devono importare questa istanza per leggere e scrivere configurazioni a runtime.
- Proprietà chiave: `marian_model_name`, `marian_model_local_path`.
- Cartella di default per i modelli: `./models/`.

### `core/capture.py` (Cattura Schermo - Completato)
- `FrameDifferencer(rows, cols)`: Analizza i frame a griglia per determinare se ci sono modifiche basate sull'MD5 delle singole celle.
- `CaptureThread(frame_queue, monitor_index, target_fps)`: Thread daemon che scrive tuple `(frame_bgr, changed_cells)` sulla queue. Supporta `pause()`, `resume()` e `stop()`.

### `core/ocr.py` (OCR e Text Detection - TODO Fase 2)
Deve esportare la classe `OCRWorker`:
- **Input:** Riceve `(frame, changed_cells)` da `input_queue`.
- **Output:** Produce `list[TextRegion]` in `output_queue`.
- **Struttura della classe `TextRegion`:**
  ```python
  @dataclass(frozen=True)
  class TextRegion:
      bbox: tuple[tuple[float, float], ...]  # 4 punti coordinati (x, y) sul frame originale
      text: str                              # Testo grezzo in lingua sorgente
      confidence: float                      # Score di confidenza [0.0 - 1.0]
      source: str = "paddle"                 # "paddle" o "easy" (in caso di fallback)
  ```
- **Attività da svolgere:**
  - Caricare `PaddleOCR` ed `EasyOCR` all'avvio in `_load_models()`.
  - Gestire la logica di downscale in `_process_frame()`.
  - Integrare un filtro per escludere testi corti (`ocr_min_text_length`) ed a bassa confidenza (`ocr_confidence_threshold`).
  - Implementare il confronto testuale con il frame precedente (`self._prev_texts`) per inviare al traduttore solo le variazioni.

### `core/translator.py` (Traduzione Locale - TODO Fase 3)
Deve esportare la classe `TranslationWorker`:
- **Input:** `list[TextRegion]` da `input_queue`.
- **Output:** `list[tuple[tuple, str]]` (ovvero `[(bbox, translated_text)]`) in `output_queue`.
- **Attività da svolgere:**
  - Caricare il modello MarianMT in `_load_model()`. Se il modello locale non esiste in `config.marian_model_local_path`, scaricarlo da HuggingFace (`config.marian_model_name`) e salvarlo permanentemente su disco.
  - Implementare `translate_batch(texts)` raggruppando i testi non presenti nella cache ed eseguendo un'inferenza batch singola tramite `self._model.generate()`.
  - Mantenere la cache LRU manuale (`_store_cache`) per evitare traduzioni ripetitive e garantire feedback istantanei.

### `core/pipeline.py` (Orchestratore - TODO Fase 4)
Gestisce il ciclo di vita di tutti i thread e la sincronizzazione:
- Instanzia `CaptureThread`, `OCRWorker` e `TranslationWorker`.
- Collega le code ed inoltra i risultati finali della traduzione alla finestra Qt.
- Gestisce i metodi globali `start()`, `stop()`, `pause()`, `resume()`.

### `ui/overlay.py` (Finestra Overlay - TODO Fase 4)
Contiene la visualizzazione PyQt6:
- `BBoxSmoother`: Applica una media mobile temporale delle coordinate per evitare micro-flickering.
- `OverlayWindow(QWidget)`: Finestra frameless trasparente. Deve connettersi ad un segnale Qt `update_signal = pyqtSignal(list)` per ricevere in modo thread-safe le traduzioni da `TranslationWorker`.
- Le label devono essere istanze personalizzate che supportano animazioni di dissolvenza (`QPropertyAnimation` su opacità) e un posizionamento dinamico sopra i testi originali.
- Deve catturare gli hotkey globali `F10` (mostra/nasconde l'overlay) ed `F9` (cambia l'opacità dello sfondo delle label).

### `ui/settings.py` (Pannello Configurazione - TODO Fase 6)
- Finestra di dialogo PyQt6 per personalizzare tutti i campi di `config.py` e salvare/caricare la configurazione nel file `settings.json`.

### `ui/tray.py` (Icona System Tray - TODO Fase 6)
- Gestione di un'icona tray tramite `pystray` per consentire un facile controllo dell'applicazione senza finestre di terminale visibili.

---

## 7. Convenzioni di Codifica e Standard del Progetto

Per mantenere la codebase pulita e coerente con la struttura preesistente:

1. **Type Annotations:** Tutte le funzioni, classi e metodi devono avere type hint rigorosi. Abilitare sempre `from __future__ import annotations`.
2. **Docstring in Italiano:** Mantenere i commenti, i log utente e le docstring in lingua italiana, per coerenza con il resto del progetto. I nomi dei file, delle variabili e delle classi rimangono invece in lingua inglese.
3. **Gestione Concorrenza:** Utilizzare thread dedicati per la cattura e la traduzione, e `ThreadPoolExecutor` per gestire il carico variabile dell'OCR in modo asincrono. Evitare assolutamente race-condition sulle code.
4. **Resilienza alle Eccezioni:** Proteggere i thread della pipeline racchiudendo le logiche interne in blocchi `try-except` generici con logging. Un crash dell'OCR o del traduttore non deve mai tirare giù l'intera applicazione o il thread grafico Qt.
5. **Logger Nominato:** Utilizzare sempre `logger = logging.getLogger(__name__)` per tracciare il flusso applicativo, evitando stampe generiche con `print()` nei moduli core e ui.
