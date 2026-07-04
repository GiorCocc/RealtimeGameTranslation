# Game Translation Overlay — Project Knowledge Base

> Documento di riferimento per il coding agent. Descrive obiettivo, architettura, stack, vincoli e decisioni di design del progetto.

---

## 1. Obiettivo del Progetto

Creare un'applicazione desktop per **Windows** che:

- Cattura in tempo reale lo schermo mentre l'utente gioca
- Individua automaticamente ogni area di testo presente sullo schermo (dialoghi, menu, HUD, inventario, testi su texture, bigliettini, cartelli, ecc.)
- Traduce il testo rilevato usando un modello locale (nessuna API esterna, nessuna connessione internet)
- Sovrappone la traduzione allo schermo tramite un overlay trasparente, posizionato esattamente sopra il testo originale
- Funziona universalmente su qualsiasi gioco, senza configurazione specifica per titolo

L'utente non deve fare nulla durante il gioco: il sistema è completamente automatico, continuo e non invasivo.

---

## 2. Contesto d'uso

- L'utente è uno streamer/gamer che gioca a titoli non localizzati in italiano
- Durante le live non può fermarsi a tradurre manualmente
- Il programma deve essere leggero, stabile e non interferire con il gioco (no input capture, no lag percettibile)
- La latenza accettabile è ~200-300ms su CPU, ~50-80ms con GPU CUDA

---

## 3. Principi Fondamentali

| Principio | Dettaglio |
|---|---|
| **Nessuna API esterna** | OCR e traduzione girano interamente in locale |
| **Universalità** | Nessuna configurazione per-gioco: il sistema trova il testo da solo |
| **Non invasivo** | L'overlay non cattura mouse né tastiera (click-through) |
| **Tempo reale** | Niente screenshot manuali, tutto continuo e automatico |
| **Scope completo** | Tutto lo schermo è analizzato: dialoghi, menu, HUD, texture con testo, inventario, mappe, ecc. |

---

## 4. Stack Tecnologico

| Componente | Tecnologia | Note |
|---|---|---|
| Linguaggio | **Python 3.11+** | Ecosistema ML/CV |
| Screen capture | **dxcam** | DXGI Desktop Duplication API, 60+ fps, bassa latenza |
| OCR + Text Detection | **PaddleOCR** | Rileva bounding box automaticamente, multilingua, restituisce `[[bbox, text, confidence]]` |
| OCR fallback (texture) | **EasyOCR** | Per testi su texture scritti a mano o con font irregolari |
| Traduzione locale | **Helsinki-NLP MarianMT** (HuggingFace `transformers`) | Modelli ~300MB, offline, supporta molte coppie di lingue |
| Overlay UI | **PyQt6** | Finestra trasparente always-on-top, click-through via win32api |
| Window management | **pywin32** | Necessario per click-through su Windows (`WS_EX_TRANSPARENT`) |
| Hotkey globali | **keyboard** | Attivare/disattivare overlay senza focus sull'app |
| Tray icon | **pystray** | Controllo app da system tray |
| Deep learning runtime | **torch** (CPU o CUDA) | Backend per MarianMT e PaddleOCR |

### requirements.txt

```
dxcam
numpy
pillow
opencv-python
paddlepaddle          # oppure paddlepaddle-gpu se CUDA disponibile
paddleocr
easyocr
transformers
sentencepiece
torch
PyQt6
pywin32
keyboard
pystray
```

---

## 5. Architettura del Sistema

Il programma è una pipeline multi-thread con quattro stadi principali:

```
[GAME WINDOW]
      │
      ▼
[CAPTURE THREAD]  — dxcam, ~30fps
  - Cattura frame continuo
  - Frame differencing (hash MD5/perceptual) per rilevare cambiamenti
  - Se frame identico al precedente → skip
  - Se cambiato → passa frame a OCR
      │
      ▼
[OCR THREAD]  — PaddleOCR (+ EasyOCR come fallback)
  - Analizza l'intero frame (nessuna zona predefinita)
  - Restituisce lista di: (bounding_box, testo, confidenza)
  - Filtra risultati con confidenza < soglia configurabile (default 0.7)
  - Text delta: confronta con testo del frame precedente
  - Invia al translator solo le stringhe nuove o cambiate
      │
      ▼
[TRANSLATION THREAD]  — MarianMT locale
  - Cache LRU per frasi già tradotte (evita ricalcolo)
  - Traduzione batch delle stringhe nuove
  - Emette: [(bounding_box, testo_tradotto)]
      │
      ▼
[OVERLAY]  — PyQt6
  - Finestra frameless, trasparente, always-on-top, a schermo intero
  - Click-through totale (win32api WS_EX_TRANSPARENT)
  - Posiziona label di testo tradotto sopra ogni bounding box
  - Sfondo label: colore scuro semi-trasparente per leggibilità
  - Fade in/out animato al cambio testo
  - Smoothing delle bounding box (media mobile su N frame) per evitare flickering
```

### Threading model

- `CaptureThread` — thread dedicato, produce frame
- `OCRWorker` — thread pool (concurrent.futures), consuma frame
- `TranslationWorker` — thread pool, consuma testo OCR
- `OverlayWindow` — main Qt thread, riceve aggiornamenti via signal/slot

---

## 6. Rilevamento del Testo (OCR)

### Approccio

PaddleOCR analizza l'intero frame e individua autonomamente le regioni di testo. Non è necessario definire zone a priori. L'output è una lista di bounding box con testo e score di confidenza.

### Due tipi di testo gestiti

**Testo standard** (menu, HUD, dialoghi, inventario):
- Font vettoriali o bitmap puliti, alto contrasto
- PaddleOCR gestisce bene questo caso
- Soglia di confidenza: 0.7

**Testo su texture** (bigliettini scritti a mano, cartelli, lettere, oggetti 3D):
- Caratteri irregolari, rumore visivo, possibile prospettiva/rotazione
- PaddleOCR con modalità `angle_class=True` (gestione rotazione)
- Fallback a EasyOCR se PaddleOCR ritorna confidenza bassa
- Nei casi estremi: possibile integrazione futura di un vision model locale (LLaVA/Moondream)

### Ottimizzazione performance

- **Frame differencing a zone**: dividi lo schermo in griglia (es. 4×4) e processa solo le celle cambiate
- **Downscale prima dell'OCR**: ridurre a 720p prima di passare a PaddleOCR (funziona bene, dimezza il carico)
- **Throttling OCR**: non eseguire OCR ogni frame, ma ogni ~150ms
- **Text delta**: non ritradurre testo già tradotto se non è cambiato

---

## 7. Traduzione Locale

### Modello scelto: Helsinki-NLP MarianMT

- Modelli disponibili su HuggingFace per ~300MB cadauno
- Esempi: `opus-mt-en-it` (EN→IT), `opus-mt-ja-en` (JA→EN), `opus-mt-zh-en` (ZH→EN)
- Caricati all'avvio, mantenuti in memoria durante tutta la sessione
- Traduzione batch per efficienza

### Cache LRU

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def translate_cached(text: str) -> str:
    ...
```

Le frasi già tradotte vengono restituite immediatamente senza ricalcolo.

### Flusso di selezione lingua

1. All'avvio l'utente seleziona lingua sorgente e lingua destinazione
2. Il modello corrispondente viene caricato (o scaricato se non presente)
3. I modelli vengono salvati nella cartella `models/` del progetto

---

## 8. Overlay (PyQt6)

### Caratteristiche tecniche

- Finestra `FramelessWindowHint` + `WindowStaysOnTopHint`
- `setAttribute(Qt.WA_TranslucentBackground)` per trasparenza totale
- Click-through su Windows tramite `win32api`:
  ```python
  import win32api, win32con
  hwnd = int(window.winId())
  win32api.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
      win32api.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_TRANSPARENT)
  ```
- Label di traduzione: sfondo `rgba(0, 0, 0, 160)`, testo bianco, font leggibile
- Animazione fade (QPropertyAnimation sull'opacità) al cambio di testo

### Gestione bounding box

- Le coordinate delle bounding box di PaddleOCR sono in pixel assoluti
- Viene applicato uno **smoothing temporale**: se una box è stabile per N frame, le sue coordinate vengono mediate per evitare micro-flickering
- Quando il testo scompare dal frame, la label corrispondente viene rimossa con fade-out

### Modalità visualizzazione

| Modalità | Comportamento |
|---|---|
| **Label sovrapposta** (default) | Testo tradotto con sfondo semi-trasparente sopra l'originale |
| **Label opaca** (toggle) | Sfondo opaco per nascondere il testo originale |

Toggle tra modalità: hotkey configurabile (default `F10`).

---

## 9. Limitazioni Note e Workaround

### Perché non si usa la pipeline grafica (DirectX) per trovare il testo

La cattura tramite DXGI Desktop Duplication API restituisce solo pixel finali renderizzati, senza metadati. Per DirectX il testo non esiste come concetto: è geometria (quad di triangoli) con texture di font applicate. Non è possibile distinguere "testo" da "albero" a livello di draw call in modo universale senza fare hooking profondo della pipeline grafica (RenderDoc-style), che è:
- Specifico per ogni engine/gioco
- Bloccato da molti anti-cheat
- Estremamente complesso da implementare universalmente

**Conclusione**: la cattura DirectX serve per ottenere il frame in modo efficiente. La localizzazione del testo si fa sempre tramite OCR sul frame risultante.

### Testo in movimento (oggetti 3D nel mondo di gioco)

Un cartello in un mondo 3D si sposta con la telecamera frame per frame. Il sistema OCR lo ri-rileva ad ogni frame con bounding box leggermente diverse → flickering dell'overlay.

**Soluzione**: smoothing delle coordinate con media mobile su 3-5 frame consecutivi.

### Falsi positivi OCR

L'analisi dell'intero schermo può rilevare elementi grafici come testo (es. pattern HUD, icone). Il filtro di confidenza (0.7) riduce il problema. In futuro: possibile whitelist/blacklist di regioni definite dall'utente.

### Testi molto brevi o singoli caratteri

OCR tende a fare errori su testi di 1-2 caratteri. Aggiungere un filtro minimo: ignorare testi con meno di 3 caratteri.

---

## 10. Struttura del Progetto

```
game-translator/
├── main.py                  # Entry point, avvio pipeline
├── config.py                # Configurazione globale (lingua, soglie, hotkey, ecc.)
├── core/
│   ├── capture.py           # Wrapper dxcam, frame differencing
│   ├── ocr.py               # Wrapper PaddleOCR + EasyOCR, text delta
│   ├── translator.py        # Wrapper MarianMT, cache LRU
│   └── pipeline.py          # Orchestrazione thread, code signal/slot Qt
├── ui/
│   ├── overlay.py           # Finestra overlay trasparente PyQt6
│   ├── settings.py          # Pannello configurazione
│   └── tray.py              # System tray icon
├── models/                  # Modelli MarianMT scaricati localmente
├── requirements.txt
└── README.md
```

---

## 11. Fasi di Sviluppo

### Fase 1 — Capture pipeline
- Setup progetto, virtualenv, dipendenze
- `capture.py`: cattura con dxcam, selezione monitor/regione
- Frame differencing con hash per evitare frame identici
- **Output atteso**: stream di frame a ~30fps, solo quando lo schermo cambia

### Fase 2 — OCR e Text Detection
- `ocr.py`: integrazione PaddleOCR su frame completo
- Filtro per confidenza minima
- Text delta: confronto con frame precedente
- Test su screenshot di giochi reali (font pixel art, serif, sottotitoli)
- **Output atteso**: lista `[(bbox, text, confidence)]` per ogni frame con testo cambiato

### Fase 3 — Traduzione locale
- `translator.py`: caricamento MarianMT, traduzione batch, cache LRU
- Download e gestione modelli in `models/`
- **Output atteso**: lista `[(bbox, translated_text)]`

### Fase 4 — Overlay PyQt6
- `overlay.py`: finestra trasparente, click-through, label animate
- Ricezione aggiornamenti via Qt signals da pipeline thread
- Smoothing bounding box
- **Output atteso**: testo tradotto visibile sopra il gioco, senza interferenze

### Fase 5 — Ottimizzazione
- Frame differencing a zone (griglia 4×4)
- Downscale frame prima di OCR
- Throttling OCR a ~150ms
- Profiling con cProfile, ottimizzazione bottleneck
- **Output atteso**: latenza totale <300ms su CPU, <100ms su GPU

### Fase 6 — Config Panel e UX
- `settings.py`: selezione lingua, monitor (o finestra specifica, stile OBS), regione, soglie, dimensione carattere...
- nel pannellino di controllo vedere le risorse disponibili (CPU, GPU, RAM usata con previsione dell'impatto)
- Hotkey globali (`keyboard`): toggle overlay, toggle modalità
- `tray.py`: icona tray per start/stop/settings
- **Output atteso**: app completa e usabile senza toccare il codice

---

## 12. Requisiti Hardware

| Componente | Minimo | Consigliato |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| CPU | 4 core | 8+ core (OCR è CPU-bound) |
| RAM | 8 GB | 16 GB |
| GPU | Non richiesta | NVIDIA con CUDA (3-5x più veloce su OCR e inferenza) |
| Disco | ~2 GB | ~5 GB (più coppie di lingue) |

---

## 13. Decisioni di Design da Non Rimettere in Discussione

1. **Nessuna API esterna**: tutto locale. Non introdurre chiamate a DeepL, Google Translate, OpenAI o servizi cloud.
2. **PaddleOCR come motore primario**: non sostituire con Tesseract (prestazioni inferiori su testo non-latino e rilevamento bounding box).
3. **dxcam per la cattura**: non usare PIL.ImageGrab o mss — troppo lenti per uso real-time.
4. **PyQt6 per l'overlay**: non usare tkinter (non supporta trasparenza reale) né pygame (non adatto per overlay).
5. **Click-through obbligatorio**: l'overlay non deve mai intercettare input dell'utente.
6. **Analisi dell'intero schermo**: non limitare l'OCR a zone predefinite — l'utente vuole che ogni area sia una possibile fonte di testo.
