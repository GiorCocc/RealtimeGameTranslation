"""
core/translator.py — Phase 3: Traduzione Locale (Offline)

Responsabilità:
  - Carica e mantiene in memoria i modelli Helsinki-NLP MarianMT
  - Traduzione batch di liste di stringhe
  - Cache LRU per evitare ricalcoli su frasi già tradotte
  - Download automatico del modello se non presente in models/

Classe pubblica principale:
  TranslationWorker  — thread consumer che produce list[(bbox, translated_text)]

Input atteso (Phase 3):
  list[TextRegion] (vedere core/ocr.py) prodotta da OCRWorker.

Output atteso (Phase 3):
  list[tuple[bbox, str]] per ogni batch di regioni ricevuto.

ATTENZIONE: NON introdurre chiamate a API cloud (DeepL, Google Translate,
OpenAI, ecc.). La traduzione deve essere interamente offline.
Vedere KNOWLEDGE_BASE.md §13.
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Optional, TYPE_CHECKING

from config import config

if TYPE_CHECKING:
    # Solo per type hint — evita import pesante/circolare a runtime.
    from core.ocr import TextRegion

logger = logging.getLogger(__name__)

# Dimensione massima di un batch passato a model.generate() in una singola
# chiamata. Valore scelto come compromesso RAM/VRAM ↔ latenza su CPU/GPU
# di fascia media (vedere KNOWLEDGE_BASE.md §12 — requisiti hardware).
_MAX_GENERATE_BATCH_SIZE = 16

# Lunghezza massima (in token) per singola stringa da tradurre. Oltre
# questa soglia il testo viene troncato: i testi di gioco (dialoghi, HUD,
# menu) restano quasi sempre ben sotto questo limite.
_MAX_TOKEN_LENGTH = 512

# Quante misurazioni di tempo traduzione mantenere per la media mobile
# esposta in .stats (avg_translation_ms).
_STATS_WINDOW = 50


# ══════════════════════════════════════════════════════════════════
# Translation Worker
# ══════════════════════════════════════════════════════════════════

class TranslationWorker:
    """
    Worker per traduzione locale con MarianMT (HuggingFace Transformers).

    Flusso:
      input_queue  → riceve list[TextRegion] da OCRWorker
      output_queue → produce list[tuple[bbox, translated_text]] verso OverlayWindow

    Ottimizzazioni:
      - LRU cache: frasi già tradotte restituite istantaneamente
      - Batch translation: raggruppa le frasi nuove e le traduce in poche
        chiamate al modello (più efficiente di una chiamata per stringa)
      - Modello in memoria per tutta la sessione (no reload tra frame)
      - Come CaptureThread/OCRWorker: se arrivano più batch mentre il
        worker è occupato, si scarta tutto tranne il più recente
        (priorità alla latenza, non alla completezza — vedere
        KNOWLEDGE_BASE.md §5, threading model)

    Uso tipico:
        input_q = Queue(maxsize=4)
        output_q = Queue(maxsize=8)
        translator = TranslationWorker(input_q, output_q)
        translator.start()      # blocca finché il modello non è caricato
        translated = output_q.get()
        translator.stop()
    """

    def __init__(
        self,
        input_queue: Queue,    # riceve list[TextRegion] da OCRWorker
        output_queue: Queue,   # produce list[tuple[bbox, str]] verso Overlay
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self._stop_event = threading.Event()
        self._model = None       # MarianMTModel
        self._tokenizer = None   # MarianTokenizer
        self._device: str = "cpu"

        # Cache LRU implementata come dict + lista d'ordine (vedi sotto).
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []

        # Stats
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._translations_total: int = 0
        self._batches_processed: int = 0
        self._translation_times: list[float] = []
        self._t_start: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Carica il modello (bloccante) e avvia il thread consumer."""
        logger.info(
            "TranslationWorker: caricamento modello %s ...",
            config.marian_model_name,
        )
        self._load_model()
        self._t_start = time.perf_counter()
        threading.Thread(
            target=self._consume_loop, daemon=True, name="TranslationWorker"
        ).start()
        logger.info("TranslationWorker avviato (device=%s)", self._device)

    def stop(self) -> None:
        self._stop_event.set()
        logger.info(
            "TranslationWorker fermato — batch=%d  traduzioni=%d  "
            "cache_hit_rate=%.1f%%",
            self._batches_processed,
            self._translations_total,
            self.stats["cache_hit_rate_pct"],
        )

    # ── Model loading ─────────────────────────────────────────────

    @staticmethod
    def _detect_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _load_model(self) -> None:
        """
        Carica MarianMT dal disco se presente localmente in models/,
        altrimenti lo scarica da HuggingFace e lo salva lì per l'uso
        offline successivo.

        NOTA: il download iniziale richiede una connessione internet.
        Una volta scaricato, il modello (~300MB) risiede interamente in
        models/opus-mt-{src}-{tgt}/ e tutte le esecuzioni successive
        sono completamente offline, in linea con KNOWLEDGE_BASE.md §13.
        """
        from transformers import MarianMTModel, MarianTokenizer

        self._device = "cuda" if self._detect_cuda() else "cpu"
        logger.info(
            "TranslationWorker: CUDA %s",
            "disponibile (GPU)" if self._device == "cuda" else "non disponibile (CPU)",
        )

        local_path = config.marian_model_local_path
        if local_path.exists():
            source = str(local_path)
            logger.info("TranslationWorker: modello locale trovato in %s", source)
        else:
            source = config.marian_model_name
            logger.info(
                "TranslationWorker: modello locale assente, download da "
                "HuggingFace: %s", source,
            )

        self._tokenizer = MarianTokenizer.from_pretrained(source)
        self._model = MarianMTModel.from_pretrained(source)
        self._model.to(self._device)
        self._model.eval()

        # Salva localmente per uso offline futuro se scaricato ora.
        if source != str(local_path):
            local_path.mkdir(parents=True, exist_ok=True)
            self._tokenizer.save_pretrained(str(local_path))
            self._model.save_pretrained(str(local_path))
            logger.info("TranslationWorker: modello salvato in %s", local_path)

    # ── Consumer loop ─────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """
        Loop: prende list[TextRegion], traduce, emette list[(bbox, str)]
        nella output_queue.

        Per ogni batch ricevuto:
          1. Scarta eventuali batch accumulati: interessa solo il più
             recente (stessa filosofia di CaptureThread/OCRWorker)
          2. Suddivide le regioni in chunk da al più
             _MAX_GENERATE_BATCH_SIZE elementi e le traduce (con cache)
             un chunk alla volta tramite translate_batch()
          3. Dopo OGNI chunk, emette subito in output_queue l'insieme
             cumulativo delle traduzioni completate finora — non aspetta
             che l'intero batch sia tradotto. Su una scena densa (es.
             un IDE o un menu con decine di elementi) tradurre tutto il
             batch su CPU può richiedere anche minuti: aspettare la fine
             prima di mostrare qualunque risultato violerebbe la
             filosofia "priorità alla latenza, non completezza" già
             usata da CaptureThread/OCRWorker (vedere KNOWLEDGE_BASE.md
             §5, threading model).
        """
        while not self._stop_event.is_set():
            try:
                regions: list["TextRegion"] = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            # Scarta eventuali batch accumulati nel frattempo: teniamo
            # solo il più recente (priorità alla latenza).
            while True:
                try:
                    regions = self.input_queue.get_nowait()
                except Empty:
                    break

            if not regions:
                # Nessuna regione di testo: propaga comunque una lista
                # vuota così l'overlay può rimuovere le label residue.
                logger.debug("TranslationWorker: batch vuoto ricevuto (nessun testo da tradurre)")
                self._emit([])
                continue

            logger.debug(
                "TranslationWorker: batch ricevuto — %d region/i totali",
                len(regions),
            )

            t0 = time.perf_counter()
            accumulated_output: list = []
            try:
                for start in range(0, len(regions), _MAX_GENERATE_BATCH_SIZE):
                    chunk_regions = regions[start:start + _MAX_GENERATE_BATCH_SIZE]
                    chunk_texts = [r.text for r in chunk_regions]

                    chunk_translated = self.translate_batch(chunk_texts)

                    accumulated_output.extend(
                        (r.bbox, t) for r, t in zip(chunk_regions, chunk_translated)
                    )
                    # Emissione progressiva: l'overlay/il consumer vede
                    # subito le traduzioni completate finora, non solo a
                    # fine batch.
                    self._emit(list(accumulated_output))
                    logger.debug(
                        "TranslationWorker: chunk tradotto ed emesso — "
                        "%d/%d region/i completate finora: %s",
                        len(accumulated_output), len(regions),
                        [t[:30] for t in chunk_translated],
                    )
            except Exception:
                logger.exception("TranslationWorker: errore durante translate_batch")
                continue

            dt_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "TranslationWorker: batch completo in %.0f ms (%d region/i totali)",
                dt_ms, len(regions),
            )

            self._translation_times.append(dt_ms)
            if len(self._translation_times) > _STATS_WINDOW:
                self._translation_times.pop(0)
            self._batches_processed += 1

    def _emit(self, output: list) -> None:
        """Mette `output` in output_queue, rimpiazzando il più vecchio se piena."""
        try:
            self.output_queue.put_nowait(output)
        except Full:
            try:
                self.output_queue.get_nowait()
            except Empty:
                pass
            try:
                self.output_queue.put_nowait(output)
            except Full:
                pass

    # ── Translation ───────────────────────────────────────────────

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Traduce una lista di stringhe, sfruttando la cache LRU per le
        frasi già tradotte in precedenza.

        Args:
            texts: Stringhe nella lingua sorgente (già filtrate da OCR:
                   lunghezza minima, confidenza minima, ecc.)

        Returns:
            Lista di traduzioni nello stesso ordine di `texts`.
        """
        if not texts:
            return []

        results: list[Optional[str]] = [None] * len(texts)
        to_translate: list[str] = []
        to_translate_idx: list[int] = []

        for i, t in enumerate(texts):
            cached = self._translate_cached(t)
            if cached is not None:
                results[i] = cached
                self._cache_hits += 1
            else:
                to_translate.append(t)
                to_translate_idx.append(i)
                self._cache_misses += 1

        if to_translate:
            translated = self._run_model(to_translate)
            for idx, orig, trans in zip(to_translate_idx, to_translate, translated):
                self._store_cache(orig, trans)
                results[idx] = trans

        self._translations_total += len(texts)
        return results  # type: ignore[return-value]

    def _run_model(self, texts: list[str]) -> list[str]:
        """
        Inferenza MarianMT su una lista di stringhe, suddivisa in batch
        da al massimo _MAX_GENERATE_BATCH_SIZE elementi per contenere
        l'uso di RAM/VRAM.
        """
        import torch

        all_translations: list[str] = []

        for start in range(0, len(texts), _MAX_GENERATE_BATCH_SIZE):
            chunk = texts[start:start + _MAX_GENERATE_BATCH_SIZE]
            t_chunk = time.perf_counter()

            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_TOKEN_LENGTH,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    num_beams=config.translation_num_beams,
                    max_new_tokens=config.translation_max_new_tokens,
                )

            decoded = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            all_translations.extend(decoded)
            logger.debug(
                "TranslationWorker: chunk di %d testi tradotto in %.0f ms "
                "(num_beams=%d, max_new_tokens=%d, device=%s)",
                len(chunk),
                (time.perf_counter() - t_chunk) * 1000,
                config.translation_num_beams,
                config.translation_max_new_tokens,
                self._device,
            )

        return all_translations

    # ── LRU Cache ─────────────────────────────────────────────────
    # Implementata come dict + lista d'ordine invece di @lru_cache
    # perché @lru_cache non funziona su metodi (self non è hashable in
    # modo stabile tra le chiamate) e perché serve poter ispezionare
    # hit/miss per le statistiche esposte in .stats.

    def _translate_cached(self, text: str) -> Optional[str]:
        """Restituisce la traduzione dalla cache o None se assente."""
        return self._cache.get(text)

    def _store_cache(self, original: str, translation: str) -> None:
        """Salva una traduzione nella cache, rispettando il limite LRU."""
        if original in self._cache:
            # Già presente (race tra due miss sullo stesso testo nello
            # stesso batch): aggiorna solo il valore, non l'ordine.
            self._cache[original] = translation
            return

        if len(self._cache) >= config.translation_cache_maxsize:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[original] = translation
        self._cache_order.append(original)

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total_lookups = self._cache_hits + self._cache_misses
        avg_ms = (
            sum(self._translation_times) / len(self._translation_times)
            if self._translation_times else 0.0
        )
        return {
            "translations_total": self._translations_total,
            "batches_processed": self._batches_processed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
            "cache_hit_rate_pct": round(
                self._cache_hits / max(1, total_lookups) * 100, 1
            ),
            "avg_translation_ms": round(avg_ms, 1),
            "device": self._device,
        }