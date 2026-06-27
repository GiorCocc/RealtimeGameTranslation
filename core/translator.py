"""
core/translator.py — Phase 3: Traduzione Locale (Offline)

Responsabilità:
  - Carica e mantiene in memoria i modelli Helsinki-NLP MarianMT
  - Traduzione batch di liste di stringhe
  - Cache LRU per evitare ricalcoli su frasi già tradotte
  - Download automatico del modello se non presente in models/

Classe pubblica principale:
  TranslationWorker  — thread consumer che produce (bbox, translated_text)[]

Output atteso (Phase 3):
  list[tuple[bbox, str]] per ogni batch di stringhe nuove.

ATTENZIONE: NON introdurre chiamate a API cloud (DeepL, Google Translate,
OpenAI, ecc.). La traduzione deve essere interamente offline.
Vedere KNOWLEDGE_BASE.md §13.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Translation Worker — Phase 3 (TODO)
# ══════════════════════════════════════════════════════════════════

class TranslationWorker:
    """
    Worker per traduzione locale con MarianMT (HuggingFace Transformers).

    Flusso:
      input_queue  → riceve list[TextRegion] da OCRWorker
      output_queue → produce list[tuple[bbox, translated_text]] verso OverlayWindow

    Ottimizzazioni:
      - LRU cache: frasi già tradotte restituite istantaneamente
      - Batch translation: raggruppa le frasi nuove e le traduce in una sola
        chiamata al modello (più efficiente di una chiamata per stringa)
      - Modello in memoria per tutta la sessione (no reload tra frame)

    TODO Phase 3: implementare i metodi marcati con # TODO
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

        # La cache è una funzione interna (vedi _translate_cached sotto)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._translations_total: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Carica il modello e avvia il thread consumer."""
        logger.info(
            "TranslationWorker: caricamento modello %s ...",
            config.marian_model_name,
        )
        self._load_model()
        threading.Thread(
            target=self._consume_loop, daemon=True, name="TranslationWorker"
        ).start()
        logger.info("TranslationWorker avviato")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("TranslationWorker fermato")

    # ── Model loading ─────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        TODO Phase 3 — Carica MarianMT dal disco o lo scarica da HuggingFace.

        Logica:
          1. Se config.marian_model_local_path esiste → carica da lì (offline)
          2. Altrimenti → scarica da HuggingFace e salva in models/
          3. Assegna self._model e self._tokenizer

        Esempio:
            from transformers import MarianMTModel, MarianTokenizer

            model_path = str(config.marian_model_local_path)
            if config.marian_model_local_path.exists():
                source = model_path
            else:
                logger.info("Download modello da HuggingFace: %s", config.marian_model_name)
                source = config.marian_model_name

            self._tokenizer = MarianTokenizer.from_pretrained(source)
            self._model = MarianMTModel.from_pretrained(source)

            # Salva localmente per uso offline futuro
            if source != model_path:
                self._tokenizer.save_pretrained(model_path)
                self._model.save_pretrained(model_path)
                logger.info("Modello salvato in %s", model_path)
        """
        raise NotImplementedError("Phase 3: implementare _load_model()")

    # ── Consumer loop ─────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """
        TODO Phase 3 — Loop: prende list[TextRegion], traduce, emette in output_queue.

        Logica per ogni batch:
          1. Separa le stringhe già in cache da quelle nuove
          2. Traduce in batch le stringhe nuove
          3. Ricompone l'output: [(bbox, translated_text)]
          4. Emette in output_queue
        """
        raise NotImplementedError("Phase 3: implementare _consume_loop()")

    # ── Translation ───────────────────────────────────────────────

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        TODO Phase 3 — Traduce una lista di stringhe, sfruttando la LRU cache.

        Args:
            texts: Stringhe nella lingua sorgente (già filtrate da OCR)

        Returns:
            Lista di traduzioni nello stesso ordine.

        Logica suggerita:
            results = []
            to_translate = []   # stringhe non in cache
            to_translate_idx = []

            for i, t in enumerate(texts):
                cached = self._translate_cached(t)
                if cached is not None:
                    results.append((i, cached))
                    self._cache_hits += 1
                else:
                    to_translate.append(t)
                    to_translate_idx.append(i)
                    self._cache_misses += 1

            if to_translate:
                translated = self._run_model(to_translate)
                for i, (orig, trans) in zip(to_translate_idx, zip(to_translate, translated)):
                    self._store_cache(orig, trans)
                    results.append((i, trans))

            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]
        """
        raise NotImplementedError("Phase 3: implementare translate_batch()")

    def _run_model(self, texts: list[str]) -> list[str]:
        """
        TODO Phase 3 — Inferenza MarianMT su una lista di stringhe.

        Esempio:
            inputs = self._tokenizer(texts, return_tensors="pt",
                                     padding=True, truncation=True, max_length=512)
            translated = self._model.generate(**inputs)
            return [self._tokenizer.decode(t, skip_special_tokens=True)
                    for t in translated]
        """
        raise NotImplementedError("Phase 3: implementare _run_model()")

    # ── LRU Cache ─────────────────────────────────────────────────
    # Implementata come dict interno invece di @lru_cache perché
    # @lru_cache non funziona su metodi (self non è hashable).

    def _translate_cached(self, text: str) -> Optional[str]:
        """Restituisce la traduzione dalla cache o None se assente."""
        return getattr(self, "_cache", {}).get(text)

    def _store_cache(self, original: str, translation: str) -> None:
        """Salva una traduzione nella cache (con limite LRU manuale)."""
        if not hasattr(self, "_cache"):
            self._cache: dict[str, str] = {}
        # Rispetta il limite di dimensione (FIFO semplice se piena)
        if len(self._cache) >= config.translation_cache_maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[original] = translation

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "translations_total": self._translations_total,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_pct": round(
                self._cache_hits / max(1, total) * 100, 1
            ),
        }
