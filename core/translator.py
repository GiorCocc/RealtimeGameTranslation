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
import multiprocessing
import time
import queue
from collections import OrderedDict
from multiprocessing import Queue
from typing import Optional, TYPE_CHECKING

from config import config
from core.gpu_utils import detect_device, get_optimal_device, check_vram_available

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
# Translation Engine
# ══════════════════════════════════════════════════════════════════

class MarianTranslatorEngine:
    """
    Motore di traduzione puro: gestisce esclusivamente il caricamento del
    modello MarianMT e l'inferenza tramite CTranslate2.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.device: str = "cpu"

    def _resolve_device(self) -> str:
        mode = config.translation_gpu_device
        if mode == "auto":
            device = get_optimal_device(vram_required_mb=400)
            logger.info("MarianTranslatorEngine: device auto-selezionato → %s", device)
            return device
        if mode == "cuda":
            device = detect_device()
            if device != "cuda":
                logger.warning(
                    "MarianTranslatorEngine: CUDA richiesto ma non disponibile, fallback a CPU"
                )
                return "cpu"
            return "cuda"
        if mode == "directml":
            logger.warning("MarianTranslatorEngine: DirectML non supportato da CTranslate2, fallback a CPU")
            return "cpu"
        if mode != "cpu":
            logger.warning(
                "MarianTranslatorEngine: translation_gpu_device='%s' non riconosciuto, uso CPU",
                mode
            )
        return "cpu"

    def load_model(self) -> None:
        from transformers import MarianTokenizer
        import ctranslate2

        self.device = self._resolve_device()
        local_path = config.marian_model_local_path

        if not local_path.exists():
            logger.info(
                "MarianTranslatorEngine: modello locale assente, download da HF e conversione: %s",
                config.marian_model_name
            )
            model_name = config.marian_model_name
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            # Converte il modello da HuggingFace al formato CT2
            converter = ctranslate2.converters.TransformersConverter(model_name)
            local_path.mkdir(parents=True, exist_ok=True)
            converter.convert(output_dir=str(local_path), force=True)
            
            # Salva il tokenizer per l'uso offline
            tokenizer.save_pretrained(str(local_path))
            logger.info("MarianTranslatorEngine: modello salvato in %s", local_path)
        else:
            logger.info("MarianTranslatorEngine: modello locale trovato in %s", local_path)

        self._tokenizer = MarianTokenizer.from_pretrained(str(local_path))
        
        compute_type = "float16" if (self.device == "cuda" and config.translation_use_fp16) else "default"
        self._model = ctranslate2.Translator(
            str(local_path),
            device=self.device,
            compute_type=compute_type
        )
        logger.info("MarianTranslatorEngine: CTranslate2 Translator caricato su %s", self.device)

    def unload_model(self) -> None:
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("MarianTranslatorEngine: modello scaricato dalla VRAM")

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modello non caricato. Chiamare load_model() prima di translate().")

        # Tokenize
        input_tokens = [self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(t)) for t in texts]

        # Standard batch size
        batch_size = 16
        translations = [None] * len(texts)
        
        # We translate in chunks
        start_idx = 0
        while start_idx < len(input_tokens):
            end_idx = min(start_idx + batch_size, len(input_tokens))
            chunk = input_tokens[start_idx:end_idx]
            try:
                results = self._model.translate_batch(
                    chunk,
                    beam_size=config.translation_num_beams,
                    max_decoding_length=config.translation_max_new_tokens
                )
                for i, res in enumerate(results):
                    output_ids = self._tokenizer.convert_tokens_to_ids(res.hypotheses[0])
                    decoded = self._tokenizer.decode(output_ids, skip_special_tokens=True)
                    translations[start_idx + i] = decoded
                start_idx = end_idx
            except Exception as exc:
                exc_str = str(exc).lower()
                is_oom = "out of memory" in exc_str or "oom" in exc_str or "cuda" in exc_str
                if is_oom and batch_size > 1:
                    logger.warning(
                        "MarianTranslatorEngine: Out of memory durante traduzione. "
                        "Svuoto cache CUDA e dimezzo batch size (%d -> %d). Errore: %s",
                        batch_size, batch_size // 2, exc
                    )
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    batch_size = max(1, batch_size // 2)
                    continue
                else:
                    logger.error("MarianTranslatorEngine: errore durante translate_batch: %s", exc)
                    raise exc
        return translations  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════
# Translation Worker
# ══════════════════════════════════════════════════════════════════

class TranslationWorker:
    """
    Worker per traduzione locale con MarianMT (HuggingFace Transformers + CTranslate2).
    """

    def __init__(self, input_queue: Queue, output_queue: Queue) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self._stop_event = multiprocessing.Event()
        self._pause_event = multiprocessing.Event()
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._model_unloaded: bool = False

        # Stats
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._translations_total: int = 0
        self._batches_processed: int = 0
        self._translation_times: list[float] = []
        self._t_start: float = 0.0

        self._stats_queue = multiprocessing.Queue(maxsize=2)
        self._last_stats_push: float = 0.0
        self._cached_stats = {
            "translations_total": 0,
            "batches_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_size": 0,
            "cache_hit_rate_pct": 0.0,
            "avg_translation_ms": 0.0,
            "device": "cpu",
        }

        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=self._run_process, daemon=True, name="TranslationWorker"
        )
        self._process.start()
        logger.info("TranslationWorker avviato")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("TranslationWorker arrestato")

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._process is not None:
            self._process.join(timeout)

    @property
    def stats(self) -> dict:
        while not self._stats_queue.empty():
            try:
                self._cached_stats = self._stats_queue.get_nowait()
            except queue.Empty:
                break
        return self._cached_stats

    def _run_process(self) -> None:
        # Inizializza il motore di traduzione nello spazio di indirizzamento del subprocesso
        self.engine = MarianTranslatorEngine()
        self.engine.load_model()

        self._t_start = time.perf_counter()
        self._last_stats_push = self._t_start

        self._consume_loop()

    def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                if self._pause_event.is_set():
                    if not self._model_unloaded:
                        self.engine.unload_model()
                        self._model_unloaded = True
                else:
                    if self._model_unloaded:
                        self.engine.load_model()
                        self._model_unloaded = False
                continue

            # Svuota buffer e tieni solo l'ultimo per azzerare la latenza
            while True:
                try:
                    item = self.input_queue.get_nowait()
                except queue.Empty:
                    break

            if isinstance(item, tuple) and len(item) == 2:
                regions, capture_time = item
            else:
                regions = item
                capture_time = time.perf_counter()

            if not regions:
                logger.debug("TranslationWorker: batch vuoto ricevuto (nessun testo da tradurre)")
                self._emit([], capture_time)
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
                    self._emit(list(accumulated_output), capture_time)
                    logger.debug(
                        "TranslationWorker: chunk tradotto ed emesso — %d/%d region/i completate finora",
                        len(accumulated_output), len(regions)
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

    def _emit(self, output: list, capture_time: float) -> None:
        payload = (output, capture_time)
        try:
            self.output_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(payload)
            except queue.Full:
                pass

        now_t = time.perf_counter()
        if now_t - self._last_stats_push >= 1.0:
            try:
                self._stats_queue.put_nowait(self._compute_stats())
            except queue.Full:
                pass
            self._last_stats_push = now_t

    def translate_batch(self, texts: list[str]) -> list[str]:
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
            translated = self.engine.translate(to_translate)
            for idx, orig, trans in zip(to_translate_idx, to_translate, translated):
                self._store_cache(orig, trans)
                results[idx] = trans

        self._translations_total += len(texts)
        return results  # type: ignore[return-value]

    def _translate_cached(self, text: str) -> Optional[str]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        return None

    def _store_cache(self, original: str, translation: str) -> None:
        if original in self._cache:
            self._cache[original] = translation
            self._cache.move_to_end(original)
            return
        if len(self._cache) >= config.translation_cache_maxsize:
            self._cache.popitem(last=False)
        self._cache[original] = translation

    def _compute_stats(self) -> dict:
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
            "device": self.engine.device,
        }