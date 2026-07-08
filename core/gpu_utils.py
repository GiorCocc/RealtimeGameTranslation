"""
gpu_utils.py — Rilevamento centralizzato GPU e gestione risorse.

Fornisce utilità per individuare la GPU disponibile (CUDA),
verificare la VRAM libera e scegliere automaticamente il device
ottimale per OCR e traduzione.  Fallback trasparente su CPU
quando CUDA non è disponibile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Import condizionale di torch ─────────────────────────────────
# torch potrebbe non essere installato o non avere il supporto CUDA;
# le funzioni del modulo gestiscono entrambi i casi senza propagare
# eccezioni di import.
_TORCH_AVAILABLE: bool = False
_CUDA_AVAILABLE: bool = False

try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]

_DML_AVAILABLE: bool = False
try:
    import onnxruntime as ort
    if "DmlExecutionProvider" in ort.get_available_providers():
        _DML_AVAILABLE = True
except ImportError:
    pass

logger.debug(
    "torch disponibile: %s  |  CUDA disponibile: %s  |  DirectML disponibile: %s",
    _TORCH_AVAILABLE,
    _CUDA_AVAILABLE,
    _DML_AVAILABLE,
)


# ── DeviceInfo dataclass ─────────────────────────────────────────

@dataclass(frozen=True)
class DeviceInfo:
    """Informazioni sul device di calcolo rilevato (GPU o CPU)."""

    backend: str                          # "cuda", "directml", oppure "cpu"
    device_name: str                      # es. "NVIDIA GeForce RTX 4070", "DirectML GPU" o "CPU"
    vram_total_mb: int                    # 0 per CPU
    vram_free_mb: int                     # 0 per CPU
    compute_capability: tuple[int, int] | None  # Solo per device CUDA


# ── Funzioni pubbliche ───────────────────────────────────────────

def detect_device() -> DeviceInfo:
    """Rileva il device di calcolo disponibile.

    Tenta di individuare una GPU NVIDIA con supporto CUDA tramite
    ``torch.cuda``. Se non c'è CUDA, controlla se DirectML è
    disponibile (per schede AMD/Intel su Windows). Altrimenti fallback su CPU.
    """
    if _CUDA_AVAILABLE:
        assert torch is not None
        device_index: int = torch.cuda.current_device()
        device_name: str = torch.cuda.get_device_name(device_index)
        cap: tuple[int, int] = torch.cuda.get_device_capability(device_index)

        total_bytes: int = torch.cuda.get_device_properties(device_index).total_mem
        free_bytes: int
        free_bytes, _ = torch.cuda.mem_get_info(device_index)

        total_mb: int = int(total_bytes / (1024 * 1024))
        free_mb: int = int(free_bytes / (1024 * 1024))

        info = DeviceInfo(
            backend="cuda",
            device_name=device_name,
            vram_total_mb=total_mb,
            vram_free_mb=free_mb,
            compute_capability=cap,
        )
        logger.info(
            "GPU rilevata: %s  |  VRAM totale: %d MB  |  VRAM libera: %d MB  |  "
            "Compute Capability: %s",
            info.device_name, info.vram_total_mb, info.vram_free_mb, f"{cap[0]}.{cap[1]}",
        )
        return info

    if _DML_AVAILABLE:
        logger.debug("DirectML disponibile — accelerazione hardware AMD/Intel attiva.")
        return DeviceInfo(
            backend="directml",
            device_name="DirectML GPU",
            vram_total_mb=4096,  # Valore fittizio, onnxruntime non espone facile API
            vram_free_mb=4096,
            compute_capability=None,
        )

    logger.info("CUDA e DirectML non disponibili — fallback a CPU.")
    return DeviceInfo(
        backend="cpu",
        device_name="CPU",
        vram_total_mb=0,
        vram_free_mb=0,
        compute_capability=None,
    )


def get_optimal_device(vram_required_mb: int = 500) -> str:
    """Restituisce il device ottimale per l'inferenza."""
    info: DeviceInfo = detect_device()

    if info.backend == "cpu":
        return "cpu"

    if info.backend == "directml":
        return "directml"

    if info.vram_free_mb < vram_required_mb:
        logger.warning(
            "VRAM libera insufficiente (%d MB < %d MB richiesti) — "
            "si utilizza la CPU.",
            info.vram_free_mb,
            vram_required_mb,
        )
        return "cpu"

    logger.debug(
        "Device ottimale selezionato: cuda  (VRAM libera: %d MB ≥ %d MB)",
        info.vram_free_mb,
        vram_required_mb,
    )
    return "cuda"


def get_gpu_info() -> dict[str, object]:
    """Restituisce un dizionario con le informazioni GPU.

    Pensato per essere consumato dal pannello Settings (Phase 6).
    Se CUDA non è disponibile restituisce comunque un dict con
    valori di fallback coerenti.

    Returns
    -------
    dict[str, object]
        Chiavi: ``backend``, ``device_name``, ``vram_total_mb``,
        ``vram_free_mb``, ``driver_version``, ``compute_capability``.
    """
    info: DeviceInfo = detect_device()

    driver_version: str = "N/A"
    if _CUDA_AVAILABLE:
        assert torch is not None
        try:
            # torch.version.cuda restituisce la versione CUDA runtime
            driver_version = str(torch.version.cuda)
        except Exception:  # noqa: BLE001
            logger.debug("Impossibile ottenere la versione del driver CUDA.")

    return {
        "backend": info.backend,
        "device_name": info.device_name,
        "vram_total_mb": info.vram_total_mb,
        "vram_free_mb": info.vram_free_mb,
        "driver_version": driver_version,
        "compute_capability": info.compute_capability,
    }


def check_vram_available(required_mb: int) -> bool:
    """Verifica rapida della disponibilità di VRAM.

    Parameters
    ----------
    required_mb : int
        VRAM minima richiesta in MB.

    Returns
    -------
    bool
        ``True`` se il device CUDA dispone di almeno *required_mb*
        di VRAM libera, ``False`` altrimenti (o se CUDA non è
        disponibile).
    """
    info: DeviceInfo = detect_device()

    if info.backend == "cpu":
        logger.debug(
            "check_vram_available(%d) → False  (backend CPU, nessuna VRAM).",
            required_mb,
        )
        return False

    available: bool = info.vram_free_mb >= required_mb
    logger.debug(
        "check_vram_available(%d) → %s  (VRAM libera: %d MB)",
        required_mb,
        available,
        info.vram_free_mb,
    )
    return available
