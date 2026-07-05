"""Shared vocabulary: enums and the quantization table. Larger data contracts live in each module's ``models.py``."""

from __future__ import annotations

from enum import Enum


class ModelTopology(str, Enum):
    """Model architecture family — drives the Dense vs. MoE prediction branch."""

    DENSE = "dense"
    MOE = "moe"


class Task(str, Enum):
    """User intent for filtering candidate models."""

    CODING = "coding"
    CHAT = "chat"
    REASONING = "reasoning"
    GENERAL = "general"


class UserProfile(str, Enum):
    """Optimization preference: raw speed vs. maximum capability."""

    SPEED = "speed"
    INTELLIGENCE = "intelligence"


class Quantization(str, Enum):
    """Common GGUF quantization levels. ``effective_bpw`` includes block metadata overhead."""

    Q2_K = "Q2_K"
    Q3_K_M = "Q3_K_M"
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    FP16 = "FP16"

    @property
    def effective_bpw(self) -> float:
        """Effective bits per weight, including k-quant block overhead."""
        return _EFFECTIVE_BPW[self]


_EFFECTIVE_BPW: dict[Quantization, float] = {
    Quantization.Q2_K: 3.35,
    Quantization.Q3_K_M: 3.91,
    Quantization.Q4_K_M: 4.85,
    Quantization.Q5_K_M: 5.69,
    Quantization.Q6_K: 6.56,
    Quantization.Q8_0: 8.50,
    Quantization.FP16: 16.0,
}


class KVCacheQuantization(str, Enum):
    """KV-cache quantization precision (``--cache-type-k`` / ``--cache-type-v``)."""

    FP16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    @property
    def bytes_per_elem(self) -> float:
        return _KV_CACHE_BYTES[self]


_KV_CACHE_BYTES: dict[KVCacheQuantization, float] = {
    KVCacheQuantization.FP16: 2.0,
    KVCacheQuantization.Q8_0: 1.0,
    KVCacheQuantization.Q4_0: 0.5,
}
