"""Typed error hierarchy.

Every user-facing failure should raise a :class:`ShadowBenchError` subclass rather than a bare exception,
so the CLI and IPC layers can render clean, actionable messages instead of tracebacks.
"""

from __future__ import annotations


class ShadowBenchError(Exception):
    """Base class for all ShadowBench errors."""


class ProfilerError(ShadowBenchError):
    """Hardware detection or benchmarking failed."""


class UnsupportedHardwareError(ProfilerError):
    """No usable GPU backend was found and CPU-only fallback is not viable."""


class PredictorError(ShadowBenchError):
    """A prediction could not be produced for the given inputs."""


class ModelNotFoundError(PredictorError):
    """The requested model is absent from the catalog."""


class OrchestratorError(ShadowBenchError):
    """Downloading or launching a local inference engine failed."""


class PoolError(ShadowBenchError):
    """Peer discovery, transport, or routing failed."""


class StorageError(ShadowBenchError):
    """The local datastore could not be read or written."""
