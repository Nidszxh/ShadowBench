"""Hardware data contract produced by the Profiler and consumed by the Predictor."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Bump when the profile shape changes so synced records remain interpretable.
PROFILE_SCHEMA_VERSION = 1


class GpuInfo(BaseModel):
    """A single detected GPU (or unified-memory device on Apple Silicon)."""

    vendor: str = Field(description="nvidia | intel | apple | amd | cpu")
    name: str
    vram_total_mb: int
    vram_free_mb: int
    temperature_c: float | None = None
    driver_version: str | None = None


class SystemInfo(BaseModel):
    """CPU and system-memory snapshot."""

    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_total_mb: int
    ram_available_mb: int


class BandwidthResult(BaseModel):
    """Measured (not spec-sheet) data-transfer and compute throughput."""

    cpu_matmul_gbps: float
    device_compute_tflops: float
    duration_s: float
    system_ram_gbps: float = 0.0


class HardwareProfile(BaseModel):
    """Complete machine profile: the Profiler's public output."""

    schema_version: int = PROFILE_SCHEMA_VERSION
    system: SystemInfo
    gpu: GpuInfo | None = None
    bandwidth: BandwidthResult | None = None

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None

    def anonymized(self) -> dict[str, object]:
        """Return a PII-free, bucketed dict safe for the public calibration dataset."""
        return {
            "schema_version": self.schema_version,
            "gpu_vendor": self.gpu.vendor if self.gpu else "cpu",
            "gpu_name": self.gpu.name if self.gpu else None,
            "vram_bucket_gb": _bucket_gb(self.gpu.vram_total_mb) if self.gpu else 0,
            "ram_bucket_gb": _bucket_gb(self.system.ram_total_mb),
            "cpu_matmul_gbps": round(self.bandwidth.cpu_matmul_gbps, 1) if self.bandwidth else None,
            "compute_tflops": round(self.bandwidth.device_compute_tflops, 1)
            if self.bandwidth
            else None,
        }


def _bucket_gb(mb: int) -> int:
    """Round a megabyte value to the nearest 4 GB (decimal) to avoid exact fingerprinting."""
    gb = mb / 1000
    return round(gb / 4) * 4
