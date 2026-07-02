-- ShadowBench local datastore (DATAFLOW.md §5).
-- Forward-only; see storage/migrations/ for versioned changes.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hardware_profiles (
    id                   TEXT PRIMARY KEY,           -- UUID
    gpu_model            TEXT,
    vram_total_mb        INTEGER,
    system_ram_gb        INTEGER,
    pcie_bandwidth_gbps  REAL,                        -- measured via live stress test
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                   TEXT PRIMARY KEY,           -- UUID
    hardware_profile_id  TEXT NOT NULL REFERENCES hardware_profiles(id) ON DELETE CASCADE,
    model_name           TEXT NOT NULL,
    quantization         TEXT NOT NULL,
    context_length       INTEGER NOT NULL,
    predicted_tps        REAL,
    actual_tps           REAL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_profile ON benchmark_runs(hardware_profile_id);
CREATE INDEX IF NOT EXISTS idx_runs_model   ON benchmark_runs(model_name, quantization);
