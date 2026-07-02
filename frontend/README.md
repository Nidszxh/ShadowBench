# frontend (Tauri desktop shell)

**Status: Phase 3 (not yet scaffolded).** This directory will hold the Tauri desktop app.

Planned layout (see [`../PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md)):

```
frontend/
├── package.json
├── src/                    # React/TS UI
│   ├── views/              # dashboard screens
│   ├── components/
│   ├── api/                # typed IPC client (talks to the Python sidecar)
│   └── styles/
└── src-tauri/              # Rust
    ├── Cargo.toml
    ├── tauri.conf.json
    └── src/
        ├── main.rs
        ├── commands.rs     # IPC command handlers
        ├── sidecar.rs      # spawn/supervise the bundled Python sidecar
        └── updater.rs      # Tauri auto-updater
```

The IPC contract the UI and sidecar share is defined in
[`../schemas/ipc/ipc.schema.json`](../schemas/ipc/ipc.schema.json) and implemented in
`core/src/shadowbench/ipc/server.py`.

To scaffold in Phase 3:

```bash
npm create tauri-app@latest        # React + TypeScript template
```
