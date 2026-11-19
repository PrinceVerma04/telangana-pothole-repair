# app_data/ — what the running app writes (D43)

Not in git (drives can show faces and number plates: keep them local).

| Path | What |
|---|---|
| `gunthameter.db` | the seeded demo SQLite database used by `make api` (`DATABASE_URL` in `.env`) |
| `store/` | stored frames and files (`STORAGE_ROOT`) |
| `capture/` | real phone drives from `make capture` / `make capture-public`, with per-frame results |
| `demo/` | demo runs: `make demo` (simulated fleet) and `make e2e-demo` (dummy phone drive); each uses its own copy of the database, so `gunthameter.db` is never changed |
| `playground/` | uploads to the model test page |
| `fleet_sim/` | simulated fleet runs |
| `backups/` | database backups made before schema refreshes |
| `restructure_moves.json`, `restructure_undo.sh` | record of the D43 folder move and its undo script |
