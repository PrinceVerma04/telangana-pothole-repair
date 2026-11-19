# app_data/ — what the running app writes (D43)

| Path | What |
|---|---|
| `gunthameter.db` | the demo SQLite database (`DATABASE_URL` in `.env`) |
| `store/` | stored frames and files (`STORAGE_ROOT`) |
| `capture/` | phone drives from `make capture` (frames may show faces / plates: keep local) |
| `playground/` | uploads to the model test page |
| `demo/`, `fleet_sim/` | simulated demo and fleet runs |
| `backups/` | database backups made before schema refreshes |
| `tiles/` | map tiles (`gis/build_tiles.sh`) |
