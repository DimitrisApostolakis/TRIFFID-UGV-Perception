Model weights are mounted read-only at `/models` (set `MODELS_DIR` in `.env`). They are never committed.

| File | Used by | Source |
|---|---|---|
| `yoloe-v8l-seg.pt` | detector (always) | https://huggingface.co/jameslahm/yoloe |
| `masa_r50.pth` | `tracker.backend: masa` (fallback only; the default SIFT tracker needs no weights) | https://huggingface.co/dereksiyuanli/masa |

Check what a config needs: `docker compose run --rm ugv check`.
