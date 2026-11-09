Model weights are mounted read-only at `/models` (set `MODELS_DIR` in `.env`). They are never committed.
Use the same folder as TRIFFID-UAV-Perception to share `sam3.pt` between the two systems.

| File | Used by | Source |
|---|---|---|
| `yoloe-v8l-seg.pt` | `detector.backend: yoloe` (default) | https://huggingface.co/jameslahm/yoloe |
| `sam3.pt`, `bpe_simple_vocab_16e6.txt.gz` | `detector.backend: sam3` | https://huggingface.co/facebook/sam3 (gated); BPE vocab ships with the SAM3 / FSS-SAM3 source |
| `masa_r50.pth` | `tracker.backend: masa` | https://huggingface.co/dereksiyuanli/masa |

Check what a config needs: `docker compose run --rm ugv check`.
