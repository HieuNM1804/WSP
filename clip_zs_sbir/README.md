# Zero-shot CLIP ViT-B/32 on KD-SBIR unseen splits

Inference only: the original OpenAI CLIP ViT-B/32 (`../ViT-B-32.pt`), frozen, with no prompts and no training.
Sketch queries retrieve photo galleries from the unseen classes in `src/data_config.py`.

Matches KD-SBIR (https://github.com/HieuNM1804/KD-SBIR, commit `741ea27`):

| Part | Source |
|---|---|
| `clip/`, `src/data_config.py`, `src/dataset.py` | copied unchanged |
| `src/metrics.py` | `_retrieval_metrics` copied unchanged from `src/model.py` |
| model loading | `_load_clip_model` (jit -> state_dict -> `build_model`, fp16), from a local path |
| features | `encode_student_image` with no prompts: `visual(x.type(dtype))`, L2-normalised |
| loaders / seeds | `get_loaders` (validation part) + `seed_everything` from `src/train.py` |

Metrics: `sketchy_1`, `tuberlin` -> mAP@all, P@100; `sketchy_2` -> mAP@200, P@200; `quickdraw` -> mAP@all, P@200.

```bash
.venv/bin/python eval_zeroshot.py --root ../Sketchy  --dataset sketchy_1 --output results_sketchy_1.json
.venv/bin/python eval_zeroshot.py --root ../Sketchy  --dataset sketchy_2 --output results_sketchy_2.json
.venv/bin/python eval_zeroshot.py --root ../tuberlin --dataset tuberlin  --output results_tuberlin.json
```
