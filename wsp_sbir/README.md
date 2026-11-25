# WSP for zero-shot SBIR: an LLM edits CLIP ViT-B/32's weights

## Local setup on Ubuntu WSL

Run from `/mnt/d/WSP`. GPU jobs and the driver use `clip_zs_sbir/.venv` (CUDA 12.4).
The sandbox reuses `~/.venvs/wsp-clip-cpu` when available, with `wsp_sbir/.venv_sandbox` as the default fallback.
`WSP_SANDBOX_VENV` selects a different CPU venv. The sandbox uses Ubuntu's Python and `bubblewrap`.
Codex can use the Windows `codex.exe` already logged into ChatGPT through WSL interop, or a Linux Codex CLI
selected with `WSP_CODEX_BIN`.

```bash
# Install the local venvs and prepare both Sketchy splits (Ubuntu Python 3.12).
bash wsp_sbir/setup_local.sh

# Rebuild a single split if needed. Use sketchy_1 for the split in examples/.
clip_zs_sbir/.venv/bin/python wsp_sbir/prepare.py \
  --root /mnt/d/Sketchy --dataset sketchy_2 \
  --weights /mnt/c/Users/Admin/.cache/clip/ViT-B-32.pt

# A new run with a budget of 50 forward passes and checkpoints at 10 and 25.
bash wsp_sbir/run_local.sh --task tasks/sketchy_2 --run runs/s2_astra_50 --budget 50
bash wsp_sbir/run_local.sh --run runs/s2_astra_50 --resume
```

`run_local.sh` selects `gpt-6.2-astra` at `max` and sets the RAM, GPU, and disk limits for this machine.
Use the direct driver with `--backend claude-code` for new Claude runs.

### Space for 50 forward passes

These estimates use the local Sketchy files. A full-practice retrieval without optional activations stores a
512-value embedding and a 768-value pre-projection vector per image, both in float16. The table excludes
small JSON and CSV outputs, logs, and weight snapshots.

| Split | Practice images | One full pass, one model | 50 full passes, one model | 50 full passes, original + edited |
|---|---:|---:|---:|---:|
| `sketchy_1` | 124,499 | 0.30 GiB | 14.84 GiB | 29.68 GiB |
| `sketchy_2` | 130,532 | 0.31 GiB | 15.56 GiB | 31.12 GiB |

For `sketchy_2`, one full retrieval with all residual and MLP layers and CLS/patch summaries would write
about 23.09 GiB for one model. The local launcher limits each forward pass to 6 GiB and budgets up to 24 GiB
in the run's `work/`, keeping a 12 GiB free-space reserve when allocating forward-pass output.
Use subsets of images or layers, or remove saved forward-pass outputs as they become unnecessary.
Each checkpoint and the final submission save a separate copy of the edited tensors outside `work/`.
A copy of every visual tensor would be 0.33 GiB in float32, or 0.56 GiB for every editable tensor.
With the default two checkpoints and the final submission, allow approximately 26 GiB for a run that fills
the 24 GiB work budget, plus the 12 GiB free-space reserve. The run length does not multiply peak RAM or VRAM
because the harness runs one job at a time and streams activation arrays to disk.

## Protocol

This is a harness in the style of *Sparks of Weight Space Programming*, [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math),
applied to zero-shot sketch-based image retrieval. An editor LLM reads the weights of CLIP ViT-B/32's image encoder and
edits them directly, with no training and no gradients. The goal is better sketch→photo retrieval on unseen
categories. The final score is computed with KD-SBIR's protocol (`../clip_zs_sbir`), and the editor never sees it.

| | Exp 8 (logs) | This harness |
|---|---|---|
| Editor | `gpt-6-astra` via `codex exec` / `codex exec resume`, tools off | `gpt-6.2-astra` at `max` via Codex CLI, tools off. The bundled example used Claude Code. |
| Network | Qwen3.5-0.8B | CLIP ViT-B/32 (`../ViT-B-32.pt`); every tensor is editable (only `visual.*` affects the retrieval); the only checks are shape and finite values |
| Actions | one ```python block (persistent CPU kernel, 300 runs) or one ```forward block (budget 10/25/50) per reply | same |
| Forward pass | `generate` / `teacher_forced` | `retrieve` (up to every practice image, KD-SBIR metric, `both` = paired SE over queries and categories, optional activations streamed to disk) / `read` (≤16,384 images, ≤4,096 texts: residual stream, MLP, attention) / `stats` (per-group means and within-group covariances over up to every practice image, accumulated on the GPU) |
| Practice data | GSM8K / MATH train | the split's **seen** categories only (KD-SBIR's `TrainDataset` rule); unseen categories are never mounted, and the forward tool refuses them |
| Evaluation | MATH-500, GSM8K… | `sketchy_2` unseen split (21 categories): mAP@200, P@200, through `clip_zs_sbir`'s KD-SBIR code |
| Enforced | explanation before each block (2 refusals max in a row), handoff every 40 turns with notebook, notices, duplicate-request refusal, per-turn tool-free audit | same. The Codex audit checks the JSONL event stream before accepting a reply. |

The task prompt is [tasks/sketchy_2/message.txt](tasks/sketchy_2/message.txt) (`sbir-v3`). It keeps the structure and
rules of Exp 8's v3 prompt, with EXPLAIN EVERY STEP from turn 1. On top of that it adds background on CLIP,
retrieval as a downstream task and the zero-shot sketch setting. It also catalogs possible causes of bad rankings at
every depth and every kind of lever over the whole encoder, and it is not limited to closing the domain gap. Earlier
versions: `message_v1.txt`, and `message_v2_filled.txt` (the run `b50_r1`). The budget and size values are filled in
per run.

## Layout

```
prepare.py                 export model/ and tasks/<split>/practice/, build data/<root>_clean (eval root)
harness/driver.py          turn loop, rules, state, logs, handoffs, notices, final export
harness/backend_codex.py   one tool-free Codex CLI call per turn + audit
harness/backend_claude.py  optional Claude Code backend
harness/kernel_client.py   bubblewrap sandbox (no network, no GPU, /model /practice ro, /work rw), timeouts, restarts
harness/sandbox_lib/       kernel.py, prelude.py (NAMES, load, save_edit, ...), edit_rules.py (shared with GPU side)
harness/forward.py         GPU forward-pass tool (uses ../clip_zs_sbir: build_model, transforms, metric loop)
harness/export_readable.py runs/<run>/readable/ in the format of the released p50c/readable
harness/resources.py       RAM / GPU / disk checks, watchdog floor, OOM-first and nice for our processes
harness/instructions.md    the neutral system prompt, "Answer the user's request." (as in the logs)
eval/evaluate_edit.py      final score of a run's edit on the unseen split (KD-SBIR code, unchanged)
eval/unedited/result.json  baseline: mAP@200 0.3635, P@200 0.3385
runs/<run>/                one run, same files as the released logs (see harness/driver.py docstring)
```

## Setup on a Linux host with uv

```bash
# CPU sandbox venv for the kernel (torch CPU, numpy, scipy, scikit-learn, pandas, sympy, networkx, safetensors, pillow)
uv venv .venv_sandbox --python 3.11
uv pip install -p .venv_sandbox/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv_sandbox/bin/python numpy scipy scikit-learn pandas sympy networkx safetensors pillow
# model tensors + practice split + clean eval root
../clip_zs_sbir/.venv/bin/python prepare.py --root ../Sketchy --dataset sketchy_2
```

## Run

```bash
# one 50-pass run; the edit is snapshotted and evaluated on the unseen split when passes 10 and 25 complete,
# and the submitted edit is evaluated at the end (none of these scores is shown to the editor)
setsid nohup python3 harness/driver.py --run runs/b50_r1 --budget 50 > runs/b50_r1.out 2>&1 < /dev/null &

tail -f runs/b50_r1/loop.log                                           # progress, checkpoints, eval lines
python3 harness/driver.py --run runs/b50_r1 --resume                   # after a stop or crash
echo "text" > runs/b50_r1/notices/001_note.txt                         # deliver a note with the next message
python3 harness/export_readable.py runs/b50_r1                         # refresh readable/ during the run
```

The direct driver's defaults are `--model gpt-6.2-astra --effort max`, 300 Python runs of up to 1,800 s and 16 GB, a
handoff every 40 turns, and checkpoints at `--checkpoints 10,25`. Cost is not tracked.

**Checkpoints.** When pass k (10, 25) completes, the driver copies `/work/edited` (the edit that pass measured) to
`runs/<run>/checkpoints/pass_0kk/` (edit/tensors, manifest, notes.md, info.json) and evaluates it with
`eval/evaluate_edit.py`. The result goes to `eval/<run>/pass_0kk.json` and the submitted edit's to
`eval/<run>/final.json`. A checkpoint is a snapshot inside a 50-pass run, so the editor planned for 50 passes and made
no final choice at pass 10 or 25. That differs from Exp 8, where 10, 25 and 50 were separate runs that each ended
with a final choice.

**Per-turn records** (as in the released logs): `turns/tNNN/prompt.txt` is exactly what the editor received that turn.
`answer.md` is its reply. `events.jsonl` is the raw CLI event stream. `run.json` holds the model, effort, the
command, token usage, wall time, and the tool-free audit. `stderr.log` is the CLI's stderr. Thinking summaries
are requested, but in tests Opus 5.5 returned empty thinking blocks (only the thinking-token count). Exp 8 likewise
recorded no reasoning summaries. At the end the driver writes `readable/` (README, prompt, trajectory, passes,
explanations, notes, final answer), as in `exp8_math/runs/p50c/readable`.

**Resources** (harness/resources.py). The machine is shared (vLLM servers, databases, other users' GPU jobs) and its
swap is often full, so the run protects it as follows:
- Every process the harness starts (kernel, forward pass, evaluation) runs at nice 10 with `oom_score_adj` 1000. If
  memory runs out, the OOM killer takes ours first.
- The kernel's address space is capped at `--mem-gb` (16). A watchdog stops a cell when the machine's free RAM falls
  below `--min-free-ram-gb` (8) while the kernel holds at least 2 GB. The editor is told why and to use less memory.
- Before every forward pass and every evaluation, the driver waits until `--forward-ram-gb` (10) of RAM and
  `--forward-gpu-gb` (6) of GPU memory are free. A CUDA out-of-memory error is retried up to 3 times and is not
  counted against the budget.
- **Streaming.** The forward tool writes everything it keeps per image straight into memory-mapped `.npy` files,
  batch by batch. Its RAM therefore does not grow with the number of images (about 1.5 GB of anonymous memory with
  the whole practice set). Statistics passes keep only per-group sums on the GPU. In the kernel, `open_npy` and
  `chunks` read large files a slice at a time.
- **Disk.** `/work` has a quota of `--work-quota-gb` (80), and one pass may write at most `--pass-gb` (40); both are
  stated in the prompt. Passes are refused (not counted) while `/work` is over its quota, or while the disk has less
  than `--min-disk-gb` (50) free. Activations of every practice image at every layer take about 25 GB.
- **CPU.** GPU jobs use 6 CPU threads (`OMP_NUM_THREADS`, `torch.set_num_threads`), 4 DataLoader workers for forward
  passes and 2 for evaluations. Results do not depend on these.
- `resources.jsonl` records RAM, swap, GPU, load, disk, `/work` size and the kernel's memory at every turn.

## Notes

- **Fidelity.** On the same embeddings, the forward tool's per-query metric reproduces KD-SBIR's `_retrieval_metrics`
  (checked for both split rules). `evaluate_edit.py --original` reproduces the earlier baseline exactly.
- **Data quirk (fixed).** `Sketchy/sketch/cow/_contact_sheets/` was not part of Sketchy and has been deleted.
  `data/Sketchy_clean` (a symlink view, 3 MB) is still the eval root, and it holds the same files as `Sketchy`.
- **Archive.** `archive/b50_r1_sbir-v2/` keeps the edits and unseen scores of the first 50-pass run (prompt v2). Its
  turns and logs were deleted. Unseen mAP@200 / P@200: pass 10 0.6515 / 0.6132, pass 25 0.6903 / 0.6546, final
  0.6979 / 0.6636.
- **Rules that cannot be enforced technically** (stated in the prompt, as in Exp 8): no training or iterative
  fitting into weights; no simulating the encoder in the kernel, including scoring transformed embeddings in place of
  a forward pass; no use of the evaluation categories. Read `explanations.md` and `tool_calls.jsonl` to audit them.
