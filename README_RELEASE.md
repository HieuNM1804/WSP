# WSP for zero-shot SBIR: release bundle

- `wsp_sbir/`: the harness (driver, Claude Code backend, bubblewrap kernel, GPU forward-pass tool, resource guards),
  `prepare.py`, `eval/evaluate_edit.py`, and the prompts `tasks/<split>/message.txt` (sketchy_2, sketchy_1, tuberlin,
  quickdraw). See `wsp_sbir/README.md`.
- `clip_zs_sbir/`: the KD-SBIR code the harness imports (CLIP model, data split, loader, metric, `eval_zeroshot.py`).
- `examples/s1_50_opus55max/`: one sample run (Opus 5.5, effort max, sketchy_1, 50 passes): config, prompt, turns
  t001, t002, t005 (first forward pass) and t006, request 1, final answer, readable summary and unseen scores.
  Paths, session ids and account ids are replaced by placeholders (`<repo>`, `<home>`, `<tmp>`, `id-xxxxxxxx`).

Not included (rebuild them): `ViT-B-32.pt`, the datasets (`<root>/{sketch,photo}/<category>/`), `wsp_sbir/model/`
(`prepare.py` exports it), `tasks/*/practice/index.jsonl` and `task.json` (`prepare.py` writes them), the venvs
(`.venv` with CUDA torch, torchvision, torchmetrics, ftfy, regex, tqdm, pillow; `.venv_sandbox` with CPU torch, numpy,
scipy, scikit-learn, pandas, sympy, networkx, safetensors, pillow), and the run work folders.

Paths still assumed by the code: `clip_zs_sbir/` next to `wsp_sbir/`, its venv at `clip_zs_sbir/.venv`, the sandbox
venv at `wsp_sbir/.venv_sandbox`, uv's Pythons at `~/.local/share/uv/python`, `claude` at `~/.local/bin/claude`,
`ViT-B-32.pt` next to the two folders. Requires: Linux, a CUDA GPU, bubblewrap, a logged-in Claude Code CLI.
