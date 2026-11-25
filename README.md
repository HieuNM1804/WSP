# Weight Space Programming for sketch-based image retrieval

WSP lets an editor model inspect and modify CLIP ViT-B/32 weights, using seen Sketchy classes for experiments and unseen classes for evaluation. The default editor is `gpt-6.2-astra` at `max`.

- [WSP harness and local WSL setup](wsp_sbir/README.md)
- [CLIP retrieval baseline](clip_zs_sbir/README.md)
- [Colab notebook](https://colab.research.google.com/github/HieuNM1804/WSP/blob/main/colab.ipynb)

The notebook downloads the public Kaggle Sketchy v1 archive directly into the GPU runtime and verifies the 125-category layout. Its setup cell checks `bubblewrap`, reuses Colab's CUDA PyTorch, and prepares the model and task files for both Sketchy splits. Before launching WSP, sign in to the notebook's Linux Codex CLI with an account that can use the editor model.

## Results branches

Every completed run creates `wsp_sbir/runs/<run>/result.zip` after evaluation. The bundle includes the prompts, per-turn replies and CLI events, token usage and tool-free audits, Python and forward requests and outputs, readable trajectory, checkpoint and final metrics, and the exact edited tensors with manifests. The archive replaces thread IDs and machine paths with placeholders and excludes authentication files, the dataset, base weights, and intermediate activations. This follows the [WSP release structure](https://github.com/RikiyaT/lm_understand-logs#files).

Use a Git checkout with write access to publish the bundle. Set `publish_to_github = True` in the Colab run cell to publish as soon as the run finishes if Git is already authenticated there. You can also download the bundle while the runtime is live and publish from your local checkout.

```bash
python3 wsp_sbir/publish_results.py --from-bundle /path/to/result.zip
python3 wsp_sbir/publish_results.py --from-bundle /path/to/result.zip --push
```

Each run is published to branch `results/<run>`, with files under `results/<run>/`. Open `summary.json` for metrics and `readable/trajectory.md` for the conversation. To rebuild the archive after a delayed evaluation, use `python3 wsp_sbir/publish_results.py --run wsp_sbir/runs/<run> --push`.
