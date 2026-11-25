# Weight Space Programming for sketch-based image retrieval

WSP lets an editor model inspect and modify CLIP ViT-B/32 weights, using seen Sketchy classes for experiments and unseen classes for evaluation. The default editor is `gpt-6.2-astra` at `max`.

- [WSP harness and local WSL setup](wsp_sbir/README.md)
- [CLIP retrieval baseline](clip_zs_sbir/README.md)
- [Colab notebook](https://colab.research.google.com/github/HieuNM1804/WSP/blob/main/colab.ipynb)

The notebook downloads the public Kaggle Sketchy v1 archive directly into the GPU runtime and verifies the 125-category layout. Its setup cell checks `bubblewrap`, reuses Colab's CUDA PyTorch, and prepares the model and task files for both Sketchy splits. Before launching WSP, sign in to the notebook's Linux Codex CLI with an account that can use the editor model.

## Results branches

From the source checkout with GitHub write access, preview the evaluated metrics and then publish them if ready.

```bash
python3 wsp_sbir/publish_results.py --run wsp_sbir/runs/s2_astra_50
python3 wsp_sbir/publish_results.py --run wsp_sbir/runs/s2_astra_50 --push
```

The publishing command updates `results/s2_astra_50` and writes `results/s2_astra_50/summary.json`. It includes the source commit, run progress, and checkpoint and final evaluation metrics. Keep the run directory and Codex session state in persistent storage if you need to resume after a Colab runtime disconnects.
