"""Publish compact WSP evaluation results to the run's own Git branch."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
OUTCOME_FIELDS = (
    "dataset", "unseen_classes", "num_sketches", "num_photos", "mAP@all", "mAP@200", "P@100", "P@200",
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run):
    config = load(run / "config.json")
    state = load(run / "state.json")
    evaluations = {}
    for path in sorted((ROOT / "wsp_sbir" / "eval" / run.name).glob("*.json")):
        if not re.fullmatch(r"final|pass_[0-9]+", path.stem):
            continue
        items = load(path)
        metrics = {
            field: items[field] for field in OUTCOME_FIELDS
            if field in items and (field == "dataset" or type(items[field]) in (int, float))
        }
        if any(field.startswith("mAP@") for field in metrics):
            evaluations[path.stem] = metrics
    if not evaluations:
        raise SystemExit("No completed checkpoint or final evaluation to publish.")

    return {
        "run": run.name,
        "source_commit": git("rev-parse", "HEAD"),
        "dataset": state["task"],
        "model": config["model"],
        "effort": config["effort"],
        "budget": config["budget"],
        "completed_passes": state["passes"],
        "completed_turns": state["turn"],
        "done": state["done"],
        "evaluations": evaluations,
    }


def publish(summary):
    branch = f"results/{summary['run']}"
    ref = f"refs/heads/{branch}"
    remote = git("remote", "get-url", "--push", "origin")
    with TemporaryDirectory(prefix="wsp-results-") as directory:
        checkout = Path(directory)
        git("init", "--quiet", cwd=checkout)
        if git("ls-remote", "--heads", remote, ref):
            git("fetch", "--quiet", "--depth=1", remote, ref, cwd=checkout)
        else:
            git("fetch", "--quiet", "--depth=1", remote, summary["source_commit"], cwd=checkout)
        git("switch", "--quiet", "-c", branch, "FETCH_HEAD", cwd=checkout)

        output = checkout / "results" / summary["run"] / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        git("add", "--", str(output.relative_to(checkout)), cwd=checkout)
        if git("diff", "--cached", "--name-only", cwd=checkout):
            git("-c", f"user.name={os.environ.get('WSP_GIT_NAME', 'WSP Colab')}",
                "-c", f"user.email={os.environ.get('WSP_GIT_EMAIL', 'wsp-colab@users.noreply.github.com')}",
                "commit", "--quiet", "-m", f"Record WSP results for {summary['run']}", cwd=checkout)
        git("push", remote, f"HEAD:{ref}", cwd=checkout)
    print(f"Published {branch} / results/{summary['run']}/summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.run.resolve().name):
        parser.error("Run name must contain letters, digits, hyphens, or underscores.")
    result = summarize(args.run.resolve())
    if args.push:
        publish(result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
