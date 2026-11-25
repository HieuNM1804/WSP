"""Archive a completed WSP run and publish it to its own Git branch."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
OUTCOME_FIELDS = (
    "dataset", "unseen_classes", "num_sketches", "num_photos", "mAP@all", "mAP@200", "P@100", "P@200",
)
RUN_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
EVALUATION = re.compile(r"final|pass_[0-9]+")
TURN_ID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
TOKEN = re.compile(
    r"\b(?:gh[pousr]_[a-zA-Z0-9_]{20,}|github_pat_[a-zA-Z0-9_]{20,}|sk-[a-zA-Z0-9_-]{20,}|"
    r"eyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,})\b"
)
SECRET_FIELD = re.compile(
    r"""(?i)((?:api[_-]?key|(?:access|refresh|id)[_-]?token|kaggle_key|password)["']?\s*[=:]\s*["']?"""
    r"""|Bearer\s+)[a-zA-Z0-9._~+/=-]{8,}"""
)
RUN_FILES = (
    "config.json", "state.json", "message.txt", "instructions.md", "loop.log", "driver.log",
    "tool_calls.jsonl", "explanations.md", "final_answer.md", "work/notes.md",
)
READABLE_FILES = tuple(f"readable/{name}.md" for name in (
    "README", "prompt", "trajectory", "passes", "explanations", "notes", "final_answer",
))
MEMBER = re.compile(
    r"(?:summary\.json|model/arch\.json|message_budget[0-9]+\.txt|final_answer_pass_[0-9]+\.md|"
    r"turns/t[0-9]+/(?:prompt\.txt|answer\.md|events\.jsonl|run\.json)|"
    r"requests/fwd_[0-9]+\.(?:json\.claimed|done\.json)|"
    r"eval/(?:unedited|final|pass_[0-9]+)\.json|"
    r"checkpoints/pass_[0-9]+/(?:info\.json|notes\.md|final_answer_pass_[0-9]+\.md)|"
    r"(?:checkpoints/pass_[0-9]+/)?edit/(?:manifest\.json|tensors/[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*\.npy))"
)


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd or ROOT, text=True, encoding="utf-8").strip()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_revision():
    try:
        return git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(run):
    config = load(run / "config.json")
    state = load(run / "state.json")
    evaluations = {}
    for path in sorted((ROOT / "wsp_sbir" / "eval" / run.name).glob("*.json")):
        if not EVALUATION.fullmatch(path.stem):
            continue
        items = load(path)
        metrics = {
            field: items[field] for field in OUTCOME_FIELDS
            if field in items and (field == "dataset" or type(items[field]) in (int, float))
        }
        if any(field.startswith("mAP@") for field in metrics):
            evaluations[path.stem] = metrics
    if not evaluations and not state["done"]:
        raise SystemExit("No completed checkpoint or final evaluation to publish.")

    return {
        "run": run.name,
        "source_commit": config.get("source_commit") or source_revision(),
        "dataset": state["task"],
        "created": config.get("created"),
        "model": config["model"],
        "effort": config["effort"],
        "budget": config["budget"],
        "completed_passes": state["passes"],
        "completed_turns": state["turn"],
        "done": state["done"],
        "evaluations": evaluations,
    }


def archive_member(name):
    return name in RUN_FILES or name in READABLE_FILES or MEMBER.fullmatch(name)


def edit_files(edit):
    manifest = edit / "manifest.json"
    if manifest.resolve() != manifest:
        raise ValueError(f"Linked edit manifest: {manifest}")
    yield manifest
    for tensor in load(manifest)["tensors"]:
        name = tensor["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*", name):
            raise ValueError(f"Invalid tensor name in {manifest}")
        path = edit / "tensors" / f"{name}.npy"
        if path.resolve() != path:
            raise ValueError(f"Linked tensor: {path}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != tensor["submitted_sha256"]:
                raise ValueError(f"Tensor does not match its manifest: {path}")
        yield path


def archive_files(run, summary):
    for name in (*RUN_FILES, *READABLE_FILES):
        path = run / name
        if path.is_file() and path.resolve() == path:
            yield name, path
    for pattern in ("message_budget*.txt", "final_answer_pass_*.md", "turns/t*/*",
                    "requests/fwd_*", "checkpoints/pass_*/*"):
        for path in sorted(run.glob(pattern)):
            name = path.relative_to(run).as_posix()
            if archive_member(name) and path.is_file() and path.resolve() == path:
                yield name, path
    checkpoint_edits = sorted(path for path in run.glob("checkpoints/pass_*/edit")
                              if re.fullmatch(r"pass_[0-9]+", path.parent.name))
    for edit in (run / "edit", *checkpoint_edits):
        for path in edit_files(edit):
            yield path.relative_to(run).as_posix(), path
    eval_root = ROOT / "wsp_sbir" / "eval"
    for path in sorted((eval_root / run.name).glob("*.json")):
        if EVALUATION.fullmatch(path.stem) and path.resolve() == path:
            yield f"eval/{path.name}", path
    baseline = eval_root / f"unedited_{summary['dataset']}" / "result.json"
    arch = ROOT / "wsp_sbir" / "model" / "arch.json"
    for name, path in (("eval/unedited.json", baseline), ("model/arch.json", arch)):
        if path.is_file() and path.resolve() == path:
            yield name, path


def redact(text, substitutions):
    for path, label in substitutions:
        text = text.replace(json.dumps(path, ensure_ascii=False)[1:-1], label).replace(path, label)
    text = TURN_ID.sub(lambda match: "id-" + hashlib.sha256(match[0].lower().encode()).hexdigest()[:8], text)
    text = TOKEN.sub("[redacted]", text)
    return SECRET_FIELD.sub(r"\1[redacted]", text)


def archive_run(run, output):
    summary = summarize(run)
    if not summary["done"]:
        raise ValueError("Only completed runs can be archived.")
    config = load(run / "config.json")
    task_path = Path(config["task"]) / "task.json"
    task = load(task_path) if task_path.is_file() else {}
    substitutions = [(str(run), "<run>"), (str(ROOT), "<repo>"), (str(Path.home()), "<home>")]
    for variable, label in (("CODEX_HOME", "<codex_home>"), ("WSP_CODEX_BIN", "<codex>"),
                            ("WSP_SANDBOX_VENV", "<venv>"), ("WSP_GPU_PYTHON", "<python>")):
        if value := os.environ.get(variable):
            if os.path.isabs(value):
                substitutions.append((value, label))
    for key, label in (("root", "<data>"), ("eval_root", "<eval_data>"), ("weights", "<weights>")):
        if value := task.get(key):
            substitutions.append((value, label))
    substitutions.sort(key=lambda item: len(item[0]), reverse=True)

    files = dict(archive_files(run, summary))
    required = {"message.txt", "instructions.md", "final_answer.md", "edit/manifest.json"}
    for turn in range(1, summary["completed_turns"] + 1):
        required.update(f"turns/t{turn:03d}/{name}" for name in ("prompt.txt", "events.jsonl", "run.json"))
    if missing := required - files.keys():
        raise ValueError(f"Incomplete run archive. Missing: {', '.join(sorted(missing))}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
            for name, path in files.items():
                if path.suffix == ".npy":
                    archive.write(path, name)
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if name.startswith("turns/") and path.name == "run.json":
                    record = json.loads(text)
                    record.pop("stderr_tail", None)
                    text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
                archive.writestr(name, redact(text, substitutions))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def publish(bundle):
    with ZipFile(bundle) as archive:
        files = archive.namelist()
        if len(files) != len(set(files)) or not all(archive_member(name) for name in files):
            raise ValueError("Unexpected or duplicate filenames in result bundle.")
        summary = json.loads(archive.read("summary.json"))
        if not RUN_NAME.fullmatch(summary["run"]) or not summary["done"]:
            raise ValueError("A completed bundle and a valid run name are required.")
        publish_files(summary, archive, files)


def publish_files(summary, archive, files):
    branch = f"results/{summary['run']}"
    ref = f"refs/heads/{branch}"
    remote = git("remote", "get-url", "--push", "origin")
    with TemporaryDirectory(prefix="wsp-results-") as directory:
        checkout = Path(directory)
        git("init", "--quiet", cwd=checkout)
        if git("ls-remote", "--heads", remote, ref):
            git("fetch", "--quiet", "--depth=1", remote, ref, cwd=checkout)
        else:
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", summary["source_commit"] or ""):
                raise ValueError("A source commit is required to create a result branch.")
            git("fetch", "--quiet", "--depth=1", remote, summary["source_commit"], cwd=checkout)
        git("switch", "--quiet", "-c", branch, "FETCH_HEAD", cwd=checkout)

        result = Path("results") / summary["run"]
        if (checkout / result).resolve() != checkout / result:
            raise ValueError("Result directory contains a symlink.")
        previous = checkout / result / "summary.json"
        if previous.is_symlink():
            raise ValueError("Previous result summary is a symlink.")
        if previous.is_file():
            created = load(previous).get("created")
            if created and created != summary.get("created"):
                raise ValueError(f"{branch} already belongs to another run.")
        tracked = git("ls-files", "-z", "--", result.as_posix(), cwd=checkout)
        for old in tracked.split("\0")[:-1]:
            name = Path(old).relative_to(result).as_posix()
            if archive_member(name) and name not in files:
                git("rm", "--", old, cwd=checkout)
        for name in files:
            output = checkout / result / name
            if output.resolve() != output:
                raise ValueError(f"Result file contains a symlink: {name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as stream, output.open("wb") as target:
                shutil.copyfileobj(stream, target)
        git("add", "--", result.as_posix(), cwd=checkout)
        if git("diff", "--cached", "--name-only", cwd=checkout):
            git("-c", f"user.name={os.environ.get('WSP_GIT_NAME', 'WSP Colab')}",
                "-c", f"user.email={os.environ.get('WSP_GIT_EMAIL', 'wsp-colab@users.noreply.github.com')}",
                "commit", "--quiet", "-m", f"Record WSP results for {summary['run']}", cwd=checkout)
        git("push", remote, f"HEAD:{ref}", cwd=checkout)
    print(f"Published {branch} / results/{summary['run']}/ ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="run directory on this machine")
    source.add_argument("--from-bundle", type=Path, help="result.zip exported on another machine")
    parser.add_argument("--bundle", type=Path, help="write a completed run to this ZIP file")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.from_bundle:
        if args.bundle:
            parser.error("--bundle requires --run.")
        if args.push:
            publish(args.from_bundle)
        else:
            with ZipFile(args.from_bundle) as archive:
                print(archive.read("summary.json").decode("utf-8"), end="")
    else:
        run = args.run.resolve()
        if not RUN_NAME.fullmatch(run.name):
            parser.error("Run name must contain letters, digits, hyphens, or underscores.")
        if args.push or args.bundle:
            bundle = args.bundle or run / "result.zip"
            summary = archive_run(run, bundle)
            if args.push:
                publish(bundle)
                return
        else:
            summary = summarize(run)
        print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
