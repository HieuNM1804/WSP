"""Download the public Sketchy release and verify its layout for WSP."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://www.kaggle.com/api/v1/datasets/download/b20dccn616nguynhutun/sketchy?datasetVersionNumber=1"
EXPECTED_COUNTS = {"sketch": 70_281, "photo": 85_497}
GIB = 2**30


def verify(root):
    counts = {}
    for modality, expected in EXPECTED_COUNTS.items():
        directory = root / modality
        categories = [p for p in directory.iterdir() if p.is_dir()]
        count = sum(p.is_file() for folder in categories for p in folder.iterdir())
        if len(categories) != 125 or count != expected:
            raise ValueError(f"{directory}: expected 125 categories and {expected:,} images, found "
                             f"{len(categories)} categories and {count:,} files")
        counts[modality] = count
    return counts


def download(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with TemporaryDirectory(prefix="sketchy-", dir=destination.parent) as stage:
            stage = Path(stage)
            archive = stage / "sketchy.zip"
            with urlopen(URL, timeout=90) as response, archive.open("wb") as output:
                size = int(response.headers.get("Content-Length", 0))
                free = shutil.disk_usage(stage).free
                if size and free < size + 3 * GIB:
                    raise OSError("Insufficient space for the Sketchy archive and extracted files")
                print("Downloading Sketchy v1 from Kaggle", flush=True)
                written = 0
                report_at = 256 * 2**20
                while chunk := response.read(8 * 2**20):
                    output.write(chunk)
                    written += len(chunk)
                    if written >= report_at:
                        print(f"Downloaded {written / GIB:.2f} GiB", flush=True)
                        report_at += 256 * 2**20
            with ZipFile(archive) as bundle:
                bundle.extractall(stage)
            root = stage / "Sketchy"
            counts = verify(root)
            root.rename(destination)
    else:
        counts = verify(destination)
    print(f"{destination}: {counts['sketch']:,} sketches, {counts['photo']:,} photos", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/content/Sketchy"))
    args = parser.parse_args()
    download(args.destination)
