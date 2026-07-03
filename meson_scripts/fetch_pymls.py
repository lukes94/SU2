#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request

PYBEAM_REPO = "https://github.com/pyBeam/pyBeam"


def fetch_pymls(commit: str, dest: Path):
    dest = Path(dest).expanduser().resolve()
    if dest.exists():
        return

    url = f"{PYBEAM_REPO}/archive/{commit}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "pyBeam.tar.gz"
        print(f"Downloading pyMLS from {url} ...")
        urllib.request.urlretrieve(url, archive)

        extracted = tmp / "extracted"
        with tarfile.open(archive) as tar:
            tar.extractall(extracted)

        # GitHub commit tarballs unpack into a single '<repo>-<commit>' directory.
        (root,) = list(extracted.iterdir())
        pymls_src = root / "pyMLS"
        if not pymls_src.is_dir():
            raise FileNotFoundError(f"pyMLS not found in {url}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(pymls_src, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("commit", help="pyBeam commit SHA to fetch pyMLS from")
    p.add_argument("dest", help="destination directory for the pyMLS sources")
    P = p.parse_args()

    try:
        fetch_pymls(P.commit, P.dest)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
