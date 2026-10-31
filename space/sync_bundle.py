"""Copy the canonical project into a Hugging Face Space bundle.

The Space app imports ``src``. This script copies ``src/``, ``config/``,
and the bundled synthetic sample next to ``app.py`` so the Hub repo root
can import them. It does not train a model and it does not upload the bundle.

Run from a checkout, before the Hub upload::

    python space/sync_bundle.py --out /tmp/cloudie-space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class SpaceBundle:
    """Build the directory that is pushed to the Space repository."""

    SPACE_FILES = ("app.py", "requirements.txt", "README.md", "sync_bundle.py")
    PROJECT_TREES = ("src", "config")
    SAMPLE = ("cloudie", "samples", "synthetic_ir.npy")
    SKIP_NAMES = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "artifacts",
        "data",
        "reports",
        ".git",
    }

    @staticmethod
    def repo_root() -> Path:
        """GitHub checkout that contains ``src/inference.py`` beside ``space/``."""
        here = Path(__file__).resolve().parent
        candidate = here.parent
        if (candidate / "src" / "inference.py").is_file() and (here / "app.py").is_file():
            return candidate
        if (here / "src" / "inference.py").is_file():
            return here
        raise FileNotFoundError(
            "src/inference.py was not found. Run this script from the Cloudie checkout."
        )

    @staticmethod
    def ignore(directory: str, names: list[str]) -> set[str]:
        """Skip caches and local data. Keep ``cloudie/samples``."""
        del directory
        skipped = set()
        for name in names:
            if name in SpaceBundle.SKIP_NAMES or name.endswith(".pyc"):
                skipped.add(name)
        return skipped

    @staticmethod
    def build(destination: Path, repo_root: Path | None = None) -> Path:
        """Write a fresh bundle. Refuses to replace the checkout or its source trees."""
        root = (repo_root or SpaceBundle.repo_root()).resolve()
        destination = destination.resolve()
        space_dir = root / "space"
        if destination == root or destination in root.parents or destination == space_dir:
            raise ValueError("Write the bundle to a new directory, for example space/bundle.")
        for name in SpaceBundle.PROJECT_TREES:
            tree = root / name
            if destination == tree or tree in destination.parents or destination in tree.parents:
                raise ValueError("The bundle destination must sit outside src, config, and cloudie.")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for name in SpaceBundle.SPACE_FILES:
            shutil.copy2(space_dir / name, destination / name)
        for name in SpaceBundle.PROJECT_TREES:
            shutil.copytree(root / name, destination / name, ignore=SpaceBundle.ignore)
        sample = root.joinpath(*SpaceBundle.SAMPLE)
        dest_sample = destination.joinpath(*SpaceBundle.SAMPLE)
        dest_sample.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample, dest_sample)
        return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("space/bundle"),
        help="Directory to write. Replaced when it already exists.",
    )
    args = parser.parse_args(argv)
    built = SpaceBundle.build(args.out)
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
