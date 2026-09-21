"""Generate the scene-to-chunk index used for main Table 2 DL3DV evaluation."""

import argparse
import json
from pathlib import Path

import torch


def generate_index(stage_path: Path) -> None:
    """Write index.json with chunk paths relative to the stage directory."""
    if not stage_path.is_dir():
        raise NotADirectoryError(stage_path)
    index = {}
    for chunk_path in sorted(stage_path.glob("*.torch")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        for example in chunk:
            index[example["key"]] = str(chunk_path.relative_to(stage_path))
    with (stage_path / "index.json").open("w") as f:
        json.dump(index, f)
    print(f"Wrote {len(index)} scene keys to {stage_path / 'index.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="stage directory holding *.torch, or dataset root when --stage is given",
    )
    parser.add_argument(
        "--stage", nargs="+", metavar="NAME",
        help="optional stage subdirectories under --root (for example: train test)",
    )
    args = parser.parse_args()
    stage_paths = [args.root / stage for stage in args.stage] if args.stage else [args.root]
    for stage_path in stage_paths:
        generate_index(stage_path)


if __name__ == "__main__":
    main()
