"""Check DL3DV chunk preparation and the loader's scene-to-chunk index on CPU."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts"


@pytest.mark.parametrize("stages", [None, ["train", "test"]])
def test_chunk_index_cli(tmp_path, stages):
    root = tmp_path / "chunks"
    stage_paths = [root / stage for stage in stages] if stages else [root]
    for stage_path in stage_paths:
        stage_path.mkdir(parents=True)
        torch.save([{"key": "scene-c", "images": [torch.zeros(1, dtype=torch.uint8)]}],
                   stage_path / "000001.torch")
        torch.save([{"key": "scene-a"}, {"key": "scene-b"}], stage_path / "000000.torch")
        (stage_path / "index.json").write_text('{"stale": "missing.torch"}')
        (stage_path / "ignored.txt").write_text("not a chunk")

    command = [sys.executable, str(SCRIPTS / "generate_chunk_index.py"), "--root", str(root)]
    if stages:
        command.extend(["--stage", *stages])
    subprocess.run(command, check=True, capture_output=True, text=True)

    expected = {"scene-a": "000000.torch", "scene-b": "000000.torch", "scene-c": "000001.torch"}
    for stage_path in stage_paths:
        index = json.loads((stage_path / "index.json").read_text())
        assert index == expected
        assert list(index) == list(expected)
        for key, chunk_file in index.items():
            chunk = torch.load(stage_path / chunk_file, map_location="cpu", weights_only=True)
            assert key in [example["key"] for example in chunk]


@pytest.mark.parametrize("img_subdir,shape", [("images_8", (270, 480)), ("images_4", (540, 960))])
def test_dl3dv_converter_image_directory(tmp_path, img_subdir, shape):
    raw_root = tmp_path / "raw"
    scene = raw_root / "scene-a" / "nerfstudio"
    image_dir = scene / img_subdir
    image_dir.mkdir(parents=True)
    image_path = image_dir / "frame_00001.jpg"
    height, width = shape
    Image.new("RGB", (width, height), (20, 40, 60)).save(image_path)
    (scene / "transforms.json").write_text(json.dumps({
        "h": height, "w": width, "fl_x": width, "fl_y": height,
        "cx": width / 2, "cy": height / 2,
        "frames": [{"file_path": "frame_00001.jpg", "transform_matrix": np.eye(4).tolist()}],
    }))
    output_root = tmp_path / "converted"
    command = [sys.executable, str(SCRIPTS / "convert_dl3dv_test.py"),
               "--input_dir", str(raw_root), "--output_dir", str(output_root)]
    if img_subdir != "images_8":
        command.extend(["--img_subdir", img_subdir])
    subprocess.run(command, check=True, capture_output=True, text=True)

    chunk_paths = sorted((output_root / "test").glob("*.torch"))
    assert [path.name for path in chunk_paths] == ["000000.torch"]
    chunk = torch.load(chunk_paths[0], map_location="cpu", weights_only=True)
    assert len(chunk) == 1
    assert chunk[0]["key"] == "scene-a"
    assert chunk[0]["timestamps"].tolist() == [1]
    assert chunk[0]["images"][0].numpy().tobytes() == image_path.read_bytes()
    torch.testing.assert_close(chunk[0]["cameras"][0, :6], torch.tensor([1., 1., .5, .5, 0., 0.]))
