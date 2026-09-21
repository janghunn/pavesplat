"""Compose each released experiment and pin its surface-adapter method flags."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg, SurfelAdapter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
EXPERIMENTS = sorted(
    p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")
    if not p.stem.startswith("_")
)
METHOD_FLAGS = {
    "pavesplat": (True, True, True),
    "surfsplat": (False, False, False),
    "tangent_frame_only": (True, True, False),
}


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_yaml_constructs_adapter(experiment):
    leaf = OmegaConf.load(CONFIG_DIR / "experiment" / f"{experiment}.yaml")
    methods = [
        entry.split("/", 1)[1] for entry in leaf.defaults
        if isinstance(entry, str) and entry.startswith("_method/")
    ]
    assert len(methods) == 1, f"{experiment} must select exactly one method"
    expected = METHOD_FLAGS[methods[0]]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="main", overrides=[f"+experiment={experiment}"])
    adapter_cfg = GaussianAdapterCfg(**OmegaConf.to_container(
        cfg.model.encoder.gaussian_adapter, resolve=True
    ))
    adapter = SurfelAdapter(adapter_cfg)
    assert (
        adapter.cfg.camera_space_frame,
        adapter.cfg.tangent_frame,
        adapter.cfg.use_rotation_residual,
    ) == expected
    assert adapter.d_in + 3 == (36 if expected[2] else 32)
    assert list(adapter.named_parameters()) == []
