#!/usr/bin/env python3
"""Compose and strictly type-check every paper experiment in train and test modes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dacite import Config, from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import RootCfg, TYPE_HOOKS, load_typed_root_config
from src.loss import LossCfgWrapper

REPO = Path(__file__).resolve().parents[1]


@dataclass
class LaunchCfg(RootCfg):
    # src.main consumes this launch option before converting to RootCfg.
    output_dir: Path | None = None


def strict_typed_config(cfg):
    """Reject unknown keys, including inside loss wrappers and nested unions."""
    @dataclass
    class WrappedLoss:
        value: LossCfgWrapper

    strict = Config(strict=True, type_hooks=TYPE_HOOKS)

    def losses(joined):
        return [from_dict(WrappedLoss, {'value': {k: v}}, strict).value
                for k, v in joined.items()]

    return from_dict(LaunchCfg, OmegaConf.to_container(cfg, resolve=True),
                     Config(strict=True, type_hooks={**TYPE_HOOKS, list[LossCfgWrapper]: losses}))


def compose_leaf(name, mode):
    overrides = [f'+experiment={name}', f'mode={mode}']
    if mode == 'test':
        overrides += ['dataset/view_sampler=evaluation']
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'config')):
        cfg = compose(config_name='main', overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def check_leaf(name, mode):
    cfg = compose_leaf(name, mode)
    typed = load_typed_root_config(cfg)
    strict = strict_typed_config(cfg)
    # Apart from the launch-only output directory, both conversions must agree.
    assert asdict(typed) == {k: v for k, v in asdict(strict).items() if k != 'output_dir'}
    return cfg


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    failures = 0
    for path in sorted((REPO / 'config/experiment').glob('*.yaml')):
        if path.stem.startswith('_'):
            continue
        for mode in ('train', 'test'):
            try:
                cfg = check_leaf(path.stem, mode)
                enc, dec = cfg.model.encoder, cfg.model.decoder
                adapter = enc.gaussian_adapter
                flags = '/'.join(str(int(adapter[k])) for k in
                                 ('camera_space_frame', 'tangent_frame', 'use_rotation_residual'))
                losses = ','.join(f'{k}={v.weight}' for k, v in cfg.loss.items())
                print(f'PASS {path.stem} {mode}: flags={flags} {enc.monodepth_vit_type} '
                      f'steps={cfg.trainer.max_steps} batch={cfg.data_loader.train.batch_size}×'
                      f'{cfg.trainer.accumulate_grad_batches} loss={losses} opacity_cap=0.6 '
                      f'alpha={dec.alpha_threshold_train}/{dec.alpha_threshold_eval} '
                      f'image={list(cfg.dataset.image_shape)} near/far={cfg.dataset.near}/{cfg.dataset.far}')
            except Exception as exc:
                failures += 1
                print(f'FAIL {path.stem} {mode}: {type(exc).__name__}: {exc}')
    print(f'{failures} configuration failures')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
