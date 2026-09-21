"""CPU key/shape equality against the two public Lightning checkpoints."""
import os
from pathlib import Path
import pytest
import torch

from scripts.check_configs import check_leaf
from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.model.model_wrapper import ModelWrapper


@pytest.mark.parametrize('experiment,filename,head_channels', [
    ('pavesplat_vitl_re10k', 're10k_core_l_step150000.ckpt', 36),
    ('surfsplat_vitl_re10k', 're10k_baseline_l_step150000.ckpt', 32),
])
def test_public_checkpoint_keys_and_shapes(monkeypatch, experiment, filename, head_channels):
    directory = os.environ.get('PAVESPLAT_CKPT_DIR')
    if not directory:
        pytest.skip('Set PAVESPLAT_CKPT_DIR to verify public checkpoint key/shape identity')
    path = Path(directory) / filename
    assert path.is_file(), f'Missing checkpoint: {path}'
    # Build the actual DINOv2 architecture without downloading/loading its weights.
    hub_load = torch.hub.load
    cached = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    def architecture_only(repository, model, *args, **kwargs):
        kwargs['pretrained'] = False
        if cached.is_dir():
            return hub_load(str(cached), model, *args, source='local', **kwargs)
        return hub_load(repository, model, *args, **kwargs)
    monkeypatch.setattr(torch.hub, 'load', architecture_only)
    cfg_dict = check_leaf(experiment, 'test')
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    encoder, visualizer = get_encoder(cfg.model.encoder)
    # Loss LPIPS buffers are nonpersistent; no loss tensors occur in these checkpoints.
    model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, visualizer,
                         get_decoder(cfg.model.decoder, cfg.dataset), [], None).cpu()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    actual, expected = model.state_dict(), checkpoint['state_dict']
    assert actual.keys() == expected.keys(), {
        'missing': sorted(expected.keys() - actual.keys()),
        'unexpected': sorted(actual.keys() - expected.keys()),
    }
    mismatches = {key: (tuple(actual[key].shape), tuple(expected[key].shape))
                  for key in actual if actual[key].shape != expected[key].shape}
    assert not mismatches, mismatches
    assert encoder.gaussian_head[-1].out_channels == head_channels
    print(f'{experiment}: {len(actual)} tensors, head={head_channels}, exact key/shape match')
