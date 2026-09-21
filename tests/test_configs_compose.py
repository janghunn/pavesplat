"""Literal paper recipes and strict schema checks for every experiment and mode."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import pytest
from dacite.exceptions import DaciteError
from omegaconf import OmegaConf
from scripts.check_configs import check_leaf, compose_leaf, strict_typed_config

# Dataset, method flags, backbone, scales, upsampling, near/far, roots, image size.
EXPECTED = {
    'pavesplat_vits_re10k': ('re10k', (True, True, True), 'vits', 1, 4, .5, 100., 'datasets/re10k', [256, 256]),
    'pavesplat_vitb_re10k': ('re10k', (True, True, True), 'vitb', 2, 2, .5, 100., 'datasets/re10k', [256, 256]),
    'pavesplat_vitl_re10k': ('re10k', (True, True, True), 'vitl', 2, 2, .5, 100., 'datasets/re10k', [256, 256]),
    'surfsplat_vits_re10k': ('re10k', (False, False, False), 'vits', 1, 4, .5, 100., 'datasets/re10k', [256, 256]),
    'surfsplat_vitl_re10k': ('re10k', (False, False, False), 'vitl', 2, 2, .5, 100., 'datasets/re10k', [256, 256]),
    'ablation_tangent_frame_vitl_re10k': ('re10k', (True, True, False), 'vitl', 2, 2, .5, 100., 'datasets/re10k', [256, 256]),
    'zeroshot_dl3dv_pavesplat_vitl': ('dl3dv', (True, True, True), 'vitl', 2, 2, .5, 200., 'datasets/dl3dv', [256, 256]),
    'zeroshot_dl3dv_surfsplat_vitl': ('dl3dv', (False, False, False), 'vitl', 2, 2, .5, 200., 'datasets/dl3dv', [256, 256]),
    'zeroshot_acid_pavesplat_vitl': ('re10k', (True, True, True), 'vitl', 2, 2, .5, 100., 'datasets/acid', [256, 256]),
    'zeroshot_acid_surfsplat_vitl': ('re10k', (False, False, False), 'vitl', 2, 2, .5, 100., 'datasets/acid', [256, 256]),
    'zeroshot_scannet_pavesplat_vitl': ('scannet', (True, True, True), 'vitl', 2, 2, .5, 100., 'datasets/scannet', [256, 256]),
    'zeroshot_scannet_surfsplat_vitl': ('scannet', (False, False, False), 'vitl', 2, 2, .5, 100., 'datasets/scannet', [256, 256]),
}


def test_every_leaf_has_literal_expectations():
    root = Path(__file__).resolve().parents[1] / 'config/experiment'
    assert {p.stem for p in root.glob('*.yaml')} == set(EXPECTED)


@pytest.mark.parametrize('name', EXPECTED)
@pytest.mark.parametrize('mode', ['train', 'test'])
def test_paper_recipe(name, mode):
    cfg = check_leaf(name, mode)
    assert '${' not in OmegaConf.to_yaml(cfg, resolve=True)
    dataset, flags, vit, scales, upsample, near, far, roots, shape = EXPECTED[name]
    enc, dec = cfg.model.encoder, cfg.model.decoder
    adapter = enc.gaussian_adapter
    assert tuple(adapter[k] for k in ('camera_space_frame', 'tangent_frame', 'use_rotation_residual')) == flags
    assert (enc.monodepth_vit_type, enc.num_scales, enc.upsample_factor, enc.lowest_feature_resolution) == (vit, scales, upsample, 4)
    assert (cfg.dataset.name, cfg.dataset.near, cfg.dataset.far) == (dataset, near, far)
    assert list(cfg.dataset.roots) == [roots]
    assert list(cfg.dataset.image_shape) == shape
    assert cfg.dataset.baseline_scale_bounds is False
    assert cfg.dataset.make_baseline_1 is False
    assert (cfg.trainer.max_steps, cfg.data_loader.train.batch_size, cfg.trainer.accumulate_grad_batches) == (150001, 4, 4)
    assert cfg.optimizer.lr == 2e-4
    assert cfg.optimizer.lr_monodepth == 2e-6
    assert cfg.loss.mse.weight == 1.0
    assert cfg.loss.lpips.weight == .05
    assert cfg.loss.lpips.apply_after_step == 0
    assert (dec.alpha_threshold_train, dec.alpha_threshold_eval) == (.1, .001)
    assert adapter.gaussian_scale_max == 3.0
    assert adapter.gaussian_scale_min == 1e-10
    assert cfg.train.hrrc_gt_mode == 'bicubic'
    assert list(cfg.test.black_eps_sweep) == [.01, .02, .05]
    assert cfg.test.save_normal_npz is False
    if dataset == 'dl3dv':
        assert list(cfg.dataset.ori_image_shape) == [270, 480]
        assert cfg.dataset.skip_bad_shape is True


@pytest.mark.parametrize('key', ['test.unknown_option', 'model.encoder.gaussian_adapter.unknown_option', 'loss.lpips.unknown_option'])
def test_strict_conversion_rejects_unknown_nested_keys(key):
    cfg = compose_leaf('pavesplat_vitl_re10k', 'test')
    OmegaConf.update(cfg, key, 1, force_add=True)
    with pytest.raises(DaciteError):
        strict_typed_config(cfg)


EVALUATION_LAUNCHERS = [
    ('eval_re10k.sh', ['pavesplat_vitl_re10k', 'checkpoints/model.ckpt', '--8x', '--coverage', '--timing']),
    ('eval_zeroshot.sh', ['dl3dv', 'pavesplat', 'checkpoints/model.ckpt', '--8x', '--coverage']),
    ('eval_scannet_normals.sh', ['pavesplat', 'checkpoints/model.ckpt', '--npz', '--images']),
]


@pytest.mark.parametrize('script,arguments', EVALUATION_LAUNCHERS)
@pytest.mark.parametrize('visible,selected,expected,note', [
    (None, None, '0', False),
    ('4', None, '4', False),
    ('4,7', None, '4', True),
    ('4,7', '7', '7', False),
    ('GPU-abc,GPU-def', None, 'GPU-abc', True),
])
def test_evaluation_launches_on_one_gpu(tmp_path, script, arguments, visible, selected, expected, note):
    root = Path(__file__).resolve().parents[1]
    # Capture the environment at exec without constructing a model or touching outputs.
    python = tmp_path / 'python'
    python.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'print(json.dumps({"gpu": os.environ["CUDA_VISIBLE_DEVICES"], '
        '"timing": os.environ.get("PAVESPLAT_SYNC_TIMING"), "args": sys.argv[1:]}))\n'
    )
    python.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ.get('PATH', '')}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('PAVESPLAT_SYNC_TIMING', None)
    if visible is not None:
        env['CUDA_VISIBLE_DEVICES'] = visible
    command = ['bash', str(root / 'scripts' / script), *arguments]
    if selected is not None:
        command += ['--gpu', selected]
    dry = subprocess.run([*command, '--dry-run'], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True)
    resolved = shlex.split(dry.stdout)
    assert resolved[0] == f'CUDA_VISIBLE_DEVICES={expected}'
    assert len(dry.stdout.splitlines()) == 1
    executed = subprocess.run(command, cwd=tmp_path, env=env,
                              capture_output=True, text=True, check=True)
    captured = json.loads(executed.stdout)
    assert captured['gpu'] == expected
    assert captured['args'] == resolved[resolved.index('python') + 1:]
    assert captured['args'][:2] == ['-m', 'src.main']
    assert 'mode=test' in captured['args']
    assert '--gpu' not in captured['args']
    if script == 'eval_re10k.sh':
        assert captured['timing'] == '1'
        assert 'PAVESPLAT_SYNC_TIMING=1' in resolved
    for result in (dry, executed):
        expected_note = (
            f'Evaluation runs on a single GPU; using first listed id {expected} '
            'from CUDA_VISIBLE_DEVICES.\n'
        )
        assert result.stderr == (expected_note if note else '')


@pytest.mark.parametrize('script,arguments', EVALUATION_LAUNCHERS)
@pytest.mark.parametrize('gpu_arguments', [['--gpu'], ['--gpu', '0,1'], ['--gpu', '']])
def test_evaluation_rejects_missing_or_multiple_gpu_ids(script, arguments, gpu_arguments):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', str(root / 'scripts' / script), *arguments,
                             *gpu_arguments, '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize('gpu', ['0,1', '', ' ', '0 1', '\t', '-1'])
def test_timing_rejects_invalid_gpu_id(gpu):
    script = Path(__file__).resolve().parents[1] / 'scripts/bench_encoder_timing.sh'
    result = subprocess.run(['bash', str(script), 'pavesplat.ckpt', 'surfsplat.ckpt', gpu,
                             '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stderr == 'gpu must select one GPU id.\n'
    assert not result.stdout


@pytest.mark.parametrize('gpu,expected', [(None, '0'), ('3', '3'), ('GPU-abc', 'GPU-abc')])
def test_timing_selects_one_gpu_for_all_pairs(gpu, expected):
    script = Path(__file__).resolve().parents[1] / 'scripts/bench_encoder_timing.sh'
    command = ['bash', str(script), 'pavesplat.ckpt', 'surfsplat.ckpt']
    if gpu is not None:
        command.append(gpu)
    result = subprocess.run([*command, '--dry-run'], check=True, capture_output=True, text=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    for index, command in enumerate(commands):
        assert command[:3] == ['env', f'CUDA_VISIBLE_DEVICES={expected}', 'PAVESPLAT_SYNC_TIMING=1']
        method = 'pavesplat' if index % 2 == 0 else 'surfsplat'
        assert f'+experiment={method}_vitl_re10k' in command
        assert f'output_dir=outputs/encoder_timing/{method}_{index // 2 + 1}' in command
