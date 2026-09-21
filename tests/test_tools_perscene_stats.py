"""Scene identity, direction and paired statistics are part of the paper protocol."""
import json
import numpy as np
import pytest
from scipy.stats import wilcoxon
from tools.perscene_stats import holm, load_run, summarize


def write_run(path, scenes, **scores):
    path.mkdir(parents=True)
    (path / 'scenes_all.json').write_text(json.dumps(scenes))
    for metric, values in scores.items():
        (path / f'scores_{metric}_all.json').write_text(json.dumps(values))
    return str(path)


def test_known_order_direction_and_holm(tmp_path):
    first = write_run(tmp_path / 'a', ['x', 'y', 'z', 'w'], psnr=[1, 2, 3, 4], lpips=[.5, .5, .5, .5])
    second = write_run(tmp_path / 'b', ['x', 'y', 'z', 'w'], psnr=[2, 4, 6, 8], lpips=[.2, .4, .5, .6])
    rows = summarize(first, second, ['psnr', 'lpips'])
    assert rows[0]['win_fraction'] == 1
    assert rows[1]['win_fraction'] == .5
    assert rows[1]['ties'] == 1
    assert rows[0]['mean_difference'] == 2.5
    assert rows[0]['p_wilcoxon'] == wilcoxon([1, 2, 3, 4]).pvalue
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    np.testing.assert_allclose([r['p_holm'] for r in rows], holm([r['p_wilcoxon'] for r in rows]))


def test_reordered_scenes_and_duplicate_shards_rejected(tmp_path):
    first = write_run(tmp_path / 'a', ['x', 'y'], psnr=[1, 2])
    second = write_run(tmp_path / 'b', ['y', 'x'], psnr=[1, 2])
    with pytest.raises(ValueError, match='order/content'):
        summarize(first, second, ['psnr'])
    with pytest.raises(ValueError, match='unique'):
        load_run(first + ',' + first, ['psnr'])


def test_all_ties_and_invalid_arrays(tmp_path):
    first = write_run(tmp_path / 'a', ['x', 'y'], psnr=[1, 2])
    assert summarize(first, first, ['psnr'])[0]['p_wilcoxon'] == 1
    second = write_run(tmp_path / 'b', ['x', 'y'], psnr=[1])
    with pytest.raises(ValueError, match='one finite value'):
        load_run(second, ['psnr'])
