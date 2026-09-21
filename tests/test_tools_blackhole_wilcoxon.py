"""Tiny sharded evaluations exercise adjacent pairing and family correction."""
import json
import numpy as np
import pytest
from scipy.stats import wilcoxon
from tools.blackhole_perscene_wilcoxon import compare
from tools.perscene_stats import holm


def shard(path, scenes, values):
    path.mkdir()
    (path / 'scenes_all.json').write_text(json.dumps(scenes))
    (path / 'scores_cov_black_s8_all.json').write_text(json.dumps(values))
    return str(path)


def test_shards_concatenate_and_adjacent_pairs(tmp_path):
    a = shard(tmp_path / 'a0', ['x', 'y'], [.2, .3]) + ',' + shard(tmp_path / 'a1', ['z', 'w'], [.4, .5])
    b = shard(tmp_path / 'b', ['x', 'y', 'z', 'w'], [.1, .1, .1, .1])
    c = shard(tmp_path / 'c', ['x', 'y', 'z', 'w'], [.1, .1, .2, .2])
    rows = compare([f'baseline={a}', f'tangent={b}', f'pavesplat={c}'])
    assert [(r['from_run'], r['to_run']) for r in rows] == [('baseline', 'tangent'), ('tangent', 'pavesplat')]
    assert rows[0]['n'] == 4
    assert rows[0]['win_frac'] == 1
    assert rows[0]['mean_diff_pct'] == pytest.approx(-25)
    assert rows[0]['p_wilcoxon'] == wilcoxon(np.array([.1]*4) - [.2, .3, .4, .5]).pvalue
    assert rows[1]['n_ties'] == 2
    assert rows[1]['win_frac'] == 0
    np.testing.assert_allclose([r['p_holm'] for r in rows], holm([r['p_wilcoxon'] for r in rows]))


def test_misaligned_shards_are_rejected(tmp_path):
    a = shard(tmp_path / 'a', ['x', 'y'], [.1, .2])
    b = shard(tmp_path / 'b', ['y', 'x'], [.1, .2])
    with pytest.raises(ValueError, match='order/content'):
        compare([f'a={a}', f'b={b}'])
    rows = compare([f'a={a}', f'b={a}'])
    assert rows[0]['p_wilcoxon'] == 1
    assert rows[0]['p_holm'] == 1
