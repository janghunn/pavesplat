"""Check that impossible ScanNet target requests terminate before sampling."""

from types import SimpleNamespace

import pytest
import torch

from src.scripts import generate_scannet_eval_index as index_tool


def config(targets, min_distance=1, max_distance=192):
    return SimpleNamespace(num_target_views=targets, min_distance=min_distance,
                           max_distance=max_distance, min_overlap=.6, max_overlap=1.)


def overlapping_cameras(monkeypatch, count):
    monkeypatch.setattr(index_tool, 'get_world_rays', lambda *args: (None, None))
    monkeypatch.setattr(index_tool, 'project_rays', lambda *args: {'overlaps_image': torch.ones(1)})
    return torch.zeros(count, 4, 4), torch.zeros(count, 3, 3)


@pytest.mark.parametrize('targets', [0, 4])
def test_target_count_must_fit_slice_before_geometry(monkeypatch, targets):
    def unexpected_geometry(*args):
        pytest.fail('Impossible target count reached geometry sampling')
    monkeypatch.setattr(index_tool, 'sample_image_grid', unexpected_geometry)
    with pytest.raises(ValueError, match=f'Cannot sample {targets} distinct targets from a 3-frame slice'):
        index_tool.find_pair_and_targets(torch.zeros(3, 4, 4), torch.zeros(3, 3, 3),
                                         1, 1, 'cpu', config(targets), torch.Generator())


def test_no_context_interval_can_fit_targets(monkeypatch):
    extr, intr = overlapping_cameras(monkeypatch, 6)
    def unexpected_sampling(*args, **kwargs):
        pytest.fail('Impossible context intervals reached rejection sampling')
    monkeypatch.setattr(torch, 'randint', unexpected_sampling)
    with pytest.raises(ValueError, match='No context pair.*4 distinct targets.*contains 3 frames'):
        index_tool.find_pair_and_targets(extr, intr, 1, 1, 'cpu', config(4, max_distance=1),
                                         torch.Generator().manual_seed(0))


def test_feasible_context_pair_supplies_distinct_targets(monkeypatch):
    extr, intr = overlapping_cameras(monkeypatch, 6)
    monkeypatch.setattr(torch, 'randperm', lambda count, **kwargs: torch.arange(count))
    (left, right), targets = index_tool.find_pair_and_targets(
        extr, intr, 1, 1, 'cpu', config(4), torch.Generator().manual_seed(0))
    assert left == 0
    assert right - left + 1 >= 4
    assert len(targets) == len(set(targets)) == 4
    assert tuple(sorted(targets)) == targets
    assert all(left <= target <= right for target in targets)


def test_no_overlap_keeps_null_index_entry(monkeypatch):
    extr, intr = overlapping_cameras(monkeypatch, 6)
    monkeypatch.setattr(index_tool, 'project_rays', lambda *args: {'overlaps_image': torch.zeros(1)})
    assert index_tool.find_pair_and_targets(extr, intr, 1, 1, 'cpu', config(3),
                                            torch.Generator()) is None
