"""Keep bounded original images that explain independent class evidence."""
import math
import pytest
from tb3_locateanything.visual_memory import Memory

SECOND = 1_000_000_000
BOX = [10, 20, 100, 150]


def observed(memory, stamp, xy, label, support=(10., 10.), complete=True):
    payload = {'source': stamp}
    source = memory.add_view(payload, stamp*SECOND, *xy, 0., dedicated_capture=True)
    obj = memory.remember(source, label, BOX, *support, support_points=[support],
                          class_evidence_complete=complete)
    return source, obj


def add_empty(memory, stamp):
    return memory.add_view({'source': stamp}, stamp*SECOND, -10., float(stamp), 0., dedicated_capture=True)


def assert_all_votes_have_retained_sources(memory):
    assert len(memory.views) <= memory.max_views
    retained = {v.view_id for v in memory.views}
    for obj in memory.landmarks.values():
        assert obj.view_ids <= retained and obj.view_ids
        for label, sources in obj.label_viewpoints.items():
            assert len(sources) == obj.label_votes[label]
            assert all(v['view_id'] in retained for v in sources)


def test_pending_first_class_position_survives_other_class_and_redundant_frames():
    memory = Memory(confirm_classes=True, max_views=4)
    first, obj = observed(memory, 1, (0., 0.), 'table')
    duplicate, _ = observed(memory, 2, (.05, 0.), 'table')
    observed(memory, 3, (1., 0.), 'chair')
    observed(memory, 4, (1.05, 0.), 'chair')
    assert not obj.confirmed and obj.label_votes == {'table': 1, 'chair': 1}
    second_position, again = observed(memory, 5, (2., 0.), 'table')
    assert again is obj and first not in memory.views and duplicate in memory.views
    assert obj.label_votes == {'table': 2, 'chair': 1} and not obj.confirmed
    add_empty(memory, 6)  # Removes another redundant image, not an independent vote.
    assert duplicate in memory.views and second_position in memory.views
    assert obj.label_votes == {'table': 2, 'chair': 1} and not obj.confirmed
    assert obj.object_id.startswith('candidate_')
    assert_all_votes_have_retained_sources(memory)


def test_unique_independent_position_survives_newer_same_class_redundancy():
    memory = Memory(confirm_classes=True, max_views=3)
    unique, obj = observed(memory, 1, (0., 0.), 'chair')
    redundant, _ = observed(memory, 2, (1., 0.), 'chair')
    replacement, _ = observed(memory, 3, (1.1, 0.), 'chair')
    assert obj.confirmed and obj.label_votes == {'chair': 2}
    add_empty(memory, 4)
    assert unique in memory.views and redundant not in memory.views and replacement in memory.views
    assert obj.label_votes == {'chair': 2}
    assert_all_votes_have_retained_sources(memory)


def test_complete_same_position_evidence_is_not_replaced_by_weak_duplicate():
    memory = Memory(confirm_classes=True, max_views=3)
    strong, obj = observed(memory, 1, (0., 0.), 'chair', complete=True)
    weak, _ = observed(memory, 2, (.1, 0.), 'chair', complete=False)
    competitor, _ = observed(memory, 3, (1., 0.), 'table')
    assert obj.label_votes == {'chair': 1, 'table': 1}
    add_empty(memory, 4)
    assert strong in memory.views and weak not in memory.views and competitor in memory.views
    complete, _ = memory._class_votes(obj, complete_only=True)
    assert complete == {'chair': 1, 'table': 1}
    assert_all_votes_have_retained_sources(memory)


def test_sole_physical_instance_has_priority_over_extra_votes_for_another_instance():
    memory = Memory(confirm_classes=True, max_views=3)
    unique, isolated = observed(memory, 1, (0., 0.), 'chair', support=(20., 20.))
    observed(memory, 2, (1., 0.), 'chair')
    observed(memory, 3, (2., 0.), 'chair')
    add_empty(memory, 4)
    assert unique in memory.views and isolated.object_id in memory.landmarks
    assert_all_votes_have_retained_sources(memory)


def test_retention_change_cannot_promote_by_changing_greedy_viewpoint_representatives():
    memory = Memory(confirm_classes=True, max_views=3)
    center, obj = observed(memory, 1, (0., 0.), 'chair')
    observed(memory, 2, (.24, 0.), 'chair')
    observed(memory, 3, (-.24, 0.), 'chair')
    assert obj.label_votes == {'chair': 1} and not obj.confirmed
    add_empty(memory, 4)
    # Removing the old central representative exposes two mutually independent
    # retained sources. Pruning still cannot promote without a new observation.
    assert center not in memory.views
    assert obj.label_votes == {'chair': 2} and not obj.confirmed
    assert memory.serialize()['landmarks'] == []
    assert_all_votes_have_retained_sources(memory)


def test_ttl_and_unavoidable_capacity_loss_do_not_keep_image_less_votes():
    memory = Memory(confirm_classes=True, max_views=2, max_age_sec=3)
    old, obj = observed(memory, 1, (0., 0.), 'table')
    observed(memory, 2, (1., 0.), 'chair')
    add_empty(memory, 3)
    assert old not in memory.views and obj.label_votes == {'chair': 1}
    assert_all_votes_have_retained_sources(memory)
    memory.views_for_query(6*SECOND)
    assert memory.landmarks == {} and not memory.serialize()['pending_landmarks']


def test_clock_reset_removes_retained_votes_and_dedicated_capture_stays_fresh():
    memory = Memory(confirm_classes=True, max_views=4)
    first, obj = observed(memory, 10, (0., 0.), 'chair')
    second, same = observed(memory, 11, (0., 0.), 'chair')
    assert second is not first and second.payload is not first.payload and same is obj
    assert obj.label_votes == {'chair': 1}
    memory.should_capture(0., 0., 0., SECOND)
    assert memory.views == [] and memory.landmarks == {}


def test_legacy_eviction_uses_its_original_four_component_key():
    memory = Memory(confirm_classes=False, max_views=3)
    first, _ = observed(memory, 1, (0., 0.), 'chair')
    observed(memory, 2, (1., 0.), 'chair')
    assert memory._eviction_key(first) == (0, 0, 1, SECOND)
    assert len(memory._eviction_key(first)) == 4
