import base64
import copy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from tb3_locateanything.gallery import build_gallery, has_spatial_relation, select_gallery_box
from tb3_locateanything.visual_memory import Memory


def add_view(memory, stamp=1, x=0):
    image = np.zeros((240, 320, 3), np.uint8)
    image[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] % 256
    image[:, :, 1] = np.arange(240, dtype=np.uint8)[:, None]
    ok, png = cv2.imencode('.png', image)
    assert ok
    return memory.add_view({'image_png': base64.b64encode(png).decode('ascii'),
                            'scan': object(), 'scan_to_odom': object()}, stamp * 10**9, x, 0, 0)


def two_objects():
    memory = Memory()
    view = add_view(memory)
    memory.remember(view, 'chair', [20.25, 25.5, 115.75, 190.25], 1, 1)
    memory.remember(view, 'chair', [175.5, 30.25, 295.25, 200.75], 4, 1)
    return memory, view


def test_pending_same_class_cannot_be_omitted_from_description_gallery():
    memory, view = two_objects()
    memory.landmarks['chair_1'].confirmed = False
    with pytest.raises(ValueError, match='unconfirmed class evidence.*chair_1'):
        build_gallery(memory, 'chair')


def test_pending_unrelated_class_does_not_block_confirmed_gallery():
    memory, view = two_objects()
    pending = memory.remember(view, 'person', [125, 30, 160, 180], 8, 1)
    pending.confirmed = False
    assert len(build_gallery(memory, 'chair').entries) == 2


def test_gallery_preserves_original_coordinates_payload_and_png():
    memory, view = two_objects()
    before = copy.deepcopy(view.observations)
    payload = view.payload
    gallery = build_gallery(memory)
    assert len(gallery.entries) == 2
    assert view.observations == before and view.payload is payload
    for entry, observation in zip(gallery.entries, before):
        assert entry.view is view
        assert entry.source_box == observation['box']
        assert entry.source_box is not view.observations[0]['box']
        assert select_gallery_box(gallery, entry.object_box) is entry
        assert entry.source_box != entry.object_box
        # Invert the rendering transform: fractional source coordinates survive.
        crop, content = entry.source_crop, entry.content_box
        sx = (content[2] - content[0]) / (crop[2] - crop[0])
        sy = (content[3] - content[1]) / (crop[3] - crop[1])
        recovered = [(entry.object_box[i] - content[i % 2]) / (sx if i % 2 == 0 else sy)
                     + crop[i % 2] for i in range(4)]
        assert recovered == pytest.approx(entry.source_box)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(gallery.image_png), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, gallery.image_bgr)


def test_larger_truncated_view_does_not_replace_complete_evidence():
    memory = Memory()
    complete = add_view(memory)
    memory.remember(complete, 'chair', [40, 30, 110, 180], 1, 1)
    closeup = add_view(memory, 2, 1)
    memory.remember(closeup, 'chair', [20, 0, 210, 220], 1, 1)
    gallery = build_gallery(memory)
    assert len(gallery.entries) == 1 and gallery.entries[0].view is complete


def test_larger_complete_evidence_wins_over_newer_tiny_box():
    memory = Memory()
    wide = add_view(memory)
    memory.remember(wide, 'chair', [20, 20, 120, 200], 1, 1)
    distant = add_view(memory, 2, 1)
    memory.remember(distant, 'chair', [30, 30, 60, 80], 1, 1)
    assert build_gallery(memory).entries[0].view is wide


@pytest.mark.parametrize("correct_box", [[40, 40, 110, 170], [0, 40, 110, 170]])
def test_confirmed_gallery_never_reuses_larger_wrong_class_evidence(correct_box):
    memory = Memory(confirm_classes=True, class_labels=["forklift", "storage_rack"])
    wrong = add_view(memory)
    obj = memory.remember(wrong, "storage_rack", [20, 20, 300, 210], 1., 1.,
                          support_points=[(1., 1.)])
    correct_views = []
    for stamp in (2, 3, 4):
        view = add_view(memory, stamp, stamp)
        correct_views.append(view)
        assert memory.remember(view, "forklift", correct_box, 1., 1.,
                               support_points=[(1., 1.)]) is obj
    assert obj.confirmed and obj.object_id == "forklift_0" and not obj.conflicted
    assert obj.label_votes == {"storage_rack": 1, "forklift": 3}
    before = memory.serialize()
    if correct_box[0] == 0:
        with pytest.raises(ValueError, match="no usable retained visual instance"):
            build_gallery(memory, "forklift")
    else:
        entry, = build_gallery(memory, "forklift").entries
        assert entry.view is correct_views[-1] and entry.source_box == correct_box
        assert entry.view is not wrong
    assert memory.serialize() == before


def test_requested_class_uses_stored_label_not_id_or_pixel_color():
    memory, view = two_objects()
    memory.remember(view, 'trash_can', [130, 30, 160, 90], 8, 1)
    gallery = build_gallery(memory, requested_label='Trash Can')
    assert [entry.object_id for entry in gallery.entries] == ['trash_can_0']
    with pytest.raises(ValueError, match='no usable'):
        build_gallery(memory, requested_label='table')


@pytest.mark.parametrize('box', [[0, 20, 100, 180], [220, 20, 320, 180],
                               [20, 20, 24, 80], [20, 20, 30, 30]])
def test_clipped_or_too_small_evidence_cannot_form_gallery(box):
    memory = Memory()
    view = add_view(memory)
    memory.remember(view, 'chair', box, 1, 1)
    with pytest.raises(ValueError, match='no usable'):
        build_gallery(memory)


def test_removed_landmarks_and_evicted_views_are_not_resurrected():
    memory, view = two_objects()
    memory.landmarks.pop('chair_0')
    assert [entry.object_id for entry in build_gallery(memory).entries] == ['chair_1']
    memory.views.clear()
    with pytest.raises(ValueError, match='no usable'):
        build_gallery(memory)


def test_bad_image_does_not_hide_another_valid_view():
    memory, view = two_objects()
    bad = add_view(memory, 2, 1)
    memory.remember(bad, 'chair', [30, 20, 140, 210], 1, 1)
    bad.payload['image_png'] = 'not valid base64!'
    assert all(entry.view is view for entry in build_gallery(memory).entries)


def test_candidate_limit_fails_instead_of_silently_omitting_a_target():
    memory, _ = two_objects()
    with pytest.raises(ValueError, match='too many'):
        build_gallery(memory, max_objects=1)
    with pytest.raises(ValueError, match='positive integer'):
        build_gallery(memory, max_objects=0)


def test_missing_visual_evidence_for_one_candidate_does_not_silently_drop_it():
    memory, view = two_objects()
    view.observations[1]["box"] = [180, 30, 184, 90]
    with pytest.raises(ValueError, match="incomplete.*chair_1"):
        build_gallery(memory, requested_label="chair")


@pytest.mark.parametrize("requested_label", [None, "chair", "person"])
def test_conflicted_blue_chair_cannot_be_dropped_to_leave_only_red_chair(requested_label):
    memory = Memory()
    image = np.zeros((240, 320, 3), np.uint8)
    red_box, blue_box = [20, 25, 115, 190], [175, 30, 295, 200]
    image[25:190, 20:115] = (0, 0, 255)
    image[30:200, 175:295] = (255, 0, 0)
    ok, png = cv2.imencode('.png', image)
    assert ok
    view = memory.add_view({'image_png': base64.b64encode(png).decode('ascii')}, 10**9, 0, 0, 0)
    red = memory.remember(view, 'chair', red_box, 1, 1)
    blue = memory.remember(view, 'chair', blue_box, 4, 1)
    memory.remember(view, 'person', blue_box, 4, 1)
    assert not red.conflicted and blue.conflicted
    assert blue.label_votes == {'chair': 1, 'person': 1}
    with pytest.raises(ValueError, match='conflicting class evidence.*chair_1'):
        build_gallery(memory, requested_label)


def test_conflict_in_unrelated_classes_does_not_block_a_valid_requested_class():
    memory, view = two_objects()
    memory.remember(view, 'person', view.observations[0]['box'], 1, 1)
    memory.remember(view, 'fire extinguisher', [130, 40, 160, 140], 8, 1)
    gallery = build_gallery(memory, 'fire extinguisher')
    assert [entry.object_id for entry in gallery.entries] == ['fire_extinguisher_0']


def test_gallery_checks_historical_conflict_after_that_class_image_expires():
    memory = Memory(max_age_sec=3)
    first = add_view(memory)
    box = [20, 20, 120, 200]
    obj = memory.remember(first, 'chair', box, 1, 1, support_points=[(1., 1.)])
    memory.remember(first, 'person', box, 1, 1, support_points=[(1., 1.)])
    later = add_view(memory, 3, 3)
    memory.remember(later, 'chair', box, 1, 1, support_points=[(1., 1.)])
    memory.views_for_query(5 * 10**9)
    assert obj.label_votes == {'chair': 1} and 'person' in obj.seen_known_labels
    with pytest.raises(ValueError, match='conflicting class evidence.*clear memory'):
        build_gallery(memory, 'person')


def test_cross_view_supported_instance_appears_once_in_gallery():
    memory = Memory()
    first = add_view(memory)
    obj = memory.remember(first, "forklift", [20, 20, 120, 200], 1, 1,
                          support_points=[(1., 1.), (1.5, 1.)])
    second = add_view(memory, 2, 1)
    memory.remember(second, "forklift", [30, 30, 140, 210], 1.9, 1,
                    support_points=[(1.6, 1.), (1.9, 1.)])
    gallery = build_gallery(memory, "forklift")
    assert len(gallery.entries) == 1
    assert gallery.entries[0].object_id == obj.object_id
    assert gallery.entries[0].view is second


def test_mapping_is_by_geometry_even_if_entry_list_order_changes():
    memory, _ = two_objects()
    gallery = build_gallery(memory)
    expected = gallery.entries[0]
    gallery.entries.reverse()
    assert select_gallery_box(gallery, expected.object_box) is expected


def test_grid_spanning_gap_caption_and_outside_boxes_fail():
    memory, _ = two_objects()
    gallery = build_gallery(memory)
    left, right = gallery.entries
    invalid = [[left.object_box[0], left.object_box[1], right.object_box[2], right.object_box[3]],
               [left.tile_box[2] + 1, 30, right.tile_box[0] - 1, 100],
               [20, 20, 100, 35], [-1, 20, 30, 100], [0, 0, float('nan'), 10]]
    for box in invalid:
        with pytest.raises(ValueError):
            select_gallery_box(gallery, box)


def test_duplicate_geometry_is_ambiguous_instead_of_first_entry_wins():
    memory, _ = two_objects()
    gallery = build_gallery(memory)
    original = gallery.entries[0]
    gallery.entries.append(replace(original, object_id='unrelated_0'))
    with pytest.raises(ValueError, match='ambiguous'):
        select_gallery_box(gallery, original.object_box)


@pytest.mark.parametrize('text', ['chair to the right of the red chair', 'leftmost chair',
    'person behind the desk', 'table in front of me', 'beside it', 'next to it', 'between chairs',
    'nearest shelf', 'farthest person', 'previous chair', 'another chair', 'the same table again',
    'the person at the corner', 'the chairs in the corners'])
def test_spatial_relations_and_coreference_keep_original_scene_context(text):
    assert has_spatial_relation(text)


@pytest.mark.parametrize('text', ['the red chair', 'the bright blue table', 'the upright shelf',
                                 'red object I can sit on'])
def test_appearance_and_function_do_not_accidentally_match_relation_substrings(text):
    assert not has_spatial_relation(text)
