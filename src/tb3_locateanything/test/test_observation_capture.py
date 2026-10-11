"""Active looks create bounded fresh evidence without changing vote rules."""
import copy
import json

import numpy as np
import pytest
from std_msgs.msg import Bool, String

from test_visual_memory_node import node, stored_view, BOX, SECOND
from tb3_locateanything.visual_memory import Memory
from tb3_locateanything.observation_node import RevisitState
from tb3_locateanything.geometry import Vector3


@pytest.fixture
def active(node):
    node.p['localization_mode'] = 'depth'
    node.memory = Memory(max_views=32, class_labels=['chair'], confirm_classes=True)
    node._reset_observation_capture()
    source = stored_view(node, 90., (0., 0., 0.))
    target = node.memory.remember(source, 'chair', BOX, 2., 0.)
    nearby = stored_view(node, 92., (1., 0., 0.))
    node.memory.remember(nearby, 'chair', BOX, 4., 0.)
    snapshot = copy.deepcopy(nearby.payload)
    snapshot['image'].header.stamp.sec = 99
    snapshot['image'].header.stamp.nanosec = 800_000_000
    snapshot['metadata']['image_stamp_ns'] = int(99.8*SECOND)
    snapshot['depth_m'] = np.ones((480, 640), dtype=np.float32)
    node._capture = lambda: snapshot
    node._publish_memory = lambda: None
    node._save_view = lambda view: None
    node.source, node.nearby, node.snapshot, node.target = source, nearby, snapshot, target
    node.capture_message = dict(epoch=0, request_id='active-0-1', track_id=target.object_id,
                                succeeded_stamp_ns=99*SECOND, expires_stamp_ns=104*SECOND)
    return node


def offer(node, **changes):
    request = dict(node.capture_message, **changes)
    node._observation_capture_cb(String(data=json.dumps(request)))
    return request


def test_dedicated_capture_uses_new_exposure_in_old_grounded_bucket(active):
    assert not active.memory.should_capture(1., 0., 0., int(99.8*SECOND))
    offer(active)
    assert active._observe_dedicated()
    view, label = active.catalog_queue[-1]
    assert view is not active.nearby and view.payload is active.snapshot
    assert view.stamp_ns == int(99.8*SECOND) and label is None
    assert active.nearby in active.memory.views
    assert active.target.confirmed is False  # Arrival/capture adds no class vote.
    active.memory.remember(view, 'chair', BOX, 2., 0.)
    assert active.target.confirmed  # Ordinary independent evidence can confirm.
    offer(active)
    assert not active._observe_dedicated()
    assert len(active.catalog_queue) == 1


def test_dedicated_same_position_cannot_manufacture_independent_votes(active):
    new = active.memory.add_view(copy.deepcopy(active.source.payload), 91*SECOND,
                                 0.1, 0., 0., dedicated_capture=True)
    active.memory.remember(new, 'chair', BOX, 2., 0.)
    assert not active.target.confirmed


def test_already_retained_exposure_cannot_be_relabelled_as_new_capture(active):
    active.memory.add_view(copy.deepcopy(active.snapshot), int(99.8*SECOND),
                           1., 0., 0., dedicated_capture=True)
    offer(active)
    assert active._observe_dedicated()
    assert not active.catalog_queue
    assert not active.capture_requests_consumed


@pytest.mark.parametrize('changes', [
    {'epoch': 1}, {'epoch': True}, {'request_id': ''}, {'track_id': ''},
    {'succeeded_stamp_ns': 101*SECOND}, {'expires_stamp_ns': 100*SECOND},
    {'expires_stamp_ns': 106*SECOND}, {'succeeded_stamp_ns': '99000000000'},
])
def test_invalid_or_expired_request_never_stores(active, changes):
    offer(active, **changes)
    assert not active._observe_dedicated()
    assert not active.catalog_queue


def test_replayed_id_cannot_extend_window(active):
    offer(active)
    offer(active, succeeded_stamp_ns=100*SECOND, expires_stamp_ns=105*SECOND)
    assert active.capture_request['succeeded_stamp_ns'] == 99*SECOND


def test_only_eight_successful_request_identities_per_epoch(active):
    for i in range(8):
        offer(active, request_id=f'active-0-{i}')
        active.capture_request = None
    offer(active, request_id='active-0-extra')
    assert active.capture_request is None
    assert len(active.capture_requests_seen) == 8


@pytest.mark.parametrize('image_time,position,moving', [
    (99.0, (1., 0., 0.), False),
    (99.8, (.1, 0., 0.), False),
    (99.8, (1., 0., 0.), True),
])
def test_old_image_near_source_or_unsettled_robot_waits(active, image_time, position, moving):
    offer(active)
    active.snapshot['image'].header.stamp.sec = int(image_time)
    active.snapshot['image'].header.stamp.nanosec = round((image_time-int(image_time))*SECOND)
    active.snapshot['odom_pose'] = position
    if moving:
        active.stationary_since = None
    assert active._observe_dedicated()
    assert not active.catalog_queue
    active.now_ns = 104*SECOND
    assert not active._observe_dedicated()


def test_user_command_cancels_and_blocks_late_capture_until_resume(active):
    offer(active)
    active._capture_user_command_cb(String(data='go to chair 0'))
    offer(active)
    assert not active._observe_dedicated()
    active._capture_enabled_cb(Bool(data=True))
    offer(active)  # This completion preceded the command and remains stale.
    assert active.capture_request is None


def test_disable_and_clock_rollback_close_capture_eligibility(active):
    offer(active)
    active._capture_enabled_cb(Bool(data=False))
    assert active.capture_request is None
    active._capture_enabled_cb(Bool(data=True))
    active._reset_observation_capture()
    offer(active)
    active.now_ns = 98*SECOND
    assert not active._observe_dedicated()
    offer(active)
    assert active.capture_request is None


def test_epoch_reset_drops_old_identities_but_not_old_message_validation(active):
    offer(active)
    active.epoch = 1
    active._reset_observation_capture()
    offer(active)
    assert active.capture_request is None
    assert not active.capture_requests_seen


def test_sensor_failure_does_not_consume_and_can_retry_within_window(active):
    offer(active)
    snapshot = active.snapshot
    active._capture = lambda: (_ for _ in ()).throw(ValueError('depth not aligned'))
    assert active._observe_dedicated()
    assert not active.capture_requests_consumed
    active._capture = lambda: snapshot
    assert active._observe_dedicated()
    assert len(active.catalog_queue) == 1


def test_capture_finishing_after_expiration_cannot_store(active):
    offer(active)
    def late():
        active.now_ns = 104*SECOND
        return active.snapshot
    active._capture = late
    assert active._observe_dedicated()
    assert not active.catalog_queue


def test_no_force_in_laser_mode_or_while_querying(active):
    active.p['localization_mode'] = 'laser'
    offer(active)
    assert active.capture_request is None
    active.p['localization_mode'] = 'depth'
    active.pending = {'id': 1}
    offer(active)
    assert active.capture_request is None


@pytest.mark.parametrize('answer,confirmed', [
    ('<box>None</box>', False),
    ('<box><156><208><391><833></box>', True),
])
def test_actual_observer_request_new_capture_then_normal_catalog_decides(active, answer, confirmed):
    ledger = RevisitState(namespace='capture-integration')
    ledger.clock(98*SECOND)
    ledger.reconcile([active.target.object_id], 98*SECOND)
    goal = ledger.offer(active.target.object_id, (1., 0., 0.), (1., 0.), 98*SECOND)
    assert ledger.capture_request(98*SECOND, active.epoch) is None
    ledger.event({**goal, 'event': 'dispatched'}, 98*SECOND)
    ledger.event({**goal, 'event': 'succeeded'}, 99*SECOND, active.epoch)
    request = ledger.capture_request(100*SECOND, active.epoch)
    active._observation_capture_cb(String(data=json.dumps(request)))
    assert active._observe_dedicated()
    view, _ = active.catalog_queue.popleft()
    active._refresh_positions = lambda: None
    active._localize = lambda *_: (Vector3(2., 0., 0.), {})
    active._catalog_result(dict(request_id='catalog', view=view, label='chair'),
                           dict(request_id='catalog', answer=answer, inference_seconds=.2))
    assert active.target.confirmed is confirmed
    assert ledger.total_attempts == 1
