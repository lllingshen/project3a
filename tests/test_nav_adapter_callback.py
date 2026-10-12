"""ROS message checks without launching nodes, Gazebo, or navigation."""

import sys
from pathlib import Path
from types import SimpleNamespace, MethodType

import pytest

pytest.importorskip("rclpy")
import tf2_ros
from rclpy.time import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rosgraph_msgs.msg import Clock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/tb3_nav_adapter"))
from tb3_nav_adapter.nav_goal_adapter_node import NavGoalAdapterNode
from tb3_nav_adapter.safe_approach import CostGrid


def adapter(buffer):
    goals, warnings, audit = [], [], []
    logger = SimpleNamespace(info=lambda _: None, debug=lambda _: None,
                             warn=warnings.append)
    obj = SimpleNamespace(
        _target_frame="map", _tf_timeout=0.0, _tf_buffer=buffer,
        _approach_dist=0.5, _min_standoff=0.3,
        _max_approach=1., _max_cost=60, _cost_age=3., _robot_tf_age=.75,
        _cost_grid=CostGrid(101, 101, .05, -2.525, -2.525, 0, [0]*10201),
        _cost_stamp_ns=41_500_000_000, _last_clock_ns=None,
        _status_pub=SimpleNamespace(publish=audit.append), _audit_messages=audit,
        _goal_pub=SimpleNamespace(publish=goals.append),
        get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=42)),
    )
    for name in ('_observe_clock', '_on_clock', '_cost_cb', '_audit'):
        setattr(obj, name, MethodType(getattr(NavGoalAdapterNode, name), obj))
    obj._stamp_ns = NavGoalAdapterNode._stamp_ns
    return obj, goals, warnings


def target(frame="map"):
    return SimpleNamespace(
        success=True, frame_id=frame,
        header=Header(frame_id=frame, stamp=Time(seconds=10).to_msg()),
        position=Point(x=0.0, y=0.0),
        object_id="selected_7", semantic_name="the red chair",
    )


def robot_transform(*args, **kwargs):
    transform = TransformStamped()
    transform.header.stamp = Time(seconds=41.8).to_msg()
    transform.transform.translation.x = -2.0
    return transform


def test_map_origin_target_produces_robot_relative_goal():
    obj, goals, _ = adapter(SimpleNamespace(lookup_transform=robot_transform))
    NavGoalAdapterNode._query_cb(obj, target())
    assert len(goals) == 1
    assert goals[0].header.frame_id == "map"
    assert goals[0].pose.position.x == pytest.approx(-0.5)


def test_missing_robot_transform_never_publishes_fallback():
    def unavailable(*args, **kwargs):
        raise tf2_ros.LookupException("no map to robot transform")
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=unavailable))
    NavGoalAdapterNode._query_cb(obj, target())
    assert goals == []
    assert "No goal" in warnings[0]


def test_body_frame_target_transform_preserves_observation_timestamp():
    observed_stamps = []
    def transform(point, frame, **kwargs):
        observed_stamps.append(point.header.stamp.sec)
        result = PointStamped()
        result.header.frame_id = frame
        result.point = Point(x=0.0, y=0.0)
        return result
    obj, goals, _ = adapter(SimpleNamespace(
        transform=transform, lookup_transform=robot_transform))
    NavGoalAdapterNode._query_cb(obj, target("base_link"))
    assert observed_stamps == [10]
    assert len(goals) == 1
    assert goals[0].header.stamp.sec == 42


def test_missing_frame_never_publishes_goal():
    obj, goals, warnings = adapter(SimpleNamespace())
    NavGoalAdapterNode._query_cb(obj, target(""))
    assert goals == []
    assert "no coordinate frame" in warnings[0]


@pytest.mark.parametrize('stamp', [38.9, 42.1])
def test_stale_or_future_costmap_never_publishes(stamp):
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    obj._cost_stamp_ns = int(stamp*1e9)
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals
    assert 'fresh full global costmap' in warnings[0]


@pytest.mark.parametrize('stamp', [41.0, 42.1])
def test_stale_or_future_robot_transform_never_publishes(stamp):
    def lookup(*args, **kwargs):
        tf = robot_transform()
        tf.header.stamp = Time(seconds=stamp).to_msg()
        return tf
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=lookup))
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals
    assert 'fresh robot TF' in warnings[0]


def test_clock_rollback_clears_cached_navigation_context():
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    obj._observe_clock(42_000_000_000)
    obj.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=1))
    obj._on_clock(Clock(clock=Time(seconds=1).to_msg()))
    assert obj._cost_grid is None and obj._cost_stamp_ns is None
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals


def test_query_detects_small_rollback_before_own_clock_callback():
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    obj._last_clock_ns = 42_100_000_000
    NavGoalAdapterNode._query_cb(obj, target())  # TimeSource is already at 42.0.
    assert not goals and obj._cost_grid is None


def test_delayed_clock_message_does_not_clear_current_context():
    obj, _, _ = adapter(SimpleNamespace())
    obj._observe_clock(42_000_000_000)
    obj._on_clock(Clock(clock=Time(seconds=41.9).to_msg()))
    assert obj._cost_grid is not None  # Node's current TimeSource remains 42.0.


def test_costmap_callback_checks_full_shape_and_frame():
    obj, goals, warnings = adapter(SimpleNamespace())
    msg = OccupancyGrid()
    msg.header.frame_id = 'odom'
    obj._cost_cb(msg)
    assert obj._cost_grid is None
    msg.header.frame_id = 'map'
    msg.header.stamp = Time(seconds=42).to_msg()
    msg.info.width = msg.info.height = 2
    msg.info.resolution = .1
    msg.info.origin.orientation.w = 1.
    msg.data = [0]*3
    obj._cost_cb(msg)
    assert obj._cost_grid is None
    msg.data = [0]*4
    obj._cost_cb(msg)
    assert obj._cost_grid.width == 2
    assert obj._cost_stamp_ns == 42_000_000_000


def test_safe_callback_selects_alternative_and_publishes_audit():
    import json
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    grid = obj._cost_grid
    grid.data[grid.cell(-.5, 0)] = 100
    NavGoalAdapterNode._query_cb(obj, target())
    assert len(goals) == 1 and not warnings
    pose = goals[0].pose.position
    assert grid.data[grid.cell(pose.x, pose.y)] <= 60
    assert (pose.x, pose.y) != (-.5, 0)
    audit = json.loads(obj._audit_messages[-1].data)
    assert audit['state'] == 'published'
    assert audit['max_approach_distance'] == 1.
    assert .5 <= audit['target_distance'] <= 1.
    assert audit['cost'] <= 60
    assert audit['robot_tf_stamp_ns'] == 41_800_000_000


def test_no_safe_approach_rejects_without_publishing_fallback():
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    grid = obj._cost_grid
    for index in range(len(grid.data)):
        x, y = grid.center(index)
        if x*x+y*y < 1.01:
            grid.data[index] = 100
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals
    assert 'no safe reachable approach' in warnings[0]


def test_nonfinite_target_is_rejected_without_invalid_json():
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    msg = target()
    msg.position.x = float('nan')
    NavGoalAdapterNode._query_cb(obj, msg)
    assert not goals
    assert 'invalid target' in warnings[0]


def test_search_rechecks_freshness_before_goal_publication(monkeypatch):
    from tb3_nav_adapter import nav_goal_adapter_node as module
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    original = module.select_safe_approach
    def slow_search(*args):
        result = original(*args)
        obj.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=43))
        return result
    monkeypatch.setattr(module, 'select_safe_approach', slow_search)
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals
    assert 'expired during approach search' in warnings[0]


def test_small_rollback_during_search_cannot_publish_goal(monkeypatch):
    from tb3_nav_adapter import nav_goal_adapter_node as module
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    original = module.select_safe_approach
    def rollback(*args):
        result = original(*args)
        obj.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=41.95))
        return result
    monkeypatch.setattr(module, 'select_safe_approach', rollback)
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals and obj._cost_grid is None


def test_monotonic_search_delay_expires_context_when_ros_clock_is_frozen(monkeypatch):
    from tb3_nav_adapter import nav_goal_adapter_node as module
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=robot_transform))
    wall = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: wall[0])
    original = module.select_safe_approach
    def slow_search(*args):
        result = original(*args)
        wall[0] += .8  # Single-threaded ROS TimeSource still reads 42.0.
        return result
    monkeypatch.setattr(module, 'select_safe_approach', slow_search)
    NavGoalAdapterNode._query_cb(obj, target())
    assert not goals
    assert 'expired during approach search' in warnings[0]


def test_real_ros_adapter_construction_and_safe_goal_publish():
    """Opt-in isolated ROS smoke also catches Node attribute/API collisions."""
    import os
    if os.environ.get('RUN_ROS_ADAPTER_SMOKE') != '1':
        pytest.skip('set RUN_ROS_ADAPTER_SMOKE=1 with an isolated ROS domain')
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from tb3_interfaces.msg import SemanticQueryResult
    import time
    rclpy.init()
    node = None
    try:
        node = NavGoalAdapterNode()
        observed = []
        subscription = node.create_subscription(
            PoseStamped, '/nav_goal_adapter_node/goal_pose', observed.append, 10)
        deadline = time.monotonic()+2.
        while node._goal_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
        assert node._goal_pub.get_subscription_count() == 1
        now = node.get_clock().now().to_msg()
        tf = TransformStamped()
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'base_link'
        tf.header.stamp = now
        tf.transform.translation.x = -2.
        tf.transform.rotation.w = 1.
        node._tf_buffer.set_transform(tf, 'isolated adapter smoke')
        # Bracket current time as a live TF publisher would, so the adapter
        # requests an interpolated current pose rather than a future latest TF.
        tf.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds+1_000_000_000).to_msg()
        node._tf_buffer.set_transform(tf, 'isolated adapter smoke')
        cost = OccupancyGrid()
        cost.header.frame_id = 'map'
        cost.header.stamp = now
        cost.info.width = cost.info.height = 101
        cost.info.resolution = .05
        cost.info.origin.position.x = cost.info.origin.position.y = -2.525
        cost.info.origin.orientation.w = 1.
        cost.data = [0]*10201
        node._cost_cb(cost)
        msg = SemanticQueryResult()
        msg.success = True
        msg.frame_id = msg.header.frame_id = 'map'
        msg.header.stamp = now
        msg.object_id = 'smoke_object'
        node._query_cb(msg)
        deadline = time.monotonic()+1.
        while not observed and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
        assert len(observed) == 1
        assert observed[0].pose.position.x == pytest.approx(-.5)
        node.destroy_subscription(subscription)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('case,expected_mode', [
    ('clock_ahead_4ms', 'latest_cached'),
    ('future_bracketed', 'current_cached_interpolation'),
    ('future_unbracketed', None),
    ('stale', None),
    ('rollback', None),
])
def test_real_single_executor_dds_clock_and_tf_phases(case, expected_mode):
    """Publish actual /clock, /tf and input messages, without future-cache fabrication."""
    import os
    if os.environ.get('RUN_ROS_ADAPTER_SMOKE') != '1':
        pytest.skip('set RUN_ROS_ADAPTER_SMOKE=1 with an isolated ROS domain')
    import json
    import time
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    from tb3_interfaces.msg import SemanticQueryResult
    rclpy.init()
    node = driver = executor = None
    try:
        node = NavGoalAdapterNode()
        node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        driver = Node('isolated_adapter_clock_driver')
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(driver)
        clock_pub = driver.create_publisher(Clock, '/clock', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        tf_pub = driver.create_publisher(TFMessage, '/tf', 10)
        cost_pub = driver.create_publisher(OccupancyGrid, '/global_costmap/costmap', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        target_pub = driver.create_publisher(
            SemanticQueryResult, '/semantic_query_node/selected_target', 10)
        observed, status = [], []
        driver.create_subscription(PoseStamped, '/nav_goal_adapter_node/goal_pose', observed.append, 10)
        driver.create_subscription(String, '/nav_goal_adapter_node/status',
                                   lambda msg: status.append(json.loads(msg.data)), 10)

        def wait_for(predicate):
            end = time.monotonic()+2.
            while not predicate() and time.monotonic() < end:
                executor.spin_once(timeout_sec=.01)
            assert predicate()

        wait_for(lambda: target_pub.get_subscription_count() == 1
                 and clock_pub.get_subscription_count() >= 2
                 and tf_pub.get_subscription_count() == 1
                 and cost_pub.get_subscription_count() == 1
                 and node._goal_pub.get_subscription_count() == 1
                 and node._status_pub.get_subscription_count() == 1)
        now_ns = 364_100_000_000 if case == 'rollback' else 364_000_000_000
        clock_pub.publish(Clock(clock=Time(nanoseconds=now_ns).to_msg()))
        wait_for(lambda: node.get_clock().now().nanoseconds == now_ns)

        def publish_tf(stamp_ns):
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = 'map', 'base_link'
            tf.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
            tf.transform.translation.x = -2.
            tf.transform.rotation.w = 1.
            tf_pub.publish(TFMessage(transforms=[tf]))
            def received():
                if not node._tf_buffer.can_transform('map', 'base_link', Time()):
                    return False
                value = node._tf_buffer.lookup_transform('map', 'base_link', Time(), timeout=Duration())
                return node._stamp_ns(value.header.stamp) == stamp_ns
            wait_for(received)

        if case == 'future_bracketed':
            publish_tf(now_ns-200_000_000)
            publish_tf(now_ns+40_000_000)
        elif case == 'future_unbracketed':
            publish_tf(now_ns+40_000_000)
        elif case == 'stale':
            publish_tf(now_ns-800_000_000)
        else:
            publish_tf(now_ns-4_000_000)
        cost = OccupancyGrid()
        cost.header.frame_id = 'map'
        cost.header.stamp = Time(nanoseconds=now_ns).to_msg()
        cost.info.width = cost.info.height = 101
        cost.info.resolution = .05
        cost.info.origin.position.x = cost.info.origin.position.y = -2.525
        cost.info.origin.orientation.w = 1.
        cost.data = [0]*10201
        cost_pub.publish(cost)
        wait_for(lambda: node._cost_grid is not None)
        if case == 'rollback':
            # Current TimeSource changes through DDS before the query arrives;
            # no direct method calls or synthetic replacement clock functions.
            now_ns -= 100_000_000
            clock_pub.publish(Clock(clock=Time(nanoseconds=now_ns).to_msg()))
            wait_for(lambda: node.get_clock().now().nanoseconds == now_ns)
        msg = SemanticQueryResult()
        msg.success = True
        msg.frame_id = msg.header.frame_id = 'map'
        msg.header.stamp = Time(nanoseconds=now_ns).to_msg()
        msg.object_id = 'phase_test'
        started = time.monotonic()
        target_pub.publish(msg)
        wait_for(lambda: bool(status))
        elapsed = time.monotonic()-started
        assert elapsed < .4  # Would fail the old blocking .5-second lookup.
        if expected_mode:
            wait_for(lambda: bool(observed))
            assert len(observed) == 1
            assert status[-1]['state'] == 'published'
            assert status[-1]['robot_tf_lookup_mode'] == expected_mode
            assert status[-1]['robot_tf_stamp_ns'] <= now_ns
            assert (now_ns-status[-1]['robot_tf_stamp_ns'])/1e9 <= .75
            assert status[-1]['robot_tf_stamp_ns'] == (
                now_ns-4_000_000 if case == 'clock_ahead_4ms' else now_ns)
        else:
            assert observed == [] and status[-1]['state'] == 'rejected'
            if case == 'stale':
                assert 'fresh robot TF' in status[-1]['reason']
            elif case == 'rollback':
                assert 'fresh full global costmap' in status[-1]['reason']
                assert node._cost_grid is None
            else:
                assert 'TF unavailable' in status[-1]['reason']
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if driver is not None:
            driver.destroy_node()
        rclpy.shutdown()
