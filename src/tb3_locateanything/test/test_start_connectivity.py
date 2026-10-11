"""Unknown-start bridge stays inside verified footprint and traversable costs."""
from dataclasses import replace
import math

import pytest
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.time import Time
import tf2_ros

from tb3_locateanything import observation_planner as planner
from tb3_locateanything.observation_node import SECOND, transform_xy
from test_observation_node import node, transform, lookup


def grid(data=None, width=5, height=3, resolution=1., origin=(0., 0.), yaw=0.):
    return planner.Grid(width, height, resolution, *origin, yaw,
                        data if data is not None else [0]*(width*height))


def marked(g, col, row, value):
    data = list(g.data); data[row*g.width+col] = value
    return replace(g, data=data)


POLYGON = ((0., 1.), (1., 1.), (1., 2.), (0., 2.))


def test_unknown_start_whole_cell_bridge_keeps_map_immutable():
    g = marked(grid(), 0, 1, -1); costs = grid(); before = list(g.data)
    assert max(planner._reachable(g, costs, (.5, 1.5))) == -1
    distances = planner._reachable(g, costs, (.5, 1.5), POLYGON)
    assert sum(d >= 0 for d in distances) == 14 and distances[5] == -1 and g.data == before


@pytest.mark.parametrize('raw,cost', [(100, 0), (1, 0), (-1, -1), (-1, 99), (-1, 100)])
def test_no_exception_for_occupied_start_or_blocked_cost(raw, cost):
    g = marked(grid(), 0, 1, raw); costs = marked(grid(), 0, 1, cost)
    assert max(planner._reachable(g, costs, (.5, 1.5), POLYGON)) == -1


@pytest.mark.parametrize('cost', [0, 1, 43, 60, 98])
def test_traversable_inflation_under_unknown_start_needs_verified_footprint(cost):
    g = marked(grid(), 0, 1, -1); costs = marked(grid(), 0, 1, cost)
    before = list(g.data), list(costs.data)
    assert max(planner._reachable(g, costs, (.5, 1.5))) == -1
    result = planner._reachable(g, costs, (.5, 1.5), POLYGON)
    assert sum(d >= 0 for d in result) == 14
    assert result[5] == -1  # unknown footprint cells cannot become goals
    assert (g.data, costs.data) == before


def test_known_start_never_uses_footprint_to_cross_unknown_barrier():
    g = grid()
    for row in range(g.height): g = marked(g, 1, row, -1)
    enormous = ((-1., -1.), (10., -1.), (10., 10.), (-1., 10.))
    a = planner._reachable(g, grid(), (.5, 1.5))
    assert planner._reachable(g, grid(), (.5, 1.5), enormous) == a
    assert a[2+g.width] == -1


def test_unknown_outside_footprint_and_occupied_corners_stay_barriers():
    g = grid()
    for row in range(g.height): g = marked(g, 1, row, -1)
    g = marked(g, 0, 1, -1)
    assert planner._reachable(g, grid(), (.5, 1.5), POLYGON)[2+g.width] == -1
    g = grid([-1, 100, 100, 0], width=2, height=2)
    assert planner._reachable(g, grid(width=2,height=2), (.5,.5),
        ((0.,0.),(2.,0.),(2.,2.),(0.,2.)))[3] == -1


@pytest.mark.parametrize('polygon', [(), ((0.,1.),(1.,1.)),
    ((0.,1.),(1.,1.),(float('nan'),2.)),
    ((0.,1.),(1.,2.),(0.,2.),(1.,1.)),
    ((0.,1.),(1.,1.),(.4,1.3),(1.,2.),(0.,2.)),
    tuple((math.cos(i*.1), math.sin(i*.1)) for i in range(65)),
    ((.1,1.1),(.9,1.1),(.9,1.9),(.1,1.9))])
def test_invalid_or_center_only_footprint_does_not_certify_full_cell(polygon):
    assert max(planner._reachable(marked(grid(),0,1,-1),grid(),(.5,1.5),polygon)) == -1


def test_reversed_polygon_and_shared_boundary_cost_do_not_change_coverage():
    g=marked(grid(),0,1,-1); costs=marked(grid(),1,1,100)
    result=planner._reachable(g,costs,(.5,1.5),tuple(reversed(POLYGON)))
    assert result[0] >= 0  # Neighbor only touches bridge boundary, not its area.
    assert result[g.width+1] == -1


@pytest.mark.parametrize('blocker', [-1, 99, 100])
def test_finer_cost_cells_all_checked_not_only_center(blocker):
    g=marked(grid(),0,1,-1);costs=grid(width=20,height=12,resolution=.25)
    assert planner._reachable(g,costs,(.5,1.5),POLYGON)[0] >= 0
    costs=marked(costs,0,4,43)  # Ordinary inflation at a cell corner is allowed.
    assert planner._reachable(g,costs,(.5,1.5),POLYGON)[0] >= 0
    costs=marked(costs,0,4,blocker)  # Unknown/collision at any overlap is blocked.
    assert costs.value(.5,1.5)==0
    assert max(planner._reachable(g,costs,(.5,1.5),POLYGON)) == -1


def test_cost_coverage_outside_grid_is_not_clipped():
    g=marked(grid(),0,1,-1);costs=grid(width=10,height=10,resolution=.25,origin=(.25,0.))
    assert costs.value(.5,1.5)==0
    assert max(planner._reachable(g,costs,(.5,1.5),POLYGON)) == -1


def test_rotated_differing_cost_grid_checks_actual_intersections():
    g=marked(grid(),0,1,-1);costs=grid(width=30,height=30,resolution=.25,origin=(-3.,-3.),yaw=.31)
    assert planner._reachable(g,costs,(.5,1.5),POLYGON)[0] >= 0
    col,row=costs.cell(.03,1.03); costs=marked(costs,col,row,100)
    assert costs.value(.5,1.5)==0
    assert max(planner._reachable(g,costs,(.5,1.5),POLYGON)) == -1


def test_rotated_map_cell_containment_and_route():
    g=grid(width=5,height=3,resolution=.2,origin=(4.,-1.),yaw=.7);g=marked(g,0,1,-1)
    robot=g.world_center(0,1);polygon=planner._cell_corners(g,0,1)
    assert planner._reachable(g,replace(g,data=[0]*15),robot,polygon)[0]>=0


def test_planner_returns_only_original_known_free_endpoint():
    costs=grid(width=80,height=80,resolution=.1,origin=(-4.,-4.))
    g=marked(costs,10,40,-1);robot=g.world_center(10,40)
    poly=planner._cell_corners(g,10,40)
    goal=planner.choose_observation_candidate(g,costs,robot,(0.,0.),[(-1.,0.)],[],
                                              start_footprint=poly)
    assert goal is not None and g.value(*goal['pose'][:2])==0


def footprint(seconds=99., vertices=POLYGON, frame='map'):
    msg=PolygonStamped();msg.header.frame_id=frame
    msg.header.stamp=Time(nanoseconds=int(seconds*SECOND)).to_msg()
    msg.polygon.points=[Point32(x=float(x),y=float(y),z=0.) for x,y in vertices]
    return msg


def unknown_start_node(n):
    for msg in (n.map_msg,n.cost_msg):
        msg.info.origin.position.x=5.;msg.info.origin.position.y=15.
    n.map_msg.data[5*10+5]=-1
    n.footprint_msgs=[]
    original=n.tf_buffer.lookup_transform
    def stamped(target,source,*args):
        result=original(target,source,*args)
        result.child_frame_id=source
        return result
    n.tf_buffer.lookup_transform=stamped


def test_exact_historical_tf_reprojects_rigid_shape_to_fresh_current_pose(node):
    old=transform(99,(2.,3.),0.);old.child_frame_id='base_link'
    local=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
    node.footprint_msgs=[footprint(99,[transform_xy(p,old.transform) for p in local])]
    calls=[]
    def exact(target,source,when):
        calls.append((target,source,when.nanoseconds));return old
    node.tf_buffer.lookup_transform=exact
    result=node._start_footprint(100*SECOND,transform(100,(10.,20.),math.pi/2).transform)
    assert calls==[('map','base_link',99*SECOND)]
    assert all(any(math.dist(p,q)<1e-6 for q in [(11.,19.),(11.,21.),(9.,21.),(9.,19.)]) for p in result)


@pytest.mark.parametrize('cost', [0, 43])
@pytest.mark.parametrize('kind',['missing','stale','future','wrong_frame','too_few','nan','tf_failure','wrong_stamp','wrong_child'])
def test_unknown_start_missing_or_invalid_proof_fails_closed_without_blocking(node,kind,cost):
    unknown_start_node(node)
    node.cost_msg.data[5*10+5] = cost
    polygon=((9.,19.),(12.,19.),(12.,22.),(9.,22.))
    msg=footprint(99,polygon)
    if kind=='stale':msg=footprint(96.9,polygon)
    if kind=='future':msg=footprint(100.001,polygon)
    if kind=='wrong_frame':msg.header.frame_id='odom'
    if kind=='too_few':msg.polygon.points=msg.polygon.points[:2]
    if kind=='nan':msg.polygon.points[0].x=float('nan')
    node.footprint_msgs=[] if kind=='missing' else [msg]
    old_lookup=node.tf_buffer.lookup_transform;calls=[]
    def exact(target,source,when):
        if when.nanoseconds==0:return old_lookup(target,source,when)
        calls.append(when.nanoseconds)
        if kind=='tf_failure':raise tf2_ros.TransformException('not cached')
        t=transform(98 if kind=='wrong_stamp' else 99);t.child_frame_id='wrong' if kind=='wrong_child' else 'base_link';return t
    node.tf_buffer.lookup_transform=exact
    node._tick()
    assert not node.request_pub.records[-1]['request_id'] and not node.planner_calls
    assert node.state.total_attempts==0 and not node.busy_pub.records[-1]
    assert all(t==99*SECOND for t in calls)


@pytest.mark.parametrize('cost', [0, 43, 98])
def test_three_message_cache_uses_previous_exact_transform_when_newest_unavailable(node,cost):
    unknown_start_node(node)
    node.cost_msg.data[5*10+5] = cost
    local=((-1.,-1.),(2.,-1.),(2.,2.),(-1.,2.))
    for sec in [97.,98.,99.,100.]:
        old=transform(sec)
        node._footprint_cb(footprint(sec,[transform_xy(p,old.transform) for p in local]))
    assert len(node.footprint_msgs)==3
    original=node.tf_buffer.lookup_transform;calls=[]
    def exact(target,source,when):
        if when.nanoseconds==0:return original(target,source,when)
        calls.append(when.nanoseconds)
        if when.nanoseconds==100*SECOND:raise tf2_ros.TransformException('phase edge')
        t=transform(when.nanoseconds/SECOND);t.child_frame_id='base_link';return t
    node.tf_buffer.lookup_transform=exact
    node._tick()
    assert calls==[100*SECOND,99*SECOND]
    assert node.request_pub.records[-1]['request_id']
    assert len(node.planner_calls[-1][1]['start_footprint'])==4
    assert node.state.total_attempts==0


def test_clock_rollback_clears_cached_polygons(node):
    node.footprint_msgs=[footprint()]
    node.now_ns=99*SECOND;node._now()
    assert node.footprint_msgs==[]


def test_known_start_does_not_request_or_depend_on_footprint(node):
    def forbidden(*args):raise AssertionError('known start must retain old path')
    node._start_footprint=forbidden
    node._tick()
    assert node.request_pub.records[-1]['request_id']


def test_collinear_extra_vertex_is_rejected_like_cpp():
    with pytest.raises(ValueError):
        planner.convex_footprint(((0.,0.),(.5,0.),(1.,0.),(1.,1.),(0.,1.)))


def test_full_xyz_transform_retains_tilt_and_height(node):
    from tb3_locateanything.observation_node import transform_xyz
    old=transform(99,(2.,3.),0.);old.child_frame_id='base_link'
    old.transform.translation.z=.2
    old.transform.rotation.y=math.sin(.15);old.transform.rotation.w=math.cos(.15)
    current=transform(100,(10.,20.),0.).transform
    current.translation.z=.1
    current.rotation.x=math.sin(.1);current.rotation.w=math.cos(.1)
    local=((-1.,-1.,.5),(1.,-1.,.5),(1.,1.,.5),(-1.,1.,.5))
    world=[transform_xyz(p,old.transform) for p in local]
    msg=footprint(99,[p[:2] for p in world])
    for vertex,point in zip(msg.polygon.points,world):vertex.z=point[2]
    node.footprint_msgs=[msg]
    node.tf_buffer.lookup_transform=lambda *args:old
    got=node._start_footprint(100*SECOND,current)
    expected=[transform_xyz(p,current)[:2] for p in local]
    assert all(any(math.dist(p,q)<1e-6 for q in expected) for p in got)
    for point in local:
        assert transform_xyz(transform_xyz(point,old.transform),old.transform,True)==pytest.approx(point)


def test_duplicate_and_out_of_order_footprints_do_not_evict_newest_proof(node):
    node.footprint_msgs=[]
    for sec in [97.,99.,98.,99.,96.]:node._footprint_cb(footprint(sec))
    assert [m.header.stamp.sec for m in node.footprint_msgs]==[97,98,99]


def test_bridge_without_any_known_free_exit_is_not_usable_coverage():
    g=grid([-1]*15);costs=grid()
    assert max(planner._reachable(g,costs,(.5,1.5),POLYGON)) == -1
    assert planner.choose_observation_candidate(g,costs,(.5,1.5),(3.,1.5),[],[],
                                               start_footprint=POLYGON) is None


def test_bridge_preserves_known_free_route_lengths_but_hides_unknown_cells():
    g=grid([-1,0,0],width=3,height=1);costs=grid(width=3,height=1)
    assert planner._reachable(g,costs,(.5,.5),((0.,0.),(1.,0.),(1.,1.),(0.,1.))) == [-1,1,2]


@pytest.mark.parametrize('scaled',[.5,2.])
def test_scaled_nonunit_tf_is_not_valid_footprint_proof(node,scaled):
    from tb3_locateanything.observation_node import transform_xyz
    t=transform().transform;t.rotation.z*=scaled;t.rotation.w*=scaled
    with pytest.raises(ValueError):transform_xyz((0.,0.,0.),t)


def test_footprint_not_containing_historical_robot_origin_is_rejected(node):
    old=transform(99,(2.,3.),0.);old.child_frame_id='base_link'
    node.footprint_msgs=[footprint(99,((10.,10.),(11.,10.),(11.,11.),(10.,11.)))]
    node.tf_buffer.lookup_transform=lambda *args:old
    with pytest.raises(ValueError,match='origin'):
        node._start_footprint(100*SECOND,transform().transform)


@pytest.mark.parametrize('exact_available',[True,False])
def test_unknown_start_future_latest_requires_nonblocking_exact_now_same_pose(node,exact_available):
    unknown_start_node(node)
    old=transform(99,(10.,20.),0.);old.child_frame_id='base_link'
    local=((-1.,-1.),(2.,-1.),(2.,2.),(-1.,2.))
    node.footprint_msgs=[footprint(99,[transform_xy(p,old.transform) for p in local])]
    original=node.tf_buffer.lookup_transform;calls=[]
    def exact(target,source,when):
        if target!='map' or source!='base_link':return original(target,source,when)
        calls.append(when.nanoseconds)
        if when.nanoseconds==99*SECOND:return old
        if when.nanoseconds==100*SECOND:
            if not exact_available:raise tf2_ros.TransformException('exact current not cached')
            t=transform(100,(10.1,20.1),0.)
        else:t=transform(100.004,(10.,20.),0.)
        t.child_frame_id='base_link';return t
    node.tf_buffer.lookup_transform=exact
    node._tick()
    assert node.state.total_attempts==0
    assert 100*SECOND in calls
    if exact_available:
        args,kwargs=node.planner_calls[-1]
        assert args[2]==pytest.approx((10.1,20.1))
        assert min(p[0] for p in kwargs['start_footprint'])==pytest.approx(9.1)
        assert node.request_pub.records[-1]['request_id']
    else:
        assert not node.planner_calls and not node.request_pub.records[-1]['request_id']


@pytest.mark.parametrize('seconds',[0.,99.249,100.001])
def test_bridge_current_transform_cannot_use_stale_or_future_exact_proof(node,seconds):
    def exact(*args):
        t=transform(seconds);t.child_frame_id='base_link';return t
    node.tf_buffer.lookup_transform=exact
    with pytest.raises(ValueError):node._bridge_robot_transform(100*SECOND)


def test_malformed_stamp_fields_do_not_alias_valid_footprint_time(node):
    from tb3_locateanything.observation_node import footprint_stamp_ns
    msg=footprint(99);msg.header.stamp.sec=98;msg.header.stamp.nanosec=1000000000
    assert footprint_stamp_ns(msg.header.stamp)==-1
    node.footprint_msgs=[msg]
    with pytest.raises(ValueError):node._start_footprint(100*SECOND,transform().transform)


def test_finite_vertices_with_overflowed_cross_products_are_rejected():
    with pytest.raises(ValueError):
        planner.convex_footprint(((1e308,0.),(0.,1e308),(-1e308,0.),(0.,-1e308)))
