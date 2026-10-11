"""Measured-height soft framing and actual observer integration, no model/simulator."""
import copy
from dataclasses import replace
import math
import pytest
from std_msgs.msg import String
import json
from tb3_locateanything import observation_node as module
from tb3_locateanything.vertical_framing import VerticalFraming, _min_sinusoid
from test_observation_node import node, depth_hint, transform, camera_info, camera_transform


def model(**kwargs):
    values=dict(height=480,fy=530.,cy=239.5,mount_quaternion=(-.5,.5,-.5,.5),
                mount_translation=(.077,0.,.1),base_translation=(0.,0.,0.),base_quaternion=(0.,0.,0.,1.),
                xy_tolerance=.25,yaw_tolerance=.25)
    values.update(kwargs)
    return VerticalFraming(**values)


def test_measured_high_surface_needs_more_distance_but_does_not_claim_unseen_height():
    camera=model();support=[(2.,-.1,.3),(2.,.1,1.2)]
    near=camera.predict((0.,0.),0.,support);far=camera.predict((-2.,0.),0.,support)
    assert not near['nominal_fits'] and not near['arrival_tolerance_fits']
    assert far['arrival_tolerance_fits']
    assert 'unseen' in far['scope'] and 'height/roll/pitch' in far['assumption']


def test_asymmetric_vertical_principal_point_and_translation_uncertainty():
    support=[(2.,0.,.1),(2.,0.,.65)]
    centered=model(xy_tolerance=0.,yaw_tolerance=0.)
    raised=model(cy=80.,xy_tolerance=0.,yaw_tolerance=0.)
    assert centered.predict((0.,0.),0.,support)['arrival_tolerance_fits']
    assert not raised.predict((0.,0.),0.,support)['arrival_tolerance_fits']
    assert not model(xy_tolerance=3.).predict((0.,0.),0.,support)['arrival_tolerance_fits']


@pytest.mark.parametrize('angle',[-3.13,-1.6,-.01,0.,.7,3.13])
def test_exact_yaw_interval_minimum_matches_dense_independent_samples(angle):
    a,b=math.cos(angle),math.sin(angle);limit=.37
    exact=_min_sinusoid(a,b,limit)
    sampled=min(a*math.cos(-limit+2*limit*i/10000)+b*math.sin(-limit+2*limit*i/10000) for i in range(10001))
    assert exact<=sampled+1e-12 and sampled-exact<1e-8


@pytest.mark.parametrize('kwargs',[{'fy':0.},{'cy':float('nan')},{'mount_quaternion':(0.,0.,0.,2.)},{'base_translation':(0.,0.,float('nan'))}])
def test_invalid_calibration_cannot_manufacture_vertical_proof(kwargs):
    with pytest.raises(ValueError):model(**kwargs)


def test_hint_geometry_preserves_full_xyz_rotation_without_mutation():
    hint=depth_hint();before=copy.deepcopy(hint);tf=transform().transform
    tf.rotation.x=0.;tf.rotation.y=math.sin(.2);tf.rotation.z=0.;tf.rotation.w=math.cos(.2)
    tf.translation.z=.3
    result=module.hint_geometry(hint,tf,include_xyz=True)
    assert len(result)==6
    for original,group,xyz in zip(hint['surface_groups'],result[3],result[5]):
        for source,xy,point in zip(original['odom_xyz'],group,xyz):
            expected=module.transform_xyz(source,tf)
            assert point==pytest.approx(expected) and xy==pytest.approx(expected[:2])
    assert hint==before
    assert len(module.hint_geometry(hint,tf))==5


def test_real_camera_model_carries_normalized_actual_mount_and_intrinsics(node):
    optical=camera_transform().transform
    for a in ('x','y','z','w'):setattr(optical.rotation,a,getattr(optical.rotation,a)*2.)
    camera=module.camera_framing(camera_info(),optical,(.25,.25))
    assert camera.vertical_calibration[:3]==(480,530.,239.5)
    assert camera.vertical_calibration[3]==pytest.approx((-.5,.5,-.5,.5))
    assert node._camera_model(node.now_ns).vertical_calibration


def test_actual_tick_passes_measured_xyz_and_prediction_to_dispatch_without_memory_vote(node,monkeypatch):
    node.memory['pending_landmarks']=[];node.memory['observation_hints']=[depth_hint()]
    old=module.choose_observation_candidate
    prediction={'arrival_tolerance_fits':True,'support_count':2,'scope':'measured only'}
    def planner(*args,**kwargs):
        assert isinstance(kwargs['vertical_framing'],VerticalFraming)
        assert len(kwargs['hypothesis_xyz_groups'])==2
        assert kwargs['hypothesis_xyz_groups'][0][0][2]==.5
        return {**old(*args,**kwargs),'vertical_framing':prediction}
    monkeypatch.setattr(module,'choose_observation_candidate',planner)
    before=copy.deepcopy(node.memory);logs=[];node.get_logger=lambda:type('Logger',(),{'info':lambda _,msg:logs.append(json.loads(msg))})()
    node._tick();request=node.request_pub.records[-1]
    assert node.state.records[request['request_id']]['vertical_framing']==prediction
    node._event_cb(String(data=json.dumps({**request,'event':'dispatched'})))
    assert any(x.get('vertical_framing')==prediction for x in logs)
    assert node.memory==before and node.state.total_attempts==1


def test_optional_vertical_failure_keeps_existing_hint_pose_and_budget(node,monkeypatch):
    node.memory['pending_landmarks']=[];node.memory['observation_hints']=[depth_hint()]
    def invalid(*a,**kw):raise ValueError('optional vertical proof unavailable')
    monkeypatch.setattr(module,'VerticalFraming',invalid)
    node._tick();request=node.request_pub.records[-1]
    assert request['request_id'] and node.state.total_attempts==0
    assert node.planner_calls[-1][1]['vertical_framing'] is None
    assert node.state.records[request['request_id']]['vertical_framing']['available'] is False


def test_pending_xy_track_does_not_invent_height(node):
    node._tick();kwargs=node.planner_calls[-1][1]
    assert 'vertical_framing' not in kwargs and 'hypothesis_xyz_groups' not in kwargs


# These isolate selection policy from separately tested projection math.
def selection(vertical=None, xyz=((0.,-.05,.1),(0.,.05,1.2)), **kwargs):
    from tb3_locateanything import observation_planner as planner
    grid=planner.Grid(100,100,.1,-5.,-5.,0.,[0]*10000)
    camera=planner.CameraFraming(640,530.,319.5,0.,0.,0.,.25,.25)
    return planner.choose_observation_candidate(grid,grid,(-3.,0.),(0.,0.),[(-4.,3.)],[],
        support_xy=[(0.,-.05),(0.,.05)],camera=camera,vertical_framing=vertical,support_xyz=xyz,**kwargs)


class VerticalPreferenceOracle:
    def __init__(self, full):self.full=full;self.calls=[]
    def predict(self, point, yaw, support):
        self.calls.append((point,yaw,support))
        return {'arrival_tolerance_fits':self.full(point)}


def test_full_clear_fallback_preferred_over_partial_nominal_without_extra_samples(monkeypatch):
    from tb3_locateanything import observation_planner as planner
    monkeypatch.setattr(planner,'_fallback_points',lambda *args:[(-3.,.05)])
    oracle=VerticalPreferenceOracle(lambda point:point[0]<-2.5)
    result=selection(oracle)
    assert result['pose'][:2]==(-3.,.05) and result['visibility_reason']==planner.CLEAR_VISIBILITY
    assert len(oracle.calls)<=160


def test_all_partial_retains_original_nominal_phase_even_if_fallback_is_nearer(monkeypatch):
    from tb3_locateanything import observation_planner as planner
    original=selection()
    monkeypatch.setattr(planner,'_fallback_points',lambda *args:[(-3.,.05)])
    assert selection(VerticalPreferenceOracle(lambda point:False))['pose']==original['pose']


def test_full_uncertain_never_displaces_partial_clear(monkeypatch):
    from tb3_locateanything import observation_planner as planner
    original=selection()
    monkeypatch.setattr(planner,'_fallback_points',lambda *args:[(-3.,.05)])
    monkeypatch.setattr(planner,'_visible',lambda grid,start,end:start[0]>-2.5)
    result=selection(VerticalPreferenceOracle(lambda point:point[0]<-2.5),allow_uncertain_visibility=True)
    assert result['pose']==original['pose'] and result['visibility_reason']==planner.CLEAR_VISIBILITY


def test_missing_xyz_keeps_original_pose_without_calling_height_predictor():
    oracle=VerticalPreferenceOracle(lambda point:True)
    assert selection(oracle,xyz=())==selection(None,xyz=())
    assert not oracle.calls


def test_finite_inputs_with_overflowing_derived_margin_cannot_break_json():
    with pytest.raises(ValueError,match='nonfinite'):
        model().predict((0.,0.),0.,[(1.7e308,1.7e308,1.7e308)])


def test_optional_prediction_error_retains_original_partial_choice():
    class Unavailable:
        def predict(self,*args):raise ValueError('optional numerical proof unavailable')
    result=selection(Unavailable())
    assert result['pose']==selection()['pose']
    assert result['vertical_framing']['available'] is False
    json.dumps(result,allow_nan=False)
