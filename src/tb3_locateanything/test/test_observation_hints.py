"""Ambiguity tasks preserve evidence without manufacturing object identities."""
import copy
import json

import pytest

from tb3_locateanything.depth_core import AmbiguousDepthError
from tb3_locateanything.geometry import Quaternion, Transform3D, Vector3
from tb3_locateanything.observation_hints import HintMemory
from tb3_locateanything.visual_memory import Landmark, Memory
from test_visual_memory_node import node, stored_view, SECOND


BOX = [10., 10., 100., 90.]


def surface(x, y=0.):
    return {'odom_xyz': [[x, y, 1.], [x+.02, y, 1.], [x, y+.02, 1.]],
            'pixels': [[20, 20], [21, 20], [20, 21]], 'count': 3}


def add(memory, *, groups=None, label='storage rack', view='source', stamp=10*SECOND,
        landmarks=(), pose=(0., 0., 0.), camera=(.1, 0., 0.)):
    return memory.add(label, view, stamp, pose, camera, BOX,
                      groups if groups is not None else [surface(2.), surface(4.)], landmarks)


def track(oid='candidate_0', x=2., *, confirmed=False, label='storage_rack',
          views=('independent',), conflicted=False):
    return Landmark(oid, label, x, 0., 1, 11*SECOND, view_ids=set(views),
                    support_points=[(x, 0.), (x+.02, 0.), (x, .02)],
                    confirmed=confirmed, conflicted=conflicted)


def test_same_unordered_surface_set_keeps_stable_id_and_original_viewpoint():
    hints=HintMemory()
    first=add(hints)
    again=add(hints, groups=[surface(4.03), surface(2.03)], view='another',
              stamp=12*SECOND, pose=(1., 0., 1.))
    assert first == again == 'depth_hint_0_0'
    item=hints.serialize(12*SECOND)[0]
    assert item['source_view_id']=='source' and item['source_pose']==[0.,0.,0.]
    assert item['stamp_ns']==10*SECOND and item['last_seen_ns']==12*SECOND


def test_surface_set_matching_requires_one_to_one_assignment():
    # Every row/column has a neighbour, but the first two new groups compete
    # for only the first old group (Hall counterexample). These are not the
    # same ambiguity task and must not share an attempt budget/identity.
    hints=HintMemory()
    first=add(hints,groups=[surface(0.),surface(1.),surface(1.4)])
    second=add(hints,groups=[surface(-.15),surface(.15),surface(1.2)],view='another',stamp=11*SECOND)
    assert second != first
    assert len(hints.hints)==2


def test_capacity_and_expiry_do_not_recreate_task_or_reset_external_budget():
    hints=HintMemory(max_hints=1,max_age_sec=5.)
    first=add(hints)
    assert add(hints,groups=[surface(6.),surface(8.)],view='other') is None
    assert hints.rejected_capacity==1
    assert hints.serialize(16*SECOND)[0]['status']=='expired'
    assert add(hints,view='later',stamp=17*SECOND)==first
    assert hints.serialize(17*SECOND)[0]['status']=='expired'
    assert add(hints,groups=[surface(6.),surface(8.)],view='other',stamp=18*SECOND) is None
    assert len(hints.hints)==1


def test_clock_epoch_clear_removes_old_surfaces_and_never_reuses_full_id():
    hints=HintMemory(epoch=3)
    first=add(hints,landmarks=[track('storage_rack_0',4.,confirmed=True)])
    hints.rejected_capacity=1
    hints.clear(4)
    assert hints.hints==[] and hints.known_surfaces=={} and hints.rejected_capacity==0
    second=add(hints,stamp=SECOND)
    assert first=='depth_hint_3_0' and second=='depth_hint_4_0'


@pytest.mark.parametrize('kwargs', [{'max_hints':0}, {'max_hints':-1}, {'max_hints':1.5},
                                    {'max_hints':True}, {'max_age_sec':0.},
                                    {'max_age_sec':-1.}, {'max_age_sec':float('nan')}])
def test_invalid_capacity_or_lifetime_is_not_silently_coerced(kwargs):
    with pytest.raises(ValueError):
        HintMemory(**kwargs)


def test_reconcile_cannot_revive_expired_hint_before_any_serialize_call():
    hints=HintMemory(max_age_sec=5.)
    add(hints)
    hints.reconcile([track()],'independent',(1.,0.,0.),16*SECOND)
    assert hints.hints[0]['status']=='expired' and hints.hints[0]['resolved_by']==[]


@pytest.mark.parametrize('change', ['one_group','nine_groups','nonfinite_xyz','wrong_dimension',
                                    'overfull_group','pixel_count','nonfinite_pixel','small_count'])
def test_corrupt_or_incomplete_surface_groups_are_rejected(change):
    groups=[surface(2.),surface(4.)]
    if change=='one_group':groups=groups[:1]
    elif change=='nine_groups':groups=[surface(float(i)) for i in range(9)]
    elif change=='nonfinite_xyz':groups[0]['odom_xyz'][0][0]=float('nan')
    elif change=='wrong_dimension':groups[0]['odom_xyz'][0]=[2.,0.]
    elif change=='overfull_group':
        groups[0]['odom_xyz']*=22;groups[0]['pixels']*=22;groups[0]['count']=66
    elif change=='pixel_count':groups[0]['pixels'].pop()
    elif change=='nonfinite_pixel':groups[0]['pixels'][0][0]=float('inf')
    elif change=='small_count':groups[0]['count']=2
    hints=HintMemory()
    with pytest.raises((ValueError,TypeError)):
        add(hints,groups=groups)
    assert hints.hints==[]


def test_serialized_and_original_arrays_cannot_mutate_retained_hint():
    hints=HintMemory();groups=[surface(2.),surface(4.)]
    add(hints,groups=groups)
    groups[0]['odom_xyz'][0][0]=999.
    output=hints.serialize(10*SECOND)
    output[0]['surface_groups'][0]['odom_xyz'][0][0]=888.
    assert hints.serialize(10*SECOND)[0]['surface_groups'][0]['odom_xyz'][0][0]==2.
    json.dumps(hints.serialize(10*SECOND),allow_nan=False)


@pytest.mark.parametrize('view,pose,stamp', [('source',(1.,0.,0.),11*SECOND),
                                          ('independent',(0.,0.,3.),11*SECOND),
                                          ('independent',(.249,0.,0.),11*SECOND),
                                          ('independent',(1.,0.,0.),10*SECOND)])
def test_same_view_pure_rotation_close_pose_or_old_stamp_cannot_resolve(view,pose,stamp):
    hints=HintMemory();add(hints)
    obj=track(views=(view,))
    hints.reconcile([obj],view,pose,stamp)
    assert hints.hints[0]['status']=='unresolved'


def test_new_independent_pending_track_handoffs_without_class_confirmation():
    hints=HintMemory();add(hints)
    obj=track()
    before=copy.deepcopy(vars(obj))
    hints.reconcile([obj],'independent',(.25,0.,0.),11*SECOND)
    assert hints.hints[0]['status']=='resolved'
    assert hints.hints[0]['resolved_by']==['candidate_0']
    assert vars(obj)==before and not obj.confirmed and not obj.label_votes


def test_confirmed_background_support_and_known_id_cannot_resolve():
    hints=HintMemory();background=track('storage_rack_0',4.,confirmed=True,views=('source',))
    add(hints,landmarks=[background])
    # New ID at the already known measured background must still be excluded.
    hints.reconcile([track('candidate_5',4.)],'independent',(1.,0.,0.),11*SECOND)
    assert hints.hints[0]['status']=='unresolved'
    # Existing ID remains disallowed even if subsequent supports move to the
    # other plausible surface. Otherwise a historical background can win.
    background.support_points=track(x=2.).support_points
    background.view_ids.add('independent')
    hints.reconcile([background],'independent',(1.,0.,0.),12*SECOND)
    assert hints.hints[0]['status']=='unresolved'


@pytest.mark.parametrize('fault', ['two_tracks','two_groups','conflict','class','missing_view'])
def test_multiple_or_invalid_matches_do_not_resolve(fault):
    hints=HintMemory();add(hints)
    obj=track();objects=[obj]
    if fault=='two_tracks':objects.append(track('candidate_1',4.))
    elif fault=='two_groups':obj.support_points.extend(track(x=4.).support_points)
    elif fault=='conflict':obj.conflicted=True
    elif fault=='class':obj.label='chair'
    elif fault=='missing_view':obj.view_ids={'other'}
    hints.reconcile(objects,'independent',(1.,0.,0.),11*SECOND)
    assert hints.hints[0]['status']=='unresolved'


def test_ordinary_two_dimensional_memory_supports_resolve_without_shape_error():
    memory=Memory(confirm_classes=True)
    view=memory.add_view({},11*SECOND,1.,0.,0.)
    obj=memory.remember(view,'storage rack',[10,10,100,90],2.,0.,support_points=[(2.,0.),(2.02,0.),(2.,.02)])
    assert all(len(p)==2 for p in obj.support_points)
    before=copy.deepcopy(memory.serialize())
    hints=HintMemory();add(hints)
    hints.reconcile(list(memory.landmarks.values()),view.view_id,(1.,0.,0.),11*SECOND)
    assert hints.hints[0]['resolved_by']==[obj.object_id]
    assert memory.serialize()==before


def test_catalog_typed_depth_rejection_adds_hint_but_no_memory_record_or_vote(node):
    node.memory=Memory(confirm_classes=True)
    view=stored_view(node,seconds=99.)
    view.payload['camera_to_odom']=Transform3D(Vector3(.1,0.,1.5),Quaternion(-.5,.5,-.5,.5))
    groups=[surface(2.),surface(4.)]
    for group in groups:group['pixels']=[[120,120],[121,120],[120,121]]
    error=AmbiguousDepthError([100,100,250,400],groups,2)
    def reject(*args):raise error
    node._localize=reject
    before=copy.deepcopy(node.memory.serialize())
    node._catalog_result({'view':view,'label':'storage rack','request_id':'ambiguous'},
                         {'request_id':'ambiguous','answer':'<box><156><208><390><833></box>',
                          'inference_seconds':.2})
    assert node.memory.serialize()==before
    assert not view.observations and not node.memory.landmarks
    assert len(node.observation_hints.hints)==1
    assert node.observation_hints.hints[0]['source_camera']==pytest.approx([.1,0.,0.])
    assert node.events[-1]['state']=='memory_updated'
    assert node.events[-1]['object_ids']==[] and len(node.events[-1]['depth_observation_hints'])==1
    assert not node.result_pub.messages


def test_node_clock_rollback_clears_hints_in_same_epoch_as_memory(node):
    node.observation_hints=HintMemory(epoch=node.epoch)
    old=add(node.observation_hints)
    node.now_ns=SECOND
    node.worker_ready=False
    node._tick()
    assert node.epoch==1 and node.observation_hints.epoch==1
    assert node.observation_hints.hints==[] and node.memory.landmarks=={}
    assert add(node.observation_hints,stamp=SECOND)!=old
