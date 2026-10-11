import copy
import pytest
from tb3_locateanything.observation_novelty import ranked_hint_groups
from test_observation_node import node, depth_hint, module, event, SECOND


def group(x):return {'odom_xyz':[[x,0.,.5],[x+.01,.01,.5],[x-.01,.02,.5]]}

def task(*xs):return {'surface_groups':[group(x) for x in xs]}

def known(x=0.,**kwargs):return {'confirmed':True,'conflicted':False,'support_points':[[x,0.],[x+.01,.01],[x-.01,.02]],**kwargs}


def test_novelty_is_global_group_order_not_task_best_then_wrong_focus():
 tasks={'a_old':task(0.,3.),'z_new':task(5.)};before=copy.deepcopy(tasks)
 ranked=ranked_hint_groups(tasks,[known()],{}, {})
 assert [(r[0],r[1]) for r in ranked]==[('a_old',1),('z_new',0),('a_old',0)]
 assert [r[2] for r in ranked]==[1.,1.,0.]
 assert tasks==before


@pytest.mark.parametrize('status',[{'confirmed':False},{'conflicted':True}])
def test_only_confirmed_unconflicted_support_can_reduce_novelty(status):
 assert ranked_hint_groups({'x':task(0.)},[known(**status)],{}, {})[0][2]==1.


def test_untried_tier_and_equal_novelty_attempt_fairness():
 tasks={'a':task(0.,3.),'b':task(5.)}
 result=ranked_hint_groups(tasks,[known()],{'a':1},{'a':[1]})
 assert [(r[0],r[1]) for r in result]==[('b',0),('a',0),('a',1)]


def test_wholly_known_group_is_deprioritized_not_discarded():
 assert ranked_hint_groups({'x':task(0.)},[known()],{}, {})==[('x',0,0.,False)]

from types import SimpleNamespace
from std_msgs.msg import String, Bool
from test_observation_node import lookup, grid, camera_info
from test_observation_planner import focus_scene
from tb3_locateanything import observation_planner as planner
from tb3_locateanything.observation_novelty import SearchProgress


def hints_for(node, count=3, groups=2):
    hints=[]
    for i in range(count):
        h=depth_hint();h['hint_id']=f'hint_{i:02}'
        h['surface_groups']=(h['surface_groups']*4)[:groups]
        hints.append(h)
    node.memory['pending_landmarks']=[]
    node.memory['observation_hints']=hints
    node.memory['landmarks']=[]
    return hints


def result(kwargs, reason=None):
    return dict(pose=(11.,22.,.4),visibility_reason=reason or module.CLEAR_VISIBILITY,
                map_sightline_groups=[reason != module.UNKNOWN_VISIBILITY],
                focus_group_index=kwargs.get('forced_focus_index'),hypothesis_group_count=len(kwargs.get('hypothesis_groups',[])))


@pytest.mark.parametrize('bad',[None,'bad',[[None,2]],[[float('nan'),0]]])
def test_malformed_known_support_cannot_abort_or_reduce_novelty(bad):
    assert ranked_hint_groups({'x':task(0.)},[known(support_points=bad)],{}, {})[0][2]==1.


@pytest.mark.parametrize('index',[-1,2,True,1.0,'0'])
def test_forced_focus_must_be_valid_integer_and_opt_in(index):
    args,kwargs=focus_scene()
    assert planner.choose_observation_candidate(*args,**kwargs,focus_hypotheses=True,forced_focus_index=index) is None
    assert planner.choose_observation_candidate(*args,**kwargs,forced_focus_index=0) is None


def test_forced_focus_keeps_full_hypotheses_but_searches_only_requested_group(monkeypatch):
    args,kwargs=focus_scene();original=copy.deepcopy(kwargs['hypothesis_groups']);calls=[]
    original_select=planner.choose_observation_candidate
    def select(*a,**k):
        calls.append(a[3]);return dict(pose=(1.,2.,0.),visibility_reason=planner.CLEAR_VISIBILITY,map_sightline_groups=[True])
    monkeypatch.setattr(planner,'choose_observation_candidate',select)
    selected=original_select(*args,**kwargs,focus_hypotheses=True,forced_focus_index=1)
    assert calls==[(2.,0.)] and selected['focus_group_index']==1
    assert selected['hypothesis_group_count']==2 and kwargs['hypothesis_groups']==original


def test_reachability_cache_same_grid_robot_only_not_visibility(monkeypatch):
    args,kwargs=focus_scene();cache={};calls=[];original=planner._reachable
    def reachable(*a):calls.append(a[2]);return original(*a)
    monkeypatch.setattr(planner,'_reachable',reachable)
    first=planner.choose_observation_candidate(*args,**kwargs,focus_hypotheses=True,forced_focus_index=0,_reachability_cache=cache)
    second=planner.choose_observation_candidate(*args,**kwargs,focus_hypotheses=True,forced_focus_index=1,_reachability_cache=cache)
    assert len(calls)==1 and first['focus_group_index']==0 and second['focus_group_index']==1
    changed=list(args);changed[2]=(-2.4,-2.5)
    planner.choose_observation_candidate(*changed,**kwargs,focus_hypotheses=True,forced_focus_index=0,_reachability_cache=cache)
    assert len(calls)==2


def test_actual_tick_chooses_high_novelty_group_not_known_group_in_same_hint(node,monkeypatch):
    h=hints_for(node,2)
    h[0]['surface_groups']=[group(0.),group(3.)]
    h[1]['surface_groups']=[group(5.),group(0.)]
    node.memory['landmarks']=[known()]
    calls=[]
    def select(*a,**k):calls.append(k);return result(k)
    monkeypatch.setattr(module,'choose_observation_candidate',select)
    node._tick()
    req=node.request_pub.records[-1]
    assert req['track_id']=='hint_00' and calls[0]['forced_focus_index']==1
    assert len(calls[0]['hypothesis_groups'])==2
    assert node.state.records[req['request_id']]['focus_novel_fraction']==1.
    assert node.state.total_attempts==0 and node.state.focus_dispatched=={}


def test_continuation_reaches_tail_across_map_refresh_and_finishes_without_attempts(node,monkeypatch):
    hints_for(node,3);calls=[]
    monkeypatch.setattr(module,'choose_observation_candidate',lambda *a,**k:calls.append(k['forced_focus_index']))
    node._tick();assert len(calls)==4 and node.busy_pub.records[-1]
    node.now_ns+=SECOND;node.map_msg=grid(101);node.cost_msg=grid(101);node.camera_info=camera_info(101);node.tf_buffer.lookup_transform=lookup(101)
    node._tick()
    assert len(calls)==6 and not node.busy_pub.records[-1]
    assert node.state.total_attempts==0 and node.state.proposal is None


def test_stale_first_candidate_advances_identity_and_cannot_hold_busy_forever(node,monkeypatch):
    hints_for(node,1);wall=[0.];calls=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:wall[0])
    node.tf_buffer.lookup_transform=lookup(99.3)
    def select(*a,**k):calls.append(k['forced_focus_index']);wall[0]+=.1;return result(k)
    monkeypatch.setattr(module,'choose_observation_candidate',select)
    node._tick();assert node.busy_pub.records[-1] and node.state.proposal is None
    node._tick();assert not node.busy_pub.records[-1] and node.state.proposal is None
    assert calls==[0,1] and node.state.total_attempts==0


def test_long_single_group_drops_pose_and_advances_to_fresh_tail(node,monkeypatch):
    hints_for(node,1);wall=[0.];calls=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:wall[0])
    def select(*a,**k):
        calls.append(k['forced_focus_index'])
        wall[0]+=.6 if len(calls)==1 else .01
        return result(k)
    monkeypatch.setattr(module,'choose_observation_candidate',select)
    node._tick();assert node.state.proposal is None
    node._tick();assert node.state.proposal is not None and calls==[0,1]


def test_uncertain_identity_revalidated_only_after_clear_tier_exhausted(node,monkeypatch):
    hints_for(node,3);calls=[]
    def select(*a,**k):
        calls.append(k['forced_focus_index'])
        return result(k,module.UNKNOWN_VISIBILITY) if len(calls) in (1,7) else None
    monkeypatch.setattr(module,'choose_observation_candidate',select)
    node._tick();assert node.state.proposal is None
    node._tick();assert len(calls)==7 and node.state.proposal
    assert node.state.records[node.state.proposal]['visibility_reason']==module.UNKNOWN_VISIBILITY
    assert node.state.total_attempts==0


@pytest.mark.parametrize('interrupt',['command','disable','epoch'])
def test_cursor_clears_across_user_pause_and_memory_epoch(node,monkeypatch,interrupt):
    hints_for(node,3)
    monkeypatch.setattr(module,'choose_observation_candidate',lambda *a,**k:None)
    node._tick();assert len(node._search_progress.seen)==4
    if interrupt=='command':node._command_cb(String(data='go to the object'))
    if interrupt=='disable':node._enabled_cb(Bool(data=False))
    if interrupt=='epoch':
        node.memory['epoch']='different'
        # Same IDs must be re-evaluated in a new Memory epoch.
        node._tick();assert len(node._search_progress.seen)==4
    else:
        assert not node._search_progress.seen
        node._enabled_cb(Bool(data=True));node._tick();assert len(node._search_progress.seen)==4


def test_clock_epoch_resets_cursor_without_reusing_pose():
    progress=SearchProgress();rows=[('a',0,1.,False),('a',1,0.,False)]
    progress.begin(rows,(0,0,0));progress.tested(rows[0],True)
    progress.begin(rows,(1,0,0));assert progress.next(rows)==(rows[0],False)


def test_spatial_index_matches_exact_distance_fraction_including_gate_boundary():
    import numpy as np
    rng=np.random.default_rng(498)
    supports=np.vstack([rng.uniform(-3.,3.,(512,2)),[[0.,0.]]])
    points=np.vstack([rng.uniform(-3.,3.,(61,2)),[[.18,0.],[.18000001,0.],[0.,0.]]])
    h={'surface_groups':[{'odom_xyz':np.column_stack([points,np.ones(len(points))]).tolist()}]}
    measured=ranked_hint_groups({'x':h},[known(support_points=supports.tolist())],{}, {})[0][2]
    exact=float((np.linalg.norm(points[:,None,:]-supports[None,:,:],axis=2).min(axis=1)>.18).mean())
    assert measured==exact
