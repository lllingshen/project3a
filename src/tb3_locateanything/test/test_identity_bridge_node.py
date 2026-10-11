"""Actual node handoff/callback with fake sensors and no ROS spinning."""
from types import SimpleNamespace as NS
import copy
import pytest
from tb3_locateanything.locateanything_node import LocateAnythingNode
from tb3_locateanything.geometry import CameraCalibration, Vector3
from test_identity_bridge import fragments, whole, expose, points, LEFT, RIGHT, WHOLE, S


def sensor_payload(view):
    view.payload.update(image=NS(width=640,height=480,header=NS(
        stamp=NS(sec=view.stamp_ns//S,nanosec=view.stamp_ns%S),frame_id='optical')),
        depth_m=object(),camera_to_odom=object(),calibration=CameraCalibration(640,480,500.,500.,320.,240.,'optical'))
    view.payload['metadata']['localization_source']='aligned_depth'


def harness(memory):
    node=NS(memory=memory,epoch=0,p={'localization_mode':'depth'})
    node._identity_depth_proof=lambda *args: LocateAnythingNode._identity_depth_proof(node,*args)
    return node


@pytest.mark.parametrize('bad', [None,'image_stamp','depth_stamp','camera_frame','cache_box','cached_group','missing_depth','laser'])
def test_actual_node_proof_requires_exact_retained_sensor_cache(bad):
    m,source,a,b=fragments();view,proof=whole(m,2,1.);sensor_payload(view);node=harness(m)
    geometry=copy.deepcopy(view.payload['depth_geometry'][tuple(WHOLE)][1])
    if bad=='image_stamp': view.payload['image'].header.stamp.nanosec+=1
    elif bad=='depth_stamp': view.payload['metadata']['depth_stamp_ns']-=5_000_001
    elif bad=='camera_frame': view.payload['image'].header.frame_id='other'
    elif bad=='cache_box': view.payload['depth_geometry'].clear()
    elif bad=='cached_group': geometry['surface_group']['odom_xyz'][0][0]+=1
    elif bad=='missing_depth': del view.payload['depth_m']
    elif bad=='laser': node.p['localization_mode']='laser'
    evidence=node._identity_depth_proof(view,WHOLE,geometry)
    valid=m._identity_depth(view,WHOLE,evidence)
    assert bool(valid)==(bad is None)


def test_actual_catalog_callback_stores_then_resolves_only_actual_boxes():
    m,source,a,b=fragments();node=harness(m)
    node.events=[];node.now_ns=2*S
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=node.now_ns))
    node._refresh_positions=lambda:None
    node._record_class_hints=lambda *args:[]
    node._record_depth_hint=lambda *args:None
    node._status=lambda state,**kwargs:node.events.append(dict(state=state,**kwargs))
    node._localize=lambda view,box:(Vector3(.3,0.,1.),copy.deepcopy(view.payload['depth_geometry'][tuple(box)][1]))
    for stamp,position in [(2,1.),(3,2.)]:
        view,proof=whole(m,stamp,position);sensor_payload(view);node.now_ns=stamp*S
        # Native normalized box converts exactly to the already cached image box.
        response={'request_id':str(stamp),'answer':'<ref>chair</ref><box><15.625><20.833333333333332><93.75><62.5></box>',
                  'inference_seconds':.01,'generation_mode':'slow'}
        # Grounding syntax uses integer tokens; use a 1000x1000 calibration for exact native coordinates.
        view.payload['image'].width=view.payload['image'].height=1000
        view.payload['calibration']=CameraCalibration(1000,1000,500.,500.,500.,500.,'optical')
        response['answer']='<ref>chair</ref><box><10><10><60><30></box>'
        LocateAnythingNode._catalog_result(node,dict(request_id=str(stamp),view=view,label=None,categories=['chair']),response)
    assert len(m.landmarks)==1 and m.landmarks['chair_0'].label_votes=={'chair':3,'storage_rack':1}
    assert node.events[0]['object_ids']==[] and node.events[0]['identity_bridge_observations']
    assert node.events[1]['object_ids']==['chair_0']
