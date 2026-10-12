"""Spawn-only geometry validation; no Gazebo/ROS processes or motion commands."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src/tb3_bringup/models"
WORLDS = ROOT / "src/tb3_bringup/worlds"
spec = importlib.util.spec_from_file_location("test_safe_spawn_helper", ROOT / "src/tb3_bringup/launch/safe_spawn.py")
spawn = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spawn
spec.loader.exec_module(spawn)
ROBOT = Path("/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf")


@pytest.mark.parametrize("world,expected_bounds", [
    ("warehouse_models_person", (-3,3,-3,3)),
    ("warehouse_semantic_models", (-2,2,-3,3)),
    ("locateanything_chairs", (-3,3,-2.5,2.5)),
    ("locateanything_warehouse_objects", (-3,3,-3,3)),
    ("locateanything_warehouse_colors", (-3,3,-4,4)),
])
def test_five_actual_worlds_200_seeds_each(world, expected_bounds):
    scene = spawn.load_scene(WORLDS / (world + ".world"), [MODELS])
    assert scene.bounds == pytest.approx(expected_bounds)
    radius = spawn.robot_radius(ROBOT, [MODELS])
    assert radius == pytest.approx(math.hypot(.064+.265/2, .265/2))
    poses = set()
    for seed in range(200):
        result = spawn.choose_spawn(scene, radius, seed=seed)
        x,y = result["x"],result["y"]; required = radius+.15
        xmin,xmax,ymin,ymax = expected_bounds
        assert xmin+required <= x <= xmax-required
        assert ymin+required <= y <= ymax-required
        for x0,x1,y0,y1 in scene.obstacles:
            nearest_x = min(x1, max(x0,x)); nearest_y = min(y1,max(y0,y))
            assert (nearest_x-x)**2+(nearest_y-y)**2 >= required**2-1e-12
        poses.add((x,y,result["yaw"]))
    assert len(poses) == 200


def test_same_seed_reproduces_complete_pose_and_fresh_seed_is_reported():
    scene = spawn.load_scene(WORLDS / "locateanything_warehouse_objects.world", [MODELS])
    assert spawn.choose_spawn(scene,.237,seed=498) == spawn.choose_spawn(scene,.237,seed=498)
    result = spawn.choose_spawn(scene,.237)
    assert spawn.choose_spawn(scene,.237,seed=result["seed"]) == result


@pytest.mark.parametrize("xy", [(0,0),(2.2,-2.5),(2.725,2.2),(3.1,0),(99,99)])
def test_fixed_spawn_rejects_center_rack_fork_wall_slot_and_outside(xy):
    scene = spawn.load_scene(WORLDS / "locateanything_warehouse_objects.world", [MODELS])
    with pytest.raises(spawn.SpawnError, match="clearance"):
        spawn.choose_spawn(scene,.237,x=xy[0],y=xy[1],yaw=0)


def test_fixed_spawn_preserves_requested_pose_but_validates_it():
    scene = spawn.load_scene(WORLDS / "locateanything_warehouse_objects.world", [MODELS])
    result = spawn.choose_spawn(scene,.237,x=-1.5,y=0,yaw=1.2,seed=498)
    assert (result["x"],result["y"],result["yaw"],result["mode"]) == (-1.5,0,1.2,"fixed")
    with pytest.raises(spawn.SpawnError, match="both"):
        spawn.choose_spawn(scene,.237,x=-1.5)
    with pytest.raises(spawn.SpawnError, match="Nonfinite"):
        spawn.choose_spawn(scene,.237,yaw=float("nan"))


def test_impossible_region_raises_without_any_fallback():
    scene = spawn.SpawnScene((-1,1,-1,1),((-1,1,-1,1),))
    with pytest.raises(spawn.SpawnError, match="No verified safe spawn.*no fallback"):
        spawn.choose_spawn(scene,.237,seed=498,max_attempts=25)


def test_pose_composition_covers_rotated_link_and_collision(tmp_path):
    world = tmp_path / "test.world"
    world.write_text('''<sdf version="1.6"><world name="test"><model name="box">
    <pose>1 2 0 0 0 1.5707963267948966</pose><link name="link"><pose>1 0 0 0 0 0</pose>
    <collision name="box"><pose>0 1 0 0 0 0</pose><geometry><box><size>2 4 1</size></box>
    </geometry></collision></link></model></world></sdf>''')
    scene = spawn.load_scene(world, [], (-5,5,-5,5))
    assert scene.obstacles[0] == pytest.approx((-2,2,2,4))


def test_include_pose_overrides_model_pose_and_applies_scale(tmp_path):
    directory = tmp_path / "thing"; directory.mkdir()
    (directory/"mesh.obj").write_text("v -1 -2 0\nv 1 2 1\n")
    (directory/"model.sdf").write_text('''<sdf version="1.6"><model name="thing">
    <pose>100 100 0 0 0 0</pose><link name="link"><collision name="c"><geometry><mesh>
    <uri>mesh.obj</uri><scale>2 3 1</scale></mesh></geometry></collision></link></model></sdf>''')
    world=tmp_path/"test.world"
    world.write_text('''<sdf version="1.6"><world name="test"><include><uri>model://thing</uri>
    <pose>1 2 0 0 0 0</pose></include></world></sdf>''')
    scene = spawn.load_scene(world,[tmp_path],(-10,10,-10,10))
    assert scene.obstacles[0] == pytest.approx((-1,3,-4,8))


def test_collada_scene_translation_scaling_and_units(tmp_path):
    mesh=tmp_path/"test.dae"
    mesh.write_text('''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
    <asset><unit meter="0.01"/><up_axis>Z_UP</up_axis></asset><library_geometries>
    <geometry id="g"><mesh><source id="p"><float_array>0 0 0 1 2 3</float_array>
    <technique_common><accessor count="2" stride="3"/></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices></mesh></geometry>
    </library_geometries><library_visual_scenes><visual_scene id="scene"><node>
    <translate>10 20 30</translate><scale>2 3 4</scale><instance_geometry url="#g"/>
    </node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/>
    </scene></COLLADA>''')
    low,high=spawn.mesh_bounds(mesh)
    assert low == pytest.approx([.1,.2,.3])
    assert high == pytest.approx([.12,.26,.42])


@pytest.mark.parametrize("relative,height", [
    ("person_standing/meshes/standing.dae",1.91709531),
    ("first_2015_trash_can/meshes/trash_can.dae",.73660498),
    ("la_warehouse_extinguisher/meshes/extinguisher.dae",.42843371),
])
def test_actual_collada_bounds_are_in_metres(relative,height):
    low,high=spawn.mesh_bounds(MODELS/relative)
    assert high[2]-low[2] == pytest.approx(height,abs=1e-6)
    assert np.isfinite(low).all() and np.isfinite(high).all()


def test_missing_mesh_unsupported_geometry_and_named_frames_fail_closed(tmp_path):
    world=tmp_path/"test.world"
    for body,match in [
        ('<geometry><mesh><uri>missing.obj</uri></mesh></geometry>',"resolved"),
        ('<geometry><heightmap/></geometry>',"Unsupported"),
        ('<pose relative_to="another">0 0 0 0 0 0</pose><geometry><box><size>1 1 1</size></box></geometry>',"Named-frame"),
        ('<pose degrees="1">0 0 0 0 0 90</pose><geometry><box><size>1 1 1</size></box></geometry>',"degree"),
    ]:
        world.write_text('<sdf version="1.6"><world name="w"><model name="m"><link name="l">'
                         '<collision name="c">'+body+'</collision></link></model></world></sdf>')
        with pytest.raises(spawn.SpawnError,match=match):
            spawn.load_scene(world,[],(-5,5,-5,5))


def test_open_world_requires_explicit_region(tmp_path):
    world=tmp_path/"open.world"
    world.write_text('<sdf version="1.6"><world name="open"/></sdf>')
    with pytest.raises(spawn.SpawnError,match="spawn_bounds"):
        spawn.load_scene(world,[MODELS])
    assert spawn.load_scene(world,[MODELS],(-3,3,-3,3)).bounds == (-3,3,-3,3)
