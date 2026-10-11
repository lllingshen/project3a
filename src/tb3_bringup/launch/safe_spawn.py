"""Conservative, simulator-only spawn geometry; never used by perception.

Read local SDF collision geometry, including OBJ/COLLADA mesh bounds. Each
model's transformed collision envelope excludes its whole footprint, even gaps
under furniture. Unknown geometry/frame constructs fail closed. Random seeds
and the selected pose are returned so a launch can be reproduced exactly.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
import random
import secrets
import xml.etree.ElementTree as ET

import numpy as np


class SpawnError(ValueError):
    """The requested spawn cannot be verified safe from local geometry."""


def numbers(text, count):
    values = np.asarray([float(v) for v in (text or "").split()])
    if len(values) != count or not np.isfinite(values).all():
        raise SpawnError(f"Expected {count} finite geometry values: {text!r}")
    return values


def pose(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    if (element.get("relative_to") or element.get("frame") or
            element.get("degrees", "false").lower() not in ("false", "0")):
        raise SpawnError("Named-frame/degree SDF poses are unsupported; spawn not verified")
    x, y, z, roll, pitch, yaw = numbers(element.text, 6)
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def corners(low, high):
    return np.asarray(list(product(*zip(low, high))), dtype=float)


def transformed(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


@lru_cache(maxsize=32)
def mesh_bounds(path):
    """Mesh-space SI bounds, with COLLADA scene transforms applied."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        points = [numbers(" ".join(fields[1:]), 3)
                  for line in path.read_text().splitlines()
                  if (fields := line.split()) and fields[0] == "v"]
    elif path.suffix.lower() == ".dae":
        root = ET.parse(path).getroot()
        for element in root.iter():
            element.tag = element.tag.split("}")[-1]
        if root.find("library_controllers/controller") is not None:
            raise SpawnError(f"Skinned COLLADA collision unsupported: {path}")
        geometry = {}
        for item in root.findall("library_geometries/geometry"):
            sources = {s.get("id"): s for s in item.findall("mesh/source")}
            vertices = []
            for ref in item.findall("mesh/vertices/input"):
                if ref.get("semantic") != "POSITION":
                    continue
                source = sources[ref.get("source", "").lstrip("#")]
                accessor = source.find("technique_common/accessor")
                stride = int(accessor.get("stride", "1"))
                offset = int(accessor.get("offset", "0"))
                count = int(accessor.get("count"))
                values = np.fromstring(source.findtext("float_array", ""), sep=" ")
                if stride < 3 or len(values) < offset + count*stride:
                    raise SpawnError(f"Invalid COLLADA positions: {path}")
                vertices.extend(values[offset:offset+count*stride].reshape(-1, stride)[:, :3])
            if not vertices:
                raise SpawnError(f"Missing COLLADA positions: {path}")
            vertices = np.asarray(vertices)
            geometry[item.get("id")] = corners(vertices.min(0), vertices.max(0))
        points = []

        def walk(node, parent):
            local = np.eye(4)
            for e in node:
                transform = np.eye(4)
                if e.tag == "matrix":
                    transform = numbers(e.text, 16).reshape(4, 4)
                    if not np.allclose(transform[3], [0, 0, 0, 1]):
                        raise SpawnError("Non-affine COLLADA matrix")
                elif e.tag == "translate":
                    transform[:3, 3] = numbers(e.text, 3)
                elif e.tag == "scale":
                    transform[:3, :3] = np.diag(numbers(e.text, 3))
                elif e.tag == "rotate":
                    v = numbers(e.text, 4); axis = v[:3]
                    if np.linalg.norm(axis) == 0:
                        raise SpawnError("Zero COLLADA rotation axis")
                    axis = axis / np.linalg.norm(axis)
                    x, y, z = axis; angle = math.radians(v[3])
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    transform[:3, :3] = (math.cos(angle)*np.eye(3) +
                        (1-math.cos(angle))*np.outer(axis, axis) + math.sin(angle)*skew)
                elif e.tag in {"lookat", "skew", "instance_node", "instance_controller"}:
                    raise SpawnError(f"Unsupported COLLADA {e.tag}: {path}")
                else:
                    continue
                local = local @ transform
            matrix = parent @ local
            for e in node.findall("instance_geometry"):
                key = e.get("url", "").lstrip("#")
                if key not in geometry:
                    raise SpawnError(f"Unknown COLLADA geometry {key}")
                points.extend(transformed(geometry[key], matrix))
            for child in node.findall("node"):
                walk(child, matrix)

        reference = root.find("scene/instance_visual_scene")
        scene_id = reference.get("url", "").lstrip("#") if reference is not None else None
        scenes = root.findall("library_visual_scenes/visual_scene")
        scenes = [s for s in scenes if s.get("id") == scene_id]
        if len(scenes) != 1:
            raise SpawnError(f"COLLADA must select one visual scene: {path}")
        for node in scenes[0].findall("node"):
            walk(node, np.eye(4))
        unit = root.find("asset/unit")
        scale = float(unit.get("meter", "1")) if unit is not None else 1.0
        axis = root.findtext("asset/up_axis", "Y_UP").strip()
        if axis not in {"X_UP", "Y_UP", "Z_UP"} or not math.isfinite(scale) or scale <= 0:
            raise SpawnError("Invalid COLLADA units/up axis")
        rotation = {"Z_UP": np.eye(3), "Y_UP": np.array([[1,0,0],[0,0,-1],[0,1,0]]),
                    "X_UP": np.array([[0,0,-1],[0,1,0],[1,0,0]])}[axis]
        points = np.asarray(points) @ rotation.T * scale
    else:
        raise SpawnError(f"Unsupported collision mesh format: {path}")
    points = np.asarray(points)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise SpawnError(f"Missing/nonfinite mesh vertices: {path}")
    return points.min(0), points.max(0)


class CollisionReader:
    def __init__(self, model_paths):
        self.model_paths = [Path(p) for p in model_paths if str(p)]

    def resolve(self, uri, source):
        if uri.startswith("model://"):
            relative = uri[len("model://"):]
            choices = [p / relative for p in self.model_paths]
        elif uri.startswith("file://"):
            choices = [Path(uri[len("file://"):])]
        elif "://" in uri:
            raise SpawnError(f"Only local collision assets are allowed: {uri}")
        else:
            choices = [source.parent / uri]
        for candidate in choices:
            if candidate.is_dir():
                candidate = candidate / "model.sdf"
            if candidate.is_file():
                return candidate.resolve()
        raise SpawnError(f"Collision asset cannot be resolved locally: {uri}")

    def geometry(self, geometry, source):
        if geometry is None or len(geometry) != 1:
            raise SpawnError("Collision requires exactly one supported geometry")
        g = geometry[0]
        if g.tag == "box":
            size = numbers(g.findtext("size"), 3)
            if (size <= 0).any():
                raise SpawnError("Nonpositive box size")
            return corners(-size/2, size/2)
        if g.tag in {"sphere", "cylinder"}:
            radius = float(g.findtext("radius"))
            height = float(g.findtext("length")) if g.tag == "cylinder" else 2*radius
            if not math.isfinite(radius + height) or radius <= 0 or height <= 0:
                raise SpawnError("Invalid curved collision dimensions")
            return corners([-radius,-radius,-height/2], [radius,radius,height/2])
        if g.tag == "mesh":
            if g.find("submesh") is not None:
                raise SpawnError("Collision submesh selection is unsupported")
            low, high = mesh_bounds(self.resolve(g.findtext("uri", ""), source))
            scale = numbers(g.findtext("scale", "1 1 1"), 3)
            if (scale == 0).any():
                raise SpawnError("Zero mesh scale")
            return corners(low, high)*scale
        raise SpawnError(f"Unsupported collision shape: {g.tag}")

    def model_points(self, model, source, parent=None, override=None, stack=()):
        if model.findall("frame"):
            raise SpawnError("Named SDF model frames are unsupported")
        parent = np.eye(4) if parent is None else parent
        matrix = parent @ pose(override if override is not None else model.find("pose"))
        points = []
        for link in model.findall("link"):
            link_matrix = matrix @ pose(link.find("pose"))
            for collision in link.findall("collision"):
                transform = link_matrix @ pose(collision.find("pose"))
                geometry = collision.find("geometry")
                if geometry is not None and geometry.find("plane") is not None:
                    # A horizontal ground plane is support, not an obstacle.
                    normal = numbers(geometry.findtext("plane/normal", "0 0 1"), 3)
                    world_normal = transform[:3,:3] @ normal
                    if not np.allclose(world_normal, [0,0,1]) or abs(transform[2,3]) > .001:
                        raise SpawnError("Only horizontal ground planes at z=0 are supported")
                    continue
                points.extend(transformed(self.geometry(geometry, source), transform))
        for child in model.findall("model"):
            points.extend(self.model_points(child, source, matrix, stack=stack))
        for include in model.findall("include"):
            points.extend(self.include_points(include, source, matrix, stack))
        return points

    def include_points(self, include, source, parent=None, stack=()):
        if include.find("placement_frame") is not None:
            raise SpawnError("SDF placement_frame is unsupported")
        path = self.resolve(include.findtext("uri", ""), source)
        if path in stack:
            raise SpawnError("Recursive SDF include")
        model = ET.parse(path).getroot().find("model")
        if model is None:
            # A sun model contains a top-level light, without any collision.
            root = ET.parse(path).getroot()
            if root.find("light") is not None:
                return []
            raise SpawnError(f"Included SDF has no model: {path}")
        return self.model_points(model, path, parent, include.find("pose"), stack+(path,))


@dataclass(frozen=True)
class SpawnScene:
    bounds: tuple  # xmin, xmax, ymin, ymax (inside room walls)
    obstacles: tuple  # each model's conservative world XY collision envelope

    def clearance(self, x, y):
        xmin, xmax, ymin, ymax = self.bounds
        distances = [x-xmin, xmax-x, y-ymin, ymax-y]
        for a,b,c,d in self.obstacles:
            distances.append(math.hypot(max(a-x, 0, x-b), max(c-y, 0, y-d)))
        return min(distances)


def load_scene(world_path, model_paths, bounds=None):
    path = Path(world_path).resolve(); reader = CollisionReader(model_paths)
    world = ET.parse(path).getroot().find("world")
    if world is None:
        raise SpawnError("Expected an SDF world")
    if world.find("state") is not None:
        raise SpawnError("Saved SDF world-state pose overrides are unsupported")
    obstacles = []
    for item in list(world):
        if item.tag not in {"model", "include"}:
            continue
        points = (reader.model_points(item, path) if item.tag == "model" else
                  reader.include_points(item, path))
        if not points:
            continue
        points = np.asarray(points); low, high = points.min(0), points.max(0)
        obstacles.append(tuple(float(v) for v in (low[0],high[0],low[1],high[1])))
    if bounds is None:
        # Infer a rectangular room only from enclosing, thin axis-aligned walls.
        vertical = [o for o in obstacles if o[1]-o[0] <= .4 and o[3]-o[2] >= 2]
        horizontal = [o for o in obstacles if o[3]-o[2] <= .4 and o[1]-o[0] >= 2]
        candidates = []
        for west in vertical:
            for east in vertical:
                for south in horizontal:
                    for north in horizontal:
                        x0,x1,y0,y1 = west[1],east[0],south[3],north[2]
                        if x1 <= x0 or y1 <= y0:
                            continue
                        if all(o[2] <= y0 and o[3] >= y1 for o in (west,east)) and all(
                                o[0] <= x0 and o[1] >= x1 for o in (south,north)):
                            candidates.append((x0,x1,y0,y1))
        if not candidates:
            raise SpawnError("Cannot infer enclosed room; supply spawn_bounds=xmin,xmax,ymin,ymax")
        bounds = max(candidates, key=lambda b: (b[1]-b[0])*(b[3]-b[2]))
    bounds = tuple(float(v) for v in bounds)
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:
        raise SpawnError("Invalid spawn_bounds")
    return SpawnScene(bounds, tuple(obstacles))


def robot_radius(robot_sdf, model_paths):
    path = Path(robot_sdf).resolve(); model = ET.parse(path).getroot().find("model")
    if model is None:
        raise SpawnError("Robot SDF has no model")
    points = np.asarray(CollisionReader(model_paths).model_points(model, path))
    if not points.size:
        raise SpawnError("Robot has no collision geometry")
    # Enclosing circle allows any sampled yaw and rotation during startup.
    return float(np.linalg.norm(points[:,:2], axis=1).max())


def choose_spawn(scene, radius, seed=None, x=None, y=None, yaw=None, margin=.15, max_attempts=10000):
    if not math.isfinite(radius+margin) or radius<=0 or margin<0:
        raise SpawnError("Invalid robot radius or spawn margin")
    fixed = x is not None
    if (x is None) != (y is None):
        raise SpawnError("Fixed spawn requires both x_pose and y_pose, or both AUTO")
    if yaw is not None and not math.isfinite(yaw):
        raise SpawnError("Nonfinite yaw_pose")
    seed = secrets.randbits(63) if seed is None else int(seed)
    rng = random.Random(seed); required = radius+margin
    xmin,xmax,ymin,ymax = scene.bounds
    if x is not None:
        if not math.isfinite(x+y) or scene.clearance(x,y) < required:
            raise SpawnError(f"Fixed spawn ({x}, {y}) violates collision clearance {required:.3f} m")
    else:
        for _ in range(max_attempts):
            x,y = rng.uniform(xmin,xmax),rng.uniform(ymin,ymax)
            if scene.clearance(x,y) >= required:
                break
        else:
            raise SpawnError(f"No verified safe spawn after {max_attempts} samples; no fallback used")
    yaw = rng.uniform(-math.pi,math.pi) if yaw is None else yaw
    return {"x":x,"y":y,"yaw":yaw,"seed":seed,"robot_radius":radius,
            "margin":margin,"clearance":scene.clearance(x,y),"bounds":list(scene.bounds),
            "mode":"fixed" if fixed else "random"}
