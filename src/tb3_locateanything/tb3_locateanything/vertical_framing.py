"""Conditional image-height prediction from measured XYZ and calibrated poses.

This is a soft viewpoint preference, not a visibility or recognition guarantee.
The future base height/roll/pitch are held at the captured values; only the
actual Nav2 XY-disc/yaw-interval arrival tolerances are bounded here.
"""
from dataclasses import dataclass
import math


def _finite(values, size):
    values = tuple(float(v) for v in values)
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid finite 3D framing input")
    return values


def _rotate(q, p):
    x, y, z, w = q
    ux, uy, uz = 2*(y*p[2]-z*p[1]), 2*(z*p[0]-x*p[2]), 2*(x*p[1]-y*p[0])
    return (p[0]+w*ux+y*uz-z*uy, p[1]+w*uy+z*ux-x*uz, p[2]+w*uz+x*uy-y*ux)


def _quat(raw):
    q = _finite(raw, 4)
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm-1.) > 1e-3:
        raise ValueError("framing quaternion must be unit length")
    return tuple(v/norm for v in q)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _min_sinusoid(a, b, limit):
    points = [-limit, limit]
    minimum = math.atan2(b, a)+math.pi
    for turn in range(-2, 3):
        angle = minimum+turn*math.tau
        if -limit <= angle <= limit:
            points.append(angle)
    return min(a*math.cos(t)+b*math.sin(t) for t in points)


@dataclass(frozen=True)
class VerticalFraming:
    height: int
    fy: float
    cy: float
    mount_quaternion: tuple
    mount_translation: tuple
    base_translation: tuple
    base_quaternion: tuple
    xy_tolerance: float
    yaw_tolerance: float

    def __post_init__(self):
        if (not isinstance(self.height, int) or self.height < 8 or
                not all(math.isfinite(v) for v in (self.fy, self.cy, self.xy_tolerance, self.yaw_tolerance)) or
                self.fy <= 0 or self.xy_tolerance < 0 or not 0 <= self.yaw_tolerance < math.pi/2):
            raise ValueError("invalid vertical calibration or arrival tolerances")
        edge = max(2., self.height*.005)
        if not edge < self.cy < self.height-edge:
            raise ValueError("vertical principal point leaves no image margin")
        object.__setattr__(self, 'mount_quaternion', _quat(self.mount_quaternion))
        object.__setattr__(self, 'base_quaternion', _quat(self.base_quaternion))
        object.__setattr__(self, 'mount_translation', _finite(self.mount_translation, 3))
        object.__setattr__(self, 'base_translation', _finite(self.base_translation, 3))

    def predict(self, point, yaw, support_xyz):
        point = _finite(point, 2)
        if not math.isfinite(yaw):
            raise ValueError("nonfinite candidate yaw")
        support = [_finite(p, 3) for p in support_xyz]
        if not support:
            raise ValueError("no measured XYZ for vertical prediction")
        x, y, z, w = self.base_quaternion
        old_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        delta = yaw-old_yaw
        c, s = math.cos(delta), math.sin(delta)
        def world(vector):
            b = _rotate(self.base_quaternion, _rotate(self.mount_quaternion, vector))
            return (c*b[0]-s*b[1], s*b[0]+c*b[1], b[2])
        edge = max(2., self.height*.005)
        planes = [('front', (0., 0., 1.)), ('top', (0., self.fy, self.cy-edge)),
                  ('bottom', (0., -self.fy, self.height-edge-self.cy))]
        robust, nominal = {}, {}
        for name, normal in planes:
            norm = math.sqrt(_dot(normal, normal))
            normal = tuple(v/norm for v in normal)
            n = world(normal)
            offset = _dot(_rotate(self.mount_quaternion, normal), self.mount_translation)
            values, centers = [], []
            for p in support:
                dx, dy, dz = p[0]-point[0], p[1]-point[1], p[2]-self.base_translation[2]
                a, b = n[0]*dx+n[1]*dy, n[0]*dy-n[1]*dx
                constant = n[2]*dz-offset
                centers.append(a+constant)
                values.append(_min_sinusoid(a, b, self.yaw_tolerance)+constant-
                              self.xy_tolerance*math.hypot(n[0], n[1]))
            robust[name], nominal[name] = min(values), min(centers)
        if not all(math.isfinite(v) for v in (*robust.values(), *nominal.values())):
            raise ValueError("nonfinite derived vertical-framing margin")
        return {'arrival_tolerance_fits': all(v > 0 for v in robust.values()),
                'nominal_fits': all(v > 0 for v in nominal.values()),
                'minimum_plane_margin_m': robust, 'nominal_plane_margin_m': nominal,
                'support_count': len(support), 'xy_tolerance_m': self.xy_tolerance,
                'yaw_tolerance_rad': self.yaw_tolerance,
                'base_height_m': self.base_translation[2],
                'captured_base_quaternion_xyzw': list(self.base_quaternion),
                'mount_translation_xyz': list(self.mount_translation),
                'mount_quaternion_xyzw': list(self.mount_quaternion),
                'image_height': self.height, 'fy': self.fy, 'cy': self.cy,
                'measured_map_height_range_m': [min(p[2] for p in support),
                                              max(p[2] for p in support)],
                'assumption': 'Captured base height/roll/pitch unchanged; only XY/yaw arrival errors bounded.',
                'scope': 'Measured XYZ only; no unseen-height or semantic-recognition guarantee.'}
