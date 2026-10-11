"""Optional measured height from one retained, accepted RGB-D observation.

This metadata never replaces Memory's XY supports, identities or class votes.
"""
import copy
import math

from .observation_hints import _points


def _validated(source, object_id, epoch, odom_frame):
    if (not isinstance(source, dict) or type(epoch) is not int or epoch < 0
            or source.get('epoch') != epoch or type(source.get('epoch')) is not int
            or source.get('object_id') != object_id
            or not isinstance(source.get('view_id'), str) or not source['view_id']
            or not isinstance(odom_frame, str) or not odom_frame
            or source.get('odom_frame') != odom_frame):
        raise ValueError('framing source identity, epoch or frame mismatch')
    stamp, depth_stamp = source['stamp_ns'], source['depth_stamp_ns']
    if (type(stamp) is not int or type(depth_stamp) is not int or min(stamp, depth_stamp) <= 0
            or abs(stamp-depth_stamp) > 5_000_000):
        raise ValueError('framing RGB-D stamp mismatch')
    box = source['box']
    if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
            or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('invalid accepted source box')
    group = source['surface_group']
    xyz, pixels = _points(group['odom_xyz'], 3), _points(group['pixels'], 2)
    if (len(xyz) != len(pixels) or type(group['count']) is not int
            or group['count'] < len(xyz)):
        raise ValueError('framing measured count mismatch')
    if any(u != int(u) or v != int(v) or not box[0] <= u <= box[2]
           or not box[1] <= v <= box[3] for u, v in pixels):
        raise ValueError('framing pixels outside exact accepted box')
    return [tuple(p) for p in xyz.tolist()]


def retained_framing_source(memory, track, epoch, now_ns):
    """Copy only the newest accepted exact-box cache; never localize or vote.

    Missing/bad metadata on this source disables height. Do not cherry-pick
    points or fall back to a lower/easier old surface. Eviction naturally makes
    the next newest still-retained observation the source on the next publish.
    """
    try:
        choices = [(view, observation) for view in memory.views
                   for observation in view.observations
                   if observation['object_id'] == track['object_id']
                   and view.view_id in track['view_ids']]
        if not choices or track.get('confirmed') or track.get('conflicted'):
            return None
        view, observation = max(choices, key=lambda pair:
            (pair[0].stamp_ns, pair[0].view_id, tuple(pair[1]['box'])))
        payload, box = view.payload, tuple(observation['box'])
        metadata = payload['metadata']
        image = payload['image']
        image_stamp = image.header.stamp.sec*1_000_000_000+image.header.stamp.nanosec
        if ('depth_m' not in payload or metadata.get('localization_source') != 'aligned_depth'
                or metadata['image_stamp_ns'] != view.stamp_ns or image_stamp != view.stamp_ns
                or view.stamp_ns > now_ns
                or image.header.frame_id != payload['calibration'].frame_id
                or box[2] > image.width or box[3] > image.height):
            return None
        _, geometry = payload['depth_geometry'][box]
        if geometry.get('localization_source') != 'aligned_depth':
            return None
        source = dict(epoch=epoch, object_id=track['object_id'], view_id=view.view_id,
                      stamp_ns=view.stamp_ns, depth_stamp_ns=metadata['depth_stamp_ns'],
                      odom_frame=payload['odom_frame'], box=list(box),
                      surface_group=copy.deepcopy(geometry['surface_group']))
        _validated(source, track['object_id'], epoch, payload['odom_frame'])
        return source
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def framing_xyz(track, views, epoch, odom_frame):
    """Validate the entire optional source against current serialized Memory."""
    try:
        source = track['framing_source']
        xyz = _validated(source, track['object_id'], epoch, odom_frame)
        if (track.get('confirmed') or track.get('conflicted')
                or source['view_id'] not in track['view_ids']):
            return []
        view = next(v for v in views if v['view_id'] == source['view_id'])
        if (view['stamp_ns'] != source['stamp_ns'] or not any(
                o['object_id'] == track['object_id'] and list(o['box']) == source['box']
                for o in view['observations'])):
            return []
        return xyz
    except (KeyError, TypeError, ValueError, StopIteration, OverflowError):
        return []
