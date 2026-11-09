"""World objects -> RFC 7946 GeoJSON Features with the map manager's SimpleStyle properties.

Geometry: Point by default; LineString for `geometry.line_classes` (fence, wall); Polygon for
`geometry.polygon_classes`. Coordinates are [longitude, latitude]; altitude is a property.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..config import GeometryConfig
from ..geodesy import LocalFrame
from ..ontology import class_style
from ..tracking.world import WorldObject


def iso(stamp_ns: int) -> str:
    return datetime.fromtimestamp(stamp_ns / 1e9, tz=timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _square(centre: np.ndarray, half: float) -> np.ndarray:
    return np.array([centre + [half, half], centre + [half, -half], centre + [-half, -half], centre + [-half, half]])


def object_geometry(obj: WorldObject, frame: LocalFrame, geometry: GeometryConfig) -> dict:
    name = obj.class_name
    if name in geometry.line_classes and obj.line is not None:
        a, b = obj.line
        if np.linalg.norm(b - a) < geometry.min_extent_m:     # degenerate: give it a visible length
            direction = (b - a) / (np.linalg.norm(b - a) or 1.0) if np.any(b != a) else np.array([1.0, 0.0])
            mid = (a + b) / 2
            a, b = mid - direction * geometry.min_extent_m / 2, mid + direction * geometry.min_extent_m / 2
        return dict(type='LineString', coordinates=[frame.lon_lat([*a, 0.0]), frame.lon_lat([*b, 0.0])])
    if name in geometry.polygon_classes:
        ring = obj.hull if obj.hull is not None and len(obj.hull) >= 3 else _square(obj.enu[:2], geometry.min_extent_m / 2)
        coords = [frame.lon_lat([x, y, 0.0]) for x, y in ring]
        return dict(type='Polygon', coordinates=[coords + [coords[0]]])
    return dict(type='Point', coordinates=frame.lon_lat(obj.enu))


def object_feature(obj: WorldObject, frame: LocalFrame, geometry: GeometryConfig, source: str = 'ugv',
                   tracker: str = 'geo') -> dict:
    name = obj.class_name
    style = class_style(name)
    geom = object_geometry(obj, frame, geometry)
    _, _, alt = frame.to_geodetic(obj.enu)
    props = {
        'class': name,
        'id': obj.id,
        'confidence': round(float(obj.best_score), 3),
        'category': style['category'],
        'detection_type': 'seg',
        'source': source,
        'local_frame': False,
        'altitude_m': round(float(alt), 2),
        'height_m': round(float(obj.height_m), 2),
        'marker-color': style['color'],
        'marker-size': 'medium',
        'marker-symbol': style['symbol'],
        'observations': obj.hits,
        'first_seen': iso(obj.first_seen_ns),
        'last_seen': iso(obj.last_seen_ns),
        'tracker': tracker,
    }
    if geom['type'] in ('Polygon', 'LineString'):
        props.update({'stroke': style['color'], 'stroke-width': 2, 'stroke-opacity': 1.0})
    if geom['type'] == 'Polygon':
        props.update({'fill': style['color'], 'fill-opacity': 0.25})
    return {'type': 'Feature', 'id': obj.id, 'geometry': geom, 'properties': props}


def feature_collection(features: list[dict], **meta) -> dict:
    collection = {'type': 'FeatureCollection', 'features': features}
    if meta:
        collection['metadata'] = meta
    return collection


def feature_centroid(feature: dict) -> tuple[float, float] | None:
    """(lon, lat) centre of a Point / LineString / Polygon feature."""
    geom = feature.get('geometry') or {}
    coords = geom.get('coordinates')
    try:
        if geom.get('type') == 'Point':
            return float(coords[0]), float(coords[1])
        if geom.get('type') == 'LineString':
            pts = coords
        elif geom.get('type') == 'Polygon':
            pts = coords[0][:-1] if len(coords[0]) > 1 else coords[0]
        else:
            return None
        return (sum(float(p[0]) for p in pts) / len(pts), sum(float(p[1]) for p in pts) / len(pts))
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        return None
