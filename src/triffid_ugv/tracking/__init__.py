"""Object identity across frames: the geo-space world map, fed by appearance track hints (SIFT or MASA)."""
from .world import ObjectMap, Observation, WorldObject

__all__ = ['ObjectMap', 'Observation', 'WorldObject']
