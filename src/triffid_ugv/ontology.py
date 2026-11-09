"""The 63 TRIFFID UGV classes (original numeric IDs) and their map styles.

Class IDs are stable and shared with the original training ontology. Styles follow the SimpleStyle
properties the map manager has been receiving from the UGV since the first live deployment.
"""
from dataclasses import dataclass

TARGET_CLASSES = {
    0: 'water',
    1: 'fence',
    2: 'green tree',
    3: 'helmet',
    4: 'flame',
    5: 'smoke',
    6: 'first responder',
    7: 'destroyed vehicle',
    8: 'fire hose',
    9: 'scba',
    10: 'boot',
    11: 'green plant',
    12: 'mask',
    13: 'window',
    14: 'building',
    15: 'destroyed building',
    16: 'debris',
    17: 'ladder',
    18: 'dirt road',
    19: 'dry tree',
    20: 'wall',
    21: 'civilian vehicle',
    22: 'road',
    23: 'citizen',
    24: 'green grass',
    25: 'pole',
    26: 'boat',
    27: 'pavement',
    28: 'dry grass',
    29: 'animal',
    30: 'excavator',
    31: 'door',
    32: 'mud',
    33: 'barrier',
    34: 'hole in the ground',
    35: 'bag',
    36: 'burnt tree',
    37: 'ambulance',
    38: 'fire truck',
    39: 'cone',
    40: 'bicycle',
    41: 'tower',
    42: 'silo',
    43: 'military personnel',
    44: 'burnt grass',
    45: 'ax',
    46: 'glove',
    47: 'crane',
    48: 'stairs',
    49: 'dry plant',
    50: 'furniture',
    51: 'tank',
    52: 'protective glasses',
    53: 'barrel',
    54: 'shovel',
    55: 'fire hydrant',
    56: 'police vehicle',
    57: 'burnt plant',
    58: 'army vehicle',
    59: 'chainsaw',
    60: 'aerial vehicle',
    61: 'lifesaver',
    62: 'extinguisher',
}

def class_color(class_name: str) -> str:
    colors = {
        # Fire / hazard (reds)
        'flame': '#ff0000',
        'smoke': '#ff4500',
        'burnt tree': '#8b0000',
        'burnt grass': '#a52a2a',
        'burnt plant': '#b22222',
        'fire hose': '#dc143c',
        'fire hydrant': '#ff6347',
        'fire truck': '#ff0000',
        'extinguisher': '#ff1493',
        # People (blues)
        'first responder': '#1e90ff',
        'citizen': '#4169e1',
        'military personnel': '#000080',
        # Vehicles (dark blues)
        'civilian vehicle': '#0000ff',
        'destroyed vehicle': '#00008b',
        'ambulance': '#4682b4',
        'police vehicle': '#191970',
        'army vehicle': '#2f4f4f',
        'boat': '#5f9ea0',
        'bicycle': '#00ff00',
        'aerial vehicle': '#87ceeb',
        # Nature (greens)
        'green tree': '#228b22',
        'green plant': '#32cd32',
        'green grass': '#7cfc00',
        'dry tree': '#daa520',
        'dry grass': '#bdb76b',
        'dry plant': '#f0e68c',
        'animal': '#ff8c00',
        # Infrastructure (greys)
        'building': '#708090',
        'destroyed building': '#696969',
        'wall': '#808080',
        'road': '#a9a9a9',
        'pavement': '#c0c0c0',
        'dirt road': '#d2b48c',
        'window': '#b0c4de',
        'door': '#8b4513',
        'stairs': '#a0522d',
        'pole': '#778899',
        'tower': '#556b2f',
        'silo': '#6b8e23',
        # Obstacles (oranges / yellows)
        'debris': '#ff8c00',
        'fence': '#daa520',
        'barrier': '#ffd700',
        'cone': '#ff7f50',
        'hole in the ground': '#8b4513',
        'mud': '#a0522d',
        'water': '#00bfff',
        # Equipment (purples)
        'helmet': '#9370db',
        'scba': '#8a2be2',
        'boot': '#4b0082',
        'mask': '#9400d3',
        'glove': '#da70d6',
        'protective glasses': '#ba55d3',
        'ladder': '#ff8c00',
        'ax': '#cd853f',
        'shovel': '#d2691e',
        'chainsaw': '#b8860b',
        'bag': '#bc8f8f',
        'barrel': '#8b8682',
        'furniture': '#deb887',
        'tank': '#2e8b57',
        'crane': '#b8860b',
        'excavator': '#daa520',
        'lifesaver': '#ff4500',
    }
    return colors.get(class_name, '#808080')

def class_category(class_name: str) -> str:
    categories = {
        # Hazard
        'flame': 'hazard', 'smoke': 'hazard',
        'burnt tree': 'hazard', 'burnt grass': 'hazard', 'burnt plant': 'hazard',
        # People
        'first responder': 'person', 'citizen': 'person',
        'military personnel': 'person',
        # Vehicles
        'civilian vehicle': 'vehicle', 'destroyed vehicle': 'vehicle',
        'ambulance': 'vehicle', 'police vehicle': 'vehicle',
        'fire truck': 'vehicle', 'army vehicle': 'vehicle',
        'boat': 'vehicle', 'bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
        # Nature
        'green tree': 'nature', 'green plant': 'nature',
        'green grass': 'nature', 'dry tree': 'nature',
        'dry grass': 'nature', 'dry plant': 'nature', 'animal': 'nature',
        # Infrastructure
        'building': 'infrastructure', 'destroyed building': 'infrastructure',
        'wall': 'infrastructure', 'road': 'infrastructure',
        'pavement': 'infrastructure', 'dirt road': 'infrastructure',
        'window': 'infrastructure', 'door': 'infrastructure',
        'stairs': 'infrastructure', 'pole': 'infrastructure',
        'tower': 'infrastructure', 'silo': 'infrastructure',
        # Obstacle
        'debris': 'obstacle', 'fence': 'obstacle', 'barrier': 'obstacle',
        'cone': 'obstacle', 'hole in the ground': 'obstacle',
        'mud': 'obstacle', 'water': 'obstacle',
        # Equipment
        'fire hose': 'equipment', 'fire hydrant': 'equipment',
        'extinguisher': 'equipment', 'helmet': 'equipment',
        'scba': 'equipment', 'boot': 'equipment', 'mask': 'equipment',
        'glove': 'equipment', 'protective glasses': 'equipment',
        'ladder': 'equipment', 'ax': 'equipment', 'shovel': 'equipment',
        'chainsaw': 'equipment', 'bag': 'equipment', 'barrel': 'equipment',
        'furniture': 'equipment', 'tank': 'equipment', 'crane': 'equipment',
        'excavator': 'equipment', 'lifesaver': 'equipment',
    }
    return categories.get(class_name, 'unknown')

def class_symbol(class_name: str) -> str:
    symbols = {
        'first responder': 'pitch',
        'citizen': 'pitch',
        'military personnel': 'pitch',
        'civilian vehicle': 'car',
        'destroyed vehicle': 'car',
        'ambulance': 'hospital',
        'police vehicle': 'police',
        'fire truck': 'fire-station',
        'army vehicle': 'car',
        'boat': 'harbor',
        'bicycle': 'bicycle',
        'aerial vehicle': 'airfield',
        'flame': 'fire-station',
        'smoke': 'fire-station',
        'building': 'building',
        'destroyed building': 'building',
        'water': 'water',
        'animal': 'dog-park',
    }
    return symbols.get(class_name, 'marker')


@dataclass(frozen=True)
class UgvClass:
    id: int
    name: str


CLASSES = tuple(UgvClass(i, name) for i, name in TARGET_CLASSES.items())
BY_ID = {item.id: item for item in CLASSES}
BY_NAME = {item.name: item for item in CLASSES}


def resolve_class(value) -> UgvClass:
    """A class from its name (case-insensitive, `_` for spaces: first_responder) or numeric ID."""
    text = str(value).strip().lower().replace('_', ' ')
    item = BY_ID.get(int(text)) if text.isdigit() else BY_NAME.get(text)
    if item is None:
        raise ValueError(f'Unknown UGV class: {value!r}')
    return item


def resolve_classes(values) -> tuple[UgvClass, ...]:
    """Classes from a list or a comma-separated string, de-duplicated in order."""
    if isinstance(values, str):
        values = [v for v in values.split(',') if v.strip()]
    items = tuple(dict.fromkeys(resolve_class(v) for v in values))
    if not items:
        raise ValueError('Select at least one UGV class')
    return items


def class_style(class_name: str) -> dict:
    return dict(category=class_category(class_name), color=class_color(class_name), symbol=class_symbol(class_name))
