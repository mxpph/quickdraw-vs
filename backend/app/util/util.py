"""Helper functions for the main module"""
import uuid
from random import choice
from collections import Counter

def is_valid_uuid(uuid_string: str) -> bool:
    """Return true if the uuid_string given is valid"""
    try:
        val = uuid.UUID(uuid_string, version=4)
    except ValueError:
        return False
    return str(val) == uuid_string

def get_random_word() -> str:
    """Returns a random word from the classnames"""
    words = [
        "apple",
        "anvil",
        "dresser",
        "broom",
        "hat",
        "camera",
        "dog",
        "basketball",
        "pencil",
        "hammer",
        "hexagon",
        "banana",
        "angel",
        "airplane",
        "ant",
        "paper clip",
    ]
    return choice(words)

def most_common(lst: list[str]) -> str:
    """Return the most common string of a list"""
    data = Counter(lst)
    return data.most_common(1)[0][0]
