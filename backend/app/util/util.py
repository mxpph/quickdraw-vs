"""Helper functions for the main module"""
import uuid
from asyncio import sleep
from random import choice, random, choices
from collections import Counter
import string

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

def most_common(lst: list[str]) -> list[tuple[str, int]]:
    """
    List the elements of lst and their frequency in descending order
    of frequency.
    """
    data = Counter(lst)
    return data.most_common()

async def random_sleep(time: float):
    await sleep(time * random())

def random_string() -> str:
    return ''.join(choices(string.ascii_lowercase + string.digits, k=5))

def is_valid_game_id(game_id: str) -> bool:
    return len(game_id) == 5 and game_id.isalnum()
