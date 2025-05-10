import json
import logging
import re
from json import JSONDecodeError
from typing import Any

logger = logging.getLogger(__name__)


def fix_loose_json_string(s: str) -> list[dict[str, Any]]:
    """
    Converts a loosely formatted JSON string into a list of dictionaries.

    Args:
        s (str): The loosely formatted JSON string.

    Returns:
        list[dict[str, Any]]: A list of dictionaries parsed from the string.
    """
    loose_str = s.strip()
    if loose_str.startswith("[") and loose_str.endswith("]"):
        loose_str = loose_str[1:-1]

    obj_strings = re.findall(r"\{.*?\}", loose_str)
    fixed_objects = []

    for obj_str in obj_strings:
        obj_fixed = re.sub(r"(\w+)=", r'"\1":', obj_str)
        obj_fixed = re.sub(r':\s*([^"{},\[\]]+)', r': "\1"', obj_fixed)
        obj_fixed = obj_fixed.replace("'", '"')

        try:
            fixed_obj = json.loads(obj_fixed)
            fixed_objects.append(fixed_obj)
        except json.JSONDecodeError:
            logger.debug(f"Failed to decode JSON object: {obj_fixed}")
            continue

    return fixed_objects


def sanitize_json_input(bad_json_str: str) -> str:
    """
    Cleans a JSON string by escaping invalid backslashes.

    Args:
        bad_json_str (str): The JSON string with potential invalid backslashes.

    Returns:
        str: The sanitized JSON string.
    """

    def escape_bad_backslashes(match: Any) -> Any:
        return match.group(0).replace("\\", "\\\\")

    invalid_escape_re = re.compile(r'\\(?!["\\/bfnrtu])')
    cleaned = invalid_escape_re.sub(escape_bad_backslashes, bad_json_str)
    return cleaned


def safe_json_loads(json_str: str) -> Any:
    """
    Safely loads a JSON string, attempting to sanitize it if initial loading fails.

    Args:
        json_str (str): The JSON string to load.

    Returns:
        Any: The loaded JSON object.
    """
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        logger.debug(f"JSONDecodeError encountered: {e}. Attempting to sanitize input.")
        return json.loads(sanitize_json_input(json_str))
