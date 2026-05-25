import hashlib
import re


def join_texts(text, delim: str = ' ', empty_name: str | None = None) -> str:
    if isinstance(text, list):
        text = sorted(text)
        if empty_name:
            text = [expression for expression in text if expression != empty_name]
        text = f'{delim}'.join(text)
    elif isinstance(text, dict):
        flat_texts = [
            text for list_or_str in text.values()
            for text in (list_or_str if isinstance(list_or_str, list) else [list_or_str])
        ]
        if empty_name:
            flat_texts = [expression for expression in flat_texts if expression != empty_name]
        text = f'{delim}'.join(flat_texts)
    return text


def get_file_hash(string: str) -> str:
    """
    Compute a SHA-256 hash for a string.

    Args:
        string (str): The string to hash.

    Returns:
        str: The SHA-256 hash of the string.

    Example:
        >>> hash_value = get_file_hash("example content")
        >>> print(hash_value)
        '5d41402abc4b2a76b9719d911017c592'
    """

    return hashlib.sha256(string.encode(encoding='utf-8', errors='strict')).hexdigest()  # Hash the content


def split_name(name: str):
    """Splits camelCase, PascalCase, and snake_case names into words and converts them to lowercase."""
    name = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name)
    name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = name.replace("_", " ").lower()
    return name
