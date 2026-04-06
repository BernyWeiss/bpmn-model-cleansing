import re
from pathlib import Path

def split_name(name):
    """Splits camelCase, PascalCase, and snake_case names into words and converts them to lowercase."""
    name = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name)
    name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = name.replace("_", " ").lower()
    return name

def create_directories_for_path(path: str):
    file_path = Path(path)
    directory_path = file_path.parent
    directory_path.mkdir(parents=True, exist_ok=True)