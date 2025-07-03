import re

def split_name(name):
    """Splits camelCase, PascalCase, and snake_case names into words and converts them to lowercase."""
    name = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name)
    name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = name.replace("_", " ").lower()
    return name
