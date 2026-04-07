from mcp4cm.base import Model
from typing import Dict, List


def get_model_text(model: Model, key: str, delim=' ', empty_name: str | None = None) -> str:
    text = getattr(model, key, '')
    text = join_texts(text, delim=delim, empty_name=empty_name)
    return text


def join_texts(text: str | List[str] | Dict[str], delim: str = ' ', empty_name: str | None = None) -> str:
    if isinstance(text, list):
        text = sorted(text)
        if empty_name:
            text = [expression for expression in text if expression != empty_name]
        text = f'{delim}'.join(text)
    elif isinstance(text, dict):
        text_values = text.values()
        if empty_name:
            text_values = [expression for expression in text if expression != empty_name]
        text = f'{delim}'.join(text_values)
    return text
