from langdetect import detect, LangDetectException

from mcp4cm.base import Model


def get_model_text(model: Model, key: str, delim=' ', empty_name: str | None = None) -> str:
    text = getattr(model, key, '')
    text = join_texts(text, delim=delim, empty_name=empty_name)
    return text


def join_texts(text, delim: str = ' ', empty_name: str | None = None) -> str:
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


def get_text_language(text: str) -> str:
    if text and text.strip():  # Ensure it's not empty or whitespace
        try:
            return detect(text)
        except LangDetectException:
            return None
    return None
