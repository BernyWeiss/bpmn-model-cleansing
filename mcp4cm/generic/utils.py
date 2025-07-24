from mcp4cm.base import Model


def get_model_text(model: Model, key: str, delim=' ') -> str:
    text = getattr(model, key, '')
    if isinstance(text, list):
        text = sorted(text)
        text = f'{delim}'.join(text)
    elif isinstance(text, dict):
        text = f'{delim}'.join(text.values())
    
    return text