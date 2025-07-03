from mcp4cm.base import Model


def get_model_text(model: Model, key: str) -> str:
    text = getattr(model, key, '')
    if isinstance(text, list):
        text = sorted(text)
        text = ' '.join(text)
    elif isinstance(text, dict):
        text = ' '.join(text.values())
    
    return text