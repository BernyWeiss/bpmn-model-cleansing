from typing import Any, Final

BPMN_PROCESS_GROUP_NAME: Final = "BPMN2.0_Process"


def _extract_model_metadata(
        metadata_json: Any
) -> tuple[str, str, str]:
    id = metadata_json['model']['modelId']
    name = metadata_json['model']['modelName']
    language = metadata_json['model']['naturalLanguage']

    return id, name, language


def _load_model_text(fp: str) -> str:
    with open(fp, 'r') as f:
        model_text = f.read()
    return model_text
