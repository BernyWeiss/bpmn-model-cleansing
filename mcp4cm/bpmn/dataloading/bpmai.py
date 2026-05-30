from typing import Any

BPMN_PROCESS_GROUP_NAME = "BPMN2.0_Process"


def extract_model_metadata(
        metadata_json: Any
) -> tuple[str, str, str]:
    id = metadata_json['model']['modelId']
    name = metadata_json['model']['modelName']
    language = metadata_json['model']['naturalLanguage']

    return id, name, language


def load_model_text(fp: str) -> str:
    with open(fp, 'r') as f:
        model_text = f.read()
    return model_text
