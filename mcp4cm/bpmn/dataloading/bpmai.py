import json
import os
from collections import Counter
from typing import Any

from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNModel, BPMNDataset
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model
from mcp4cm.util.text_util import get_file_hash

BPMAI_MODELS_PATH = 'bpmai/models'
BPMN_PROCESS_GROUP_NAME = "BPMN2.0_Process"


def load_bpmai_bpmn(
        path: str = 'data/bpmnmodelset',
) -> BPMNDataset:
    path = os.path.join(path, BPMAI_MODELS_PATH)

    files = os.listdir(path)
    group_counter = Counter()
    models = []
    for file in files:
        if not file.endswith('.meta.json'):
            continue

        model_metadata = json.load(open(os.path.join(path, file), 'r'))
        group = model_metadata['model']['groupName']
        group_counter[group] += 1
        if not group == BPMN_PROCESS_GROUP_NAME:
            continue

        id, name, language = _extract_model_metadata(model_metadata)

        file_path = os.path.join(path, f'{id}.json')

        model_json = _load_model_text(file_path)
        reduced_model_json = reduce_json_model(model_json)
        hash = get_file_hash(json.dumps(reduced_model_json))

        bpmn_model = BPMNModel(
            id=id,
            file_path=file_path,
            hash=hash,
            language=language,
            model_json=reduced_model_json,
            name=name,
        )

        models.append(bpmn_model)

    print('Groups')
    print(group_counter)

    print(f'len(models): {len(models)}')

    return BPMNDataset(name='BPMAI Dataset', models=models)


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
