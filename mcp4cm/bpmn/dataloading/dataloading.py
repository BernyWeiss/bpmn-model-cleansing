import json
import os

import pandas as pd

from enum import Enum
from ast import literal_eval
from tqdm.asyncio import tqdm

from mcp4cm.bpmn.dataloading.bpmai import BPMN_PROCESS_GROUP_NAME, extract_model_metadata, \
    load_model_text
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset, BPMNModel
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model
from mcp4cm.bpmn.dataloading.sap_sam import SapSam2022Namespaces, _load_sap_sam_csv_to_df
from mcp4cm.util.text_util import get_file_hash

PROCESSED_MODELS_PATH = 'processed/reduced'
SAM_MODELS_PATH = 'sap_sam_2022/models'
BPMAI_MODELS_PATH = 'bpmai/models'



class BPMNModelCollection(Enum):
    SAP_SAM = 'sap_sam'
    BPMAI = 'bpmai'


def load_bpmn_dataset(path: str, model_collection: BPMNModelCollection, reduced_size: bool) -> BPMNDataset:
    if model_collection.value == BPMNModelCollection.SAP_SAM.value:
        return load_sap_sam_bpmn(path, reduced_size=reduced_size)
    if model_collection.value == BPMNModelCollection.BPMAI.value:
        return load_bpmai_bpmn(path)
    raise ValueError(f"Could not load BPMNDataset: BPMNModelCollection is unknown: {model_collection.value}")


def load_processed_dataset_from_csv(name: str, fp: str) -> BPMNDataset:
    def load_processed_names(name: str):
        if not name:
            return None
        return literal_eval(name)

    models = pd.read_csv(fp, na_filter=False, converters={
        "model_json": lambda x: reduce_json_model(x) if x is not None else None,
        "names": lambda x: load_processed_names(x),
        "names_with_types": lambda x: load_processed_names(x),
        "element_counts": lambda x: load_processed_names(x),
    })
    models.replace("", None, inplace=True)
    return BPMNDataset(name=name, models=models)


def load_sap_sam_bpmn(
        dataset_path: str = 'data/bpmnmodelset',
        namespace: SapSam2022Namespaces = SapSam2022Namespaces.BPMN2,
        reduced_size: bool = False,
) -> BPMNDataset:
    """

    Args:
        dataset_path:
        namespace:

    Returns:

    """
    n_files_processed = 0

    dataset_path = os.path.join(dataset_path, SAM_MODELS_PATH)
    full_dataset = None
    for model_file in tqdm(os.listdir(dataset_path), desc=f'Loading SAP SAM Dataset @ {dataset_path}'):
        if not model_file.endswith('.csv'):
            continue

        full_file_path = os.path.join(dataset_path, model_file)

        partial_df = _load_sap_sam_csv_to_df(file_path=full_file_path, relevant_namespace=namespace, cull_json=True)

        if full_dataset is None:
            full_dataset = partial_df
        else:
            full_dataset = pd.concat([full_dataset, partial_df])

        if reduced_size:
            n_files_processed += 1
            if n_files_processed > 1:
                break

    full_dataset.reset_index(drop=False, inplace=True, names='id')
    full_dataset.fillna({'name': ''}, inplace=True)

    bpmn2_dataset = BPMNDataset(name="sapsam_2022_bpmn2", models=full_dataset)
    return bpmn2_dataset


def load_bpmai_bpmn(
        path: str = 'data/bpmnmodelset',
) -> BPMNDataset:
    path = os.path.join(path, BPMAI_MODELS_PATH)

    files = os.listdir(path)
    models = []
    for file in files:
        if not file.endswith('.meta.json'):
            continue

        model_metadata = json.load(open(os.path.join(path, file), 'r'))
        group = model_metadata['model']['groupName']
        if not group == BPMN_PROCESS_GROUP_NAME:
            continue

        id, name, language = extract_model_metadata(model_metadata)

        file_path = os.path.join(path, f'{id}.json')

        model_json_str = load_model_text(file_path)
        reduced_model_json = reduce_json_model(model_json_str)

        bpmn_model = BPMNModel(
            id=id,
            file_path=file_path,
            hash=None,
            language=language,
            model_json=reduced_model_json,
            name=name,
        )

        models.append(bpmn_model)

    return BPMNDataset(name='BPMAI Dataset', models=models)


