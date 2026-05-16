import json
import os
import pandas as pd

from enum import Enum
from tqdm.asyncio import tqdm

from mcp4cm.util.text_util import get_file_hash
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model

SAM_MODELS_PATH = 'sap_sam_2022/models'


class SapSam2022Namespaces(Enum):
    """
    Enum for different Namespaces in the sap_sam_2022 dataset.

    This enumeration defines all namespaces of the dataset which are supported to load as a dataset.

    Currently only BPMN 2.0 models are supported.
    """
    BPMN2 = 'http://b3mn.org/stencilset/bpmn2.0#'


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

        partial_df = pd.read_csv(os.path.join(dataset_path, model_file), dtype={"Namespace": "category"})

        model_type = namespace.value

        partial_df.query(f'Namespace =="{model_type}"', inplace=True)
        partial_df.drop(columns=['Revision ID', 'Organization ID', 'Datetime', 'Description', 'Type', 'Namespace'],
                        inplace=True)
        partial_df.rename(columns={'Model ID': 'id', 'Name': 'name'}, inplace=True)
        partial_df.set_index('id', inplace=True)

        partial_df['model_json'] = partial_df['Model JSON'].apply(reduce_json_model)
        partial_df.drop(columns=['Model JSON'], inplace=True)

        partial_df['file_path'] = os.path.join(dataset_path, model_file)
        partial_df['hash'] = partial_df['model_json'].apply(lambda model_json: get_file_hash(json.dumps(model_json)))

        partial_df['language'] = None
        partial_df['names'] = None
        partial_df['names_with_types'] = None
        partial_df['model_xmi'] = None
        partial_df['model_txt'] = None
        partial_df['category'] = None
        partial_df['tags'] = None

        if full_dataset is None:
            full_dataset = partial_df
        else:
            full_dataset = pd.concat([full_dataset, partial_df])

        if reduced_size:
            n_files_processed += 1
            if n_files_processed > 10:
                break

    full_dataset.reset_index(drop=False, inplace=True, names='id')
    full_dataset.fillna({'name': ''}, inplace=True)

    bpmn2_dataset = BPMNDataset(name="sapsam_2022_bpmn2", models=full_dataset)
    return bpmn2_dataset
