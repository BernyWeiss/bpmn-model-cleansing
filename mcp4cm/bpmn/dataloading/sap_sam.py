import json
import pandas as pd

from enum import Enum

from pandas import DataFrame

from mcp4cm.util.text_util import get_file_hash
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model


class SapSam2022Namespaces(Enum):
    """
    Enum for different Namespaces in the sap_sam_2022 dataset.

    This enumeration defines all namespaces of the dataset which are supported to load as a dataset.

    Currently only BPMN 2.0 models are supported.
    """
    BPMN2 = 'http://b3mn.org/stencilset/bpmn2.0#'


def _load_sap_sam_csv_to_df(file_path: str, relevant_namespace: SapSam2022Namespaces, cull_json: bool) -> DataFrame:
    partial_df = pd.read_csv(file_path, dtype={"Namespace": "category"})

    model_type = relevant_namespace.value

    partial_df.query(f'Namespace =="{model_type}"', inplace=True)
    partial_df.drop(columns=['Revision ID', 'Organization ID', 'Datetime', 'Description', 'Type', 'Namespace'],
                    inplace=True)
    partial_df.rename(columns={'Model ID': 'id', 'Name': 'name'}, inplace=True)
    partial_df.set_index('id', inplace=True)

    if cull_json:
        partial_df['model_json'] = partial_df['Model JSON'].apply(reduce_json_model)
        partial_df.drop(columns=['Model JSON'], inplace=True)
    else:
        partial_df.rename(columns={'Model JSON': 'model_json'}, inplace=True)

    partial_df['file_path'] = file_path
    partial_df['hash'] = partial_df['model_json'].apply(lambda model_json: get_file_hash(json.dumps(model_json)))

    partial_df['language'] = None
    partial_df['names'] = None
    partial_df['names_with_types'] = None
    partial_df['model_xmi'] = None
    partial_df['model_txt'] = None
    partial_df['category'] = None
    partial_df['tags'] = None
    return partial_df
