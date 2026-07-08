import pandas as pd

from typing import Final
from pandas import DataFrame

from bpmn.constants import HASH_COLUMN, LANGUAGE_COLUMN, NAMES_COLUMN, NAMES_WITH_TYPES_COLUMN
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model

SAP_SAM_BPMN_NAMESPACE: Final = 'http://b3mn.org/stencilset/bpmn2.0#'

def _load_sap_sam_csv_to_df(file_path: str, relevant_namespace: str, cull_json: bool) -> DataFrame:
    partial_df = pd.read_csv(file_path, dtype={"Namespace": "category"})

    partial_df.query(f'Namespace =="{relevant_namespace}"', inplace=True)
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
    partial_df[HASH_COLUMN] = None
    partial_df[LANGUAGE_COLUMN] = None
    partial_df[NAMES_COLUMN] = None
    partial_df[NAMES_WITH_TYPES_COLUMN] = None
    partial_df['model_xmi'] = None
    partial_df['model_txt'] = None
    partial_df['category'] = None
    partial_df['tags'] = None
    return partial_df
