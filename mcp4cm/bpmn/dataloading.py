import json
import os
from pathlib import Path

from typing import Optional, List
from enum import Enum
from ast import literal_eval

import pandas as pd
from pydantic import field_validator

from tqdm.auto import tqdm

from mcp4cm.base import Model, Dataset
from mcp4cm.bpmn.json_model import reduce_json_model
from mcp4cm.util.text_util import get_file_hash

SAM_MODELS_PATH = 'sap_sam_2022/models'
PROCESSED_MODELS_PATH = 'processed/reduced'
CSV_FIELD_SIZE_LIMIT = 6000000
BPMN_MODEL_COLUMNS = ['id','name','model_json','file_path','hash','language','names','names_with_types','model_xmi','model_txt','category','tags']


class BPMNModel(Model):
    """
    Class representing a BPMN model.

    This class extends the base Model class with BPMN-specific attributes
    and functionality.

    Attributes:
        names (Optional[List[str]]): Extracted element names from the model.
        names_with_types (Optional[List[str]]): Element names with their types
            (e.g., 'class: Customer', 'actor: User', etc.).
    """
    name: Optional[str] = None
    names_with_types: Optional[List[str]] = None
    duplicate_group: Optional[str] = None

    def __repr__(self):
        return f"({self.name}, {self.file_path})"

    def __str__(self):
        return f"({self.name}, {self.file_path})"


class BPMNDataset(Dataset):
    """
    Class representing a dataset of BPMN models.

    This class extends the base Dataset class to work specifically with
    BPMN models and provides BPMN-specific operations.

    Attributes:
        models (List[BPMNModel]| pandas.DataFrame): List of BPMN models in the dataset.
    """

    class Config:
        arbitrary_types_allowed = True

    models: pd.DataFrame = pd.DataFrame(columns=BPMN_MODEL_COLUMNS)

    @field_validator("models", mode="before")
    def convert_to_df(cls, models: List['BPMNModel'] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(models, list):
            return pd.DataFrame([model.model_dump() if isinstance(model, BPMNModel) else model for model in models])
        if isinstance(models, pd.DataFrame):
            return models
        raise TypeError("'models' must be a list of BPMNModels or a pd.DataFrame")

    def __getitem__(self, index: int) -> BPMNModel:
        """
        Get a BPMNModel by index.

        Args:
            index (int): Index of the model to retrieve.

        Returns:
            BPMNModel: The BPMN model at the specified index.
        """
        model = BPMNModel.model_validate(self.models.iloc[index])
        return model

    def __iter__(self):
        columns = list(self.models.columns)
        for df_row in self.models.itertuples(index=False, name=None):
            model_dict = dict(zip(columns, df_row))
            yield BPMNModel.model_validate(model_dict)

    @staticmethod
    def to_csv(dataset: 'BPMNDataset', fp: str):
        create_directories_for_path(fp)
        models_json_series = dataset.models["model_json"].apply(
            lambda model: json.dumps(model) if model is not None else None)
        models_copy = dataset.models.copy(deep=False)
        models_copy['model_json'] = models_json_series
        models_copy.to_csv(fp, index=False)


class SapSam2022Namespaces(Enum):
    """
    Enum for different Namespaces in the sap_sam_2022 dataset.

    This enumeration defines all namespaces of the dataset which are supported to load as a dataset.

    Currently only BPMN 2.0 models are supported.
    """
    BPMN2 = 'http://b3mn.org/stencilset/bpmn2.0#'


def _load_names(name: str):
    if not name:
        return None
    return literal_eval(name)


def load_dataset_from_csv(name: str, fp: str) -> BPMNDataset:
    models = pd.read_csv(fp, na_filter=False, converters={
        "model_json": lambda x: reduce_json_model(x) if x is not None else None,
        "names": lambda x: _load_names(x),
        "names_with_types": lambda x: _load_names(x),
    })
    models.replace("", None, inplace=True)
    return BPMNDataset(name=name, models=models)


def load_dataset(
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
        partial_df['hash'] = partial_df['model_json'].apply(_compute_hash_of_modeldict)

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
            if n_files_processed > 4:
                break

    full_dataset.reset_index(drop=False, inplace=True, names='id')
    full_dataset.fillna({'name': ''}, inplace=True)

    bpmn2_dataset = BPMNDataset(name="sapsam_2022_bpmn2", models=full_dataset)
    return bpmn2_dataset


def _compute_hash_of_modeldict(modeldict: dict) -> str:
    return get_file_hash(json.dumps(modeldict))


def create_directories_for_path(path: str):
    file_path = Path(path)
    directory_path = file_path.parent
    directory_path.mkdir(parents=True, exist_ok=True)
