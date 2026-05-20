import json
import os
import shutil

import pandas as pd

from pathlib import Path
from typing import Optional, List

from pydantic import field_validator
from mcp4cm.base import Model, Dataset

BPMN_MODEL_COLUMNS = ['id', 'name', 'model_json', 'file_path', 'hash', 'language', 'names', 'names_with_types',
                      'model_xmi', 'model_txt', 'category', 'tags']


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
        models (List[mcp4cm.bpmn.dataloading.bpmn_dataset.BPMNModel]| pandas.DataFrame): List of BPMN models in the dataset.
    """

    class Config:
        arbitrary_types_allowed = True

    models: pd.DataFrame = pd.DataFrame(columns=BPMN_MODEL_COLUMNS)

    @field_validator("models", mode="before")
    def convert_to_df(cls, models: List['BPMNModel'] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(models, list):
            return pd.DataFrame(
                [model.model_dump(exclude=['duplicate_group']) if isinstance(model, BPMNModel) else model for model in
                 models])
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
        file_path = Path(fp)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        models_json_series = dataset.models["model_json"].apply(
            lambda model: json.dumps(model) if model is not None else None)
        models_copy = dataset.models.copy(deep=False)
        models_copy['model_json'] = models_json_series
        models_copy.to_csv(fp, index=False)

    @staticmethod
    def to_files(dataset: 'BPMNDataset', output_directory: str):
        directory_path = Path(output_directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        models = dataset.models.copy(deep=False)
        has_duplicates = models['file_path'].duplicated().any()
        if has_duplicates:
            models.sort_values(by=['file_path'], inplace=True)
        # TODO: Implement export for csv files.
        # load first csv file
        # for filepath in model
        # if filepath != loaded file - load new file
        # find entry in csv
        # transform entry (with additional information from model) to json export format
        # write new file.
        # write new metadate file
        #
        for model_tupel in models.itertuples(index=False, name='BPMNModel'):

            json_file_path = Path(model_tupel.file_path)

            base_path = Path(json_file_path.parent)
            metadata_file_name = json_file_path.name.replace('.json', '.meta.json')

            metadata_file_path = base_path.joinpath(metadata_file_name)

            if json_file_path.name.endswith('.json'):
                new_file_path = directory_path.joinpath(json_file_path.name)
                new_meta_file_path = directory_path.joinpath(metadata_file_path.name)
                shutil.copy2(json_file_path, new_file_path)
                shutil.copy2(metadata_file_path, new_meta_file_path)

            if json_file_path.name.endswith('.csv'):
                # TODO: implement export from csv
                # Find
                raise NotImplementedError






