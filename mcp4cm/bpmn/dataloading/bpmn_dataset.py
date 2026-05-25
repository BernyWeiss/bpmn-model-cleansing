import json
import shutil

import pandas as pd

from collections import Counter

from pathlib import Path
from typing import Optional, List

from pydantic import field_validator
from mcp4cm.base import Model, Dataset
from mcp4cm.bpmn.dataloading.sap_sam import SapSam2022Namespaces, _load_sap_sam_csv_to_df

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
    hash: Optional[str] = None
    name: Optional[str] = None
    names_with_types: Optional[List[str]] = None
    duplicate_group: Optional[str] = None
    element_counts: Optional[dict] = None

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
    def to_files(dataset: 'BPMNDataset', output_directory: str, include_svg: bool = False):
        directory_path = Path(output_directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        models = dataset.models.copy(deep=False)
        has_duplicates = models['file_path'].duplicated().any()
        if has_duplicates:
            models.sort_values(by=['file_path'], inplace=True)

        current_csv_name = ''
        current_csv_df = None

        for model_tupel in models.itertuples(index=False, name='BPMNModel'):
            model_path = Path(model_tupel.file_path)

            if model_path.name.endswith('.json'):
                json_file_path = model_path
                base_path = Path(json_file_path.parent)
                metadata_file_name = json_file_path.name.replace('.json', '.meta.json')
                metadata_file_path = base_path.joinpath(metadata_file_name)

                new_model_file_path = directory_path.joinpath(json_file_path.name)
                new_meta_file_path = directory_path.joinpath(metadata_file_path.name)
                shutil.copy2(json_file_path, new_model_file_path)
                shutil.copy2(metadata_file_path, new_meta_file_path)

                if include_svg:
                    svg_file_name = json_file_path.name.replace('.json', '.svg')
                    svg_file_path = base_path.joinpath(svg_file_name)

                    new_svc_file_path = directory_path.joinpath(svg_file_path.name)
                    shutil.copy2(svg_file_path, new_svc_file_path)
                continue

            if model_path.name.endswith('.csv'):
                csv_file_path = model_path

                if csv_file_path.name != current_csv_name:
                    current_csv_name = csv_file_path.name
                    current_csv_df = _load_sap_sam_csv_to_df(model_path,relevant_namespace=SapSam2022Namespaces.BPMN2, cull_json=False)

                # find entry in csv
                model_df_entry = current_csv_df.loc[model_tupel.id]
                new_model_file_path = directory_path.joinpath(f"{model_tupel.id}.json")
                new_meta_file_path = directory_path.joinpath(f"{model_tupel.id}.meta.json")

                # write new file.
                with open(new_model_file_path, 'w', encoding='utf-8') as json_file:
                    json_content = json.loads(model_df_entry['model_json'])
                    json.dump(json_content, json_file, ensure_ascii=False)


                full_metadata = {}
                model_metadata = {}

                model_metadata['modelId'] = model_tupel.id
                model_metadata['modelName'] = model_tupel.name
                model_metadata['naturalLanguage'] = model_tupel.language

                full_metadata['model'] = model_metadata

                with open(new_meta_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(full_metadata, json_file, ensure_ascii=False)
                continue
