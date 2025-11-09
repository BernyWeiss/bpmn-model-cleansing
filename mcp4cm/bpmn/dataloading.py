import csv
import hashlib
import json
import os
from collections import deque
from csv import reader
from typing import Optional, List, Dict
from enum import Enum

import pandas as pd
from pydantic import ValidationError
from tqdm.auto import tqdm

import sapsam.parser
from mcp4cm.base import Model, Dataset
from mcp4cm.bpmn.data_extraction import extract_names_from_model

SAM_MODELS_PATH = 'sap_sam_2022/models'
PROCESSED_MODELS_PATH = 'processed'
CSV_FIELD_SIZE_LIMIT = 6000000


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


class BPMNDataset(Dataset):
    """
    Class representing a dataset of BPMN models.

    This class extends the base Dataset class to work specifically with
    BPMN models and provides BPMN-specific operations.

    Attributes:
        models (List[BPMNModel]): List of BPMN models in the dataset.
    """
    models: List[BPMNModel]

    def __getitem__(self, index: int) -> BPMNModel:
        """
        Get a UML model by index.

        Args:
            index (int): Index of the model to retrieve.

        Returns:
            UMLModel: The UML model at the specified index.
        """
        return self.models[index]

    @staticmethod
    def from_csv(file_path: str, model_name: str) -> 'BPMNDataset':
        df = pd.read_csv(file_path)
        models = []
        for df_record in df.to_dict('records'):
            model = BPMNModel.model_construct(**df_record)
            models.append(model)

        return BPMNDataset(name=model_name, models=models)


class Namespaces(Enum):
    """
    Enum for different Namespaces in the sap_sam_2022 dataset.

    This enumeration defines all namespaces of the dataset which are supported to load as a dataset.

    Currently only BPMN 2.0 models are supported.
    """
    BPMN2 = 'http://b3mn.org/stencilset/bpmn2.0#'


def load_dataset(
        dataset_path: str = 'bpmnmodelset/raw/sap_sam_2022',
        namespace: Namespaces = Namespaces.BPMN2,
) -> List[str]:
    """
    Load the sap_sam_2022 dataset.
    This function parses the models from the sap_sam_2022 csv files. It processes all models with the given namespace.
    It returns a BPMNDataset object.


    Args:
        dataset_path: path to the folder where the dataset is stored.
        namespace: Namespace of models to load.

    Returns:
        LIST[str]: List of strings with the paths to the created datasets.

    """
    data_path = os.path.join(dataset_path, SAM_MODELS_PATH)
    path_strings = list()

    # TODO: Make this loading possible for all data or lazy
    for models_file in tqdm(os.listdir(data_path), desc=f'Loading SAP SAM Dataset @ {data_path}'):
        if models_file.endswith('.csv'):
            dataset_name= f'BPMN2 models {models_file}'
            models: List[BPMNModel] = list()
            with open (f'{data_path}/{models_file}', encoding='utf-8') as csv_file:
                csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
                reader = csv.DictReader(csv_file, delimiter=',')

                for raw_model in reader:
                    if raw_model['Namespace'] != namespace.value:
                        continue
                    id = raw_model['Model ID']
                    file_path = os.path.join(data_path, models_file)

                    model_json = json.loads(raw_model['Model JSON'])

                    model_name = raw_model['Name']
                    hash = hashlib.sha256(raw_model['Model JSON'].encode()).hexdigest()

                    language = model_json['properties']['language'] if 'language' in model_json['properties'].keys()  else None

                    model = BPMNModel(
                        id = id,
                        name = model_name,
                        file_path = file_path,
                        model_json=model_json,
                        language=language,
                        hash=hash
                    )
                    model = extract_names_from_model(model, use_types=False)

                    models.append(model)
                dataset = BPMNDataset(name = dataset_name, models = models)
                processed_path = os.path.join(dataset_path, PROCESSED_MODELS_PATH, models_file)
                BPMNDataset.to_csv(dataset, fp=processed_path)
                path_strings.append(processed_path)


    return path_strings

