from enum import Enum
from ast import literal_eval

import pandas as pd

from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.bpmn.dataloading.json_model import reduce_json_model
from mcp4cm.bpmn.dataloading.sap_sam import load_sap_sam_bpmn
from mcp4cm.bpmn.dataloading.bpmai import load_bpmai_bpmn

PROCESSED_MODELS_PATH = 'processed/reduced'
CSV_FIELD_SIZE_LIMIT = 6000000


class BPMNModelCollection(Enum):
    SAP_SAM = 'sap_sam'
    BPMAI = 'bpmai'


def load_bpmn_dataset(path: str, model_collection: BPMNModelCollection, reduced_size: bool) -> BPMNDataset:
    if model_collection.value == BPMNModelCollection.SAP_SAM.value:
        return load_sap_sam_bpmn(path, reduced_size=reduced_size)
    if model_collection.value == BPMNModelCollection.BPMAI.value:
        return load_bpmai_bpmn(path)
    raise ValueError(f"Could not load BPMNDataset: BPMNModelCollection is unknown: {model_collection.value}")


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
