from bpmn.constants import NAMES_COLUMN, LANGUAGE_COLUMN
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.util.text_util import join_texts
from mcp4cm._language_detector import _get_text_language
from functools import partial


def filter_models_by_language(dataset: BPMNDataset,
                              language: str,
                              key: str = NAMES_COLUMN,
                              empty_name: str | None = None
                              )->BPMNDataset:
    language_extraction = partial(_get_or_calculate_language, key=key, empty_name=empty_name)

    dataset.models[LANGUAGE_COLUMN] = dataset.models.apply(language_extraction, axis=1)

    return BPMNDataset(name=dataset.name, models=dataset.models[dataset.models.language == language])



def _get_or_calculate_language(row,key: str = NAMES_COLUMN,empty_name: str | None = None):
    if row[LANGUAGE_COLUMN] is not None:
        return row[LANGUAGE_COLUMN]
    return _get_text_language(join_texts(row[key], empty_name=empty_name))