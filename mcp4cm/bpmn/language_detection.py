from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.generic.utils import join_texts, get_text_language
from functools import partial


def filter_models_by_language(dataset: BPMNDataset,
                              language: str,
                              key: str = 'names',
                              empty_name: str | None = None
                              )->BPMNDataset:
    language_extraction = partial(_get_or_calculate_language, key=key, empty_name=empty_name)

    dataset.models['language'] = dataset.models.apply(language_extraction, axis=1)

    return BPMNDataset(name=dataset.name, models=dataset.models[dataset.models.language == language])



def _get_or_calculate_language(row,key: str = 'names',empty_name: str | None = None):
    if row['language'] is not None:
        return row['language']
    return get_text_language(join_texts(row[key], empty_name=empty_name))