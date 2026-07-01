import json
from collections import deque, defaultdict, Counter
from functools import partial
from typing import List, Dict, Any

from bpmn.filtering_patterns import SWIMLANE_PATTERN
from mcp4cm.bpmn.dataloading.json_model import Shape
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.bpmn.filtering_patterns import ACTIVITY_PATTERN, DATA_OBJECT_PATTERN, EVENT_PATTERN
from mcp4cm.util.text_util import join_texts, get_file_hash
from mcp4cm._language_detector import _get_text_language
from tqdm.auto import tqdm

translation_table = str.maketrans({'\n': ' '})


def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name",
                              include_texts: bool = False,
                              include_documentation: bool = False) -> None:
    column = 'names'
    if use_types:
        column = 'names_with_types'

    name_extraction = partial(_extract_names_from_shape,
                              use_types=use_types,
                              empty_name_pattern=empty_name_pattern,
                              include_texts=include_texts,
                              include_documentation=include_documentation)

    dataset.models[column], dataset.models['element_counts'] = zip(*dataset.models['model_json'].map(name_extraction))

def _combine_name_and_count_dicts(name_dict: dict, type_counter: dict) -> dict:
    combined_dict = {}

    for name, count in type_counter.items():
        names = []
        if name in name_dict:
            names = name_dict[name]

        combined_dict[name] = {'count': count, 'names': names}

    return combined_dict



def calculate_model_hashes(dataset: BPMNDataset,
                           key) -> None:
    if key == 'names':
        dataset.models['hash'] = dataset.models[key].apply(lambda texts: get_file_hash(json.dumps(texts)))
    elif key == 'names_with_types':
        full_representation = dataset.models[key].combine(dataset.models['element_counts'], _combine_name_and_count_dicts)
        dataset.models['hash'] = full_representation.apply(lambda texts: get_file_hash(json.dumps(texts, sort_keys=True)))
    else:
        raise ValueError(f"Unknown key {key}")



def _extract_names_from_shape(model_json: List | Dict,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name",
                              empty_type_pattern: str = "unknown type",
                              include_texts: bool = False,
                              include_documentation: bool = False,
                              **_) -> tuple[list[str]|dict[str, list], dict]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_of_type_dict = defaultdict(list)

    stack = deque([bpmn_model_shape])
    counter = Counter()
    while len(stack) > 0:
        element = stack.pop()
        for child in element.childShapes:
            stack.append(child)

        node_type, element_texts = _extract_element_node_type_and_texts(element, empty_name_pattern, empty_type_pattern,
                                                                        include_documentation, include_texts, use_types)
        counter[node_type] += 1
        if use_types:
            names_of_type_dict[node_type].extend(element_texts)
        else:
            names.extend(element_texts)

    counter = dict(counter) # convert to built-in dict to make serialization possible

    if use_types:
        for key, names_list in names_of_type_dict.items():
            names_of_type_dict[key] = sorted(names_list)
            # convert to built-in dict to make serialization possible
        return (dict(names_of_type_dict), counter)
    else:
        return (sorted(names), counter)


def _extract_element_node_type_and_texts(element: Shape, empty_name_pattern: str, empty_type_pattern: str,
                                         include_documentation: bool, include_texts: bool, use_types: bool) -> tuple[
    str, list[Any]]:

    if element.stencil and element.stencil.id:
        node_type = element.stencil.id
    else:
        node_type = empty_type_pattern

    name, text, documentation = None, None, None

    element_texts = []
    if element.properties:
        if element.properties.name:
            name = _replace_linebreaks_and_strip(element.properties.name)

            if use_types or _type_should_have_name(node_type):
                name = name or empty_name_pattern

            if name:
                element_texts.append(name)

        if include_texts:
            if element.properties.text:
                text = _replace_linebreaks_and_strip(element.properties.text)
                text = f"text: {text}" if text else None

            if text:
                element_texts.append(text)
        if include_documentation:
            if element.properties.documentation:
                documentation = _replace_linebreaks_and_strip(element.properties.documentation)
                documentation = f"documentation: {documentation}" if documentation else None
            if documentation:
                element_texts.append(documentation)
    return node_type, element_texts


def _replace_linebreaks_and_strip(string: str) -> str:
    name = string.strip()
    name = name.translate(translation_table)
    return name


def _type_should_have_name(node_type: str) -> bool:
    match = ACTIVITY_PATTERN.fullmatch(node_type)
    if match is not None:
        return True
    match = EVENT_PATTERN.fullmatch(node_type)
    if match is not None:
        return True
    match = DATA_OBJECT_PATTERN.fullmatch(node_type)
    if match is not None:
        return True
    match = SWIMLANE_PATTERN.fullmatch(node_type)
    if match is not None:
        return True

    return False


def extract_dataset_languages(dataset: BPMNDataset, text_key: str = 'names',
                              empty_name: str = "empty name", override: bool = False) -> None:

    language_column = 'language'
    tqdm.pandas(desc='Language Extraction Progress')

    if override:
        dataset.models[language_column] = dataset.models[text_key].progress_apply(
            lambda text: _get_text_language(join_texts(text, empty_name=empty_name)))
        return


    models_without_language = dataset.models[dataset.models[language_column].isna()]
    models_without_language[language_column] = models_without_language[text_key].progress_apply(
        lambda  text: _get_text_language(join_texts(text, empty_name=empty_name)))

    dataset.models.update(models_without_language, join='left', overwrite=True, errors='raise')



