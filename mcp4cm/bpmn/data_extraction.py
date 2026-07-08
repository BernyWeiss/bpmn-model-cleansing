import json
from collections import deque, defaultdict, Counter
from functools import partial
from typing import List, Dict, Any, Optional

from bpmn.constants import LANGUAGE_COLUMN, CALL_ACTIVITY_COLUMN
from bpmn.filtering_patterns import SWIMLANE_PATTERN

from mcp4cm.bpmn.constants import EMPTY_NAME_TOKEN, EMPTY_TYPE_TOKEN, NAMES_COLUMN, NAMES_WITH_TYPES_COLUMN, ELEMENT_COUNTS_COLUMN, HASH_COLUMN
from mcp4cm.bpmn.dataloading.json_model import Shape
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.bpmn.filtering_patterns import ACTIVITY_PATTERN, DATA_OBJECT_PATTERN, EVENT_PATTERN
from mcp4cm.util.text_util import join_texts, get_file_hash
from mcp4cm._language_detector import _get_text_language
from tqdm.auto import tqdm

translation_table = str.maketrans({'\n': ' '})


def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = EMPTY_NAME_TOKEN,
                              include_texts: bool = False,
                              include_documentation: bool = False) -> None:
    column = NAMES_COLUMN
    if use_types:
        column = NAMES_WITH_TYPES_COLUMN

    name_extraction = partial(_extract_names_from_shape,
                              use_types=use_types,
                              empty_name_pattern=empty_name_pattern,
                              include_texts=include_texts,
                              include_documentation=include_documentation)

    dataset.models[column], dataset.models[ELEMENT_COUNTS_COLUMN], dataset.models[CALL_ACTIVITY_COLUMN] = zip(*dataset.models['model_json'].map(name_extraction))

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
    if key == NAMES_COLUMN:
        dataset.models[HASH_COLUMN] = dataset.models[key].apply(lambda texts: get_file_hash(json.dumps(texts)))
    elif key == NAMES_WITH_TYPES_COLUMN:
        full_representation = dataset.models[key].combine(dataset.models[ELEMENT_COUNTS_COLUMN], _combine_name_and_count_dicts)
        dataset.models[HASH_COLUMN] = full_representation.apply(lambda texts: get_file_hash(json.dumps(texts, sort_keys=True)))
    else:
        raise ValueError(f"Unknown key {key}")


def _extract_names_from_shape(model_json: List | Dict,
                              use_types: bool = False,
                              empty_name_pattern: str = EMPTY_NAME_TOKEN,
                              empty_type_pattern: str = EMPTY_TYPE_TOKEN,
                              include_texts: bool = False,
                              include_documentation: bool = False,
                              **_) -> tuple[list[str]|dict[str, list], dict]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_of_type_dict = defaultdict(list)
    n_call_activities = 0

    stack = deque([bpmn_model_shape])
    counter = Counter()
    while len(stack) > 0:
        element = stack.pop()
        for child in element.childShapes:
            stack.append(child)

        node_type, element_texts, is_call_activity = _extract_element_node_type_and_texts(element, empty_name_pattern, empty_type_pattern,
                                                                        include_documentation, include_texts, use_types)
        counter[node_type] += 1
        if use_types:
            names_of_type_dict[node_type].extend(element_texts)
            if is_call_activity:
                n_call_activities += 1
        else:
            names.extend(element_texts)

    counter = dict(counter) # convert to built-in dict to make serialization possible

    if use_types:
        for key, names_list in names_of_type_dict.items():
            names_of_type_dict[key] = sorted(names_list)
            # convert to built-in dict to make serialization possible
        return (dict(names_of_type_dict), counter, n_call_activities)
    else:
        return (sorted(names), counter, None)


def _extract_element_node_type_and_texts(element: Shape, empty_name_pattern: str, empty_type_pattern: str,
                                         include_documentation: bool, include_texts: bool, use_types: bool) -> tuple[
    str, list[Any], Optional[int]]:

    if element.stencil and element.stencil.id:
        node_type = element.stencil.id
    else:
        node_type = empty_type_pattern

    name, text, documentation, is_call_activity = None, None, None, None

    element_texts = []
    if element.properties:
        if element.properties.name is not None:
            name = _replace_linebreaks_and_strip(element.properties.name)
            if not name and _type_should_have_name(node_type):
                name = empty_name_pattern
            if name:
                element_texts.append(name)

        if element.properties.callactivity is not None:
            is_call_activity = element.properties.callactivity

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
    return node_type, element_texts, is_call_activity


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


def extract_dataset_languages(dataset: BPMNDataset, text_key: str = NAMES_COLUMN,
                              empty_name: str = EMPTY_NAME_TOKEN, override: bool = False) -> None:

    tqdm.pandas(desc='Language Extraction Progress')

    if override:
        dataset.models[LANGUAGE_COLUMN] = dataset.models[text_key].progress_apply(
            lambda text: _get_text_language(join_texts(text, empty_name=empty_name)))
        return


    models_without_language = dataset.models[dataset.models[LANGUAGE_COLUMN].isna()]
    models_without_language[LANGUAGE_COLUMN] = models_without_language[text_key].progress_apply(
        lambda  text: _get_text_language(join_texts(text, empty_name=empty_name)))

    dataset.models.update(models_without_language, join='left', overwrite=True, errors='ignore')



