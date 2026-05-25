import json
from collections import deque, defaultdict, Counter
from functools import partial
from typing import List, Dict, Any

import numpy as np

from mcp4cm.bpmn.filtering_patterns import (MIN_ELEMENT_COUNT,
                                            MAX_ELEMENT_COUNT,
                                            MAX_EMPTY_NAME_PERCENTAGE,
                                            DUMMY_WORD_THRESHOLD,
                                            DUMMY_KEYWORDS, MIN_MEDIAN_NAME_LENGTH)
from mcp4cm.bpmn.dataloading.json_model import Shape
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
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

    #dataset.models[[column, 'element_counts']] = dataset.models['model_json'].apply(_extract_names_from_shape,
    #                                   args=(use_types, empty_name_pattern, 'unknown type',
    #                                         include_texts, include_documentation), result_type='expand')

    name_extraction = partial(_extract_names_from_shape,
                              use_types=use_types,
                              empty_name_pattern=empty_name_pattern,
                              include_texts=include_texts,
                              include_documentation=include_documentation)

    #dataset.models[[column, 'element_counts']] = dataset.models['model_json'].apply(name_extraction, result_type='expand')

    dataset.models[column], dataset.models['element_counts'] = zip(*dataset.models['model_json'].map(name_extraction))


def calculate_model_hashes(dataset: BPMNDataset,
                           key) -> None:
    dataset.models['hash'] = dataset.models[key].apply(lambda texts: get_file_hash(json.dumps(texts, sort_keys=True)))


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
    if node_type.endswith('Flow'):
        return False
    if node_type.endswith('Gateway'):
        return False
    if node_type.startswith('Association'):
        return False
    return True


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



def filter_empty_models(dataset: BPMNDataset, key: str = 'names', inplace: bool = False,
                        empty_name: str = "empty name") -> BPMNDataset:
    empty_models = dataset.models[key].apply(lambda names: all(name == empty_name for name in names))
    non_empty_models = ~empty_models
    print(f'Filtered out models where every element is unnamed: {sum(empty_models)}')
    if inplace:
        dataset.models = dataset.models[non_empty_models]
        return dataset
    return BPMNDataset(name=dataset.name, models=dataset.models[non_empty_models])

def filter_models_by_min_element_count(
    dataset: BPMNDataset,
    min_count: int = MIN_ELEMENT_COUNT,
    inplace: bool = False,
) -> BPMNDataset:
    """
    Filter models based on the number of named elements they contain.

    This function filters the dataset to include only models that have a name count
    within the specified range. Models with too few names might be incomplete,
    while models with too many names might be overly complex or auto-generated.

    Args:
        dataset (UMLDataset): The dataset to filter.
        min_count (int): The minimum number of names a model should have. Defaults to 25.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an appropriate number of names.

    Example:
        >>> filtered_dataset = filter_models_by_name_count(dataset, min_count=50)
        >>> print(f"Kept {len(filtered_dataset.models)} models with appropriate complexity")
    """
    # TODO: Update Documentation
    n_models_before = len(dataset)
    models = dataset.models

    models['element_count'] = models['names'].str.len()
    models.query(f'element_count >= {min_count}', inplace=inplace)
    models.drop(columns=['element_count'], inplace=True)

    print(
        f"Filtered out models with element counts smaller than {min_count}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def filter_models_by_max_element_count(
    dataset: BPMNDataset,
    max_count: int = MAX_ELEMENT_COUNT,
    inplace: bool = False,
) -> BPMNDataset:
    """
    Filter models based on the number of named elements they contain.

    This function filters the dataset to include only models that have a name count
    within the specified range. Models with too few names might be incomplete,
    while models with too many names might be overly complex or auto-generated.

    Args:
        dataset (UMLDataset): The dataset to filter.
        min_count (int): The minimum number of names a model should have. Defaults to 25.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an appropriate number of names.

    Example:
        >>> filtered_dataset = filter_models_by_name_count(dataset, min_count=50)
        >>> print(f"Kept {len(filtered_dataset.models)} models with appropriate complexity")
    """
    # TODO: Update Documentation
    n_models_before = len(dataset)
    models = dataset.models

    models['element_count'] = models['names'].str.len()
    models.query(f'element_count <= {max_count}', inplace=inplace)
    models.drop(columns=['element_count'], inplace=True)

    print(
        f"Filtered out models with element counts greater than {max_count}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def filter_models_by_element_count(
    dataset: BPMNDataset,
    min_count: int = MIN_ELEMENT_COUNT,
    max_count: int = MAX_ELEMENT_COUNT,
    inplace: bool = False,
) -> BPMNDataset:
    """
    Filter models based on the number of named elements they contain.

    This function filters the dataset to include only models that have a name count
    within the specified range. Models with too few names might be incomplete,
    while models with too many names might be overly complex or auto-generated.

    Args:
        dataset (UMLDataset): The dataset to filter.
        min_count (int): The minimum number of names a model should have. Defaults to 25.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an appropriate number of names.

    Example:
        >>> filtered_dataset = filter_models_by_name_count(dataset, min_count=50)
        >>> print(f"Kept {len(filtered_dataset.models)} models with appropriate complexity")
    """
    # TODO: Update Documentation
    n_models_before = len(dataset)
    models = dataset.models

    models['element_count'] = models['names'].str.len()
    models.query(f'element_count >= {min_count} and element_count <= {max_count}', inplace=inplace)
    models.drop(columns=['element_count'], inplace=True)

    print(
        f"Filtered out models with element counts outside of {min_count} and {max_count}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def filter_models_by_empty_name_percentage(
        dataset: BPMNDataset,
        empty_name_percentage: float = MAX_EMPTY_NAME_PERCENTAGE,
        empty_name: str = "empty name",
        inplace: bool = False,
) -> BPMNDataset:
    """

    Args:
        empty_name:
        dataset:
        empty_name_percentage:
        inplace:

    Returns:

    """
    n_models_before = len(dataset)
    models = dataset.models
    models['element_count'] = models['names'].str.len()
    models['empty_name_count'] = models['names'].apply(lambda names: len([name for name in names if name==empty_name]))
    models['empty_name_percentage'] = models['empty_name_count'] / models['element_count']
    models.query(f'empty_name_percentage <= {empty_name_percentage}', inplace=inplace)

    models.drop(columns=['element_count','empty_name_count','empty_name_percentage'], inplace=True)

    print(
        f"Filtered out models with a empty name percentage higher than {empty_name_percentage}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)


def filter_models_by_dummy_words(
        dataset: BPMNDataset,
        dummy_keywords: List[str] = DUMMY_KEYWORDS,
        dummy_word_threshold: float = DUMMY_WORD_THRESHOLD,
        inplace: bool = False,
) -> BPMNDataset:
    n_models_before = len(dataset)
    models = dataset.models
    models['element_count'] = models['names'].str.len()
    models['dummy_word_count'] = models['names'].apply(lambda names: sum([1 for name in names if name.lower() in dummy_keywords]))
    models['dummy_percentage'] = models['dummy_word_count'] / models['element_count']
    models.query(f'dummy_percentage <= {dummy_word_threshold}', inplace=inplace)

    models.drop(columns=['element_count','dummy_word_count','dummy_percentage'], inplace=True)

    print(
        f"Filtered out models with a dummy_percentage higher than {dummy_word_threshold}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def filter_models_by_median_name_lengh(
        dataset: BPMNDataset,
        min_median_length: int = MIN_MEDIAN_NAME_LENGTH,
        inplace: bool = False
) -> BPMNDataset:
    n_models_before = len(dataset)
    models = dataset.models
    models['median_name_lengh'] = models['names'].apply(lambda names: np.median([len(name) for name in names]))
    models.query(f'median_name_lengh >= {min_median_length}', inplace=inplace)
    models.drop(columns=['median_name_lengh'], inplace=True)
    print(f"Filtered out models with a median name length smaller than {min_median_length}: {n_models_before - len(models)}")
    return BPMNDataset(name=dataset.name, models=models)
