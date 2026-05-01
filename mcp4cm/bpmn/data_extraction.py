from collections import deque
from functools import partial
from typing import List, Dict

from mcp4cm.bpmn.filtering_patterns import (MIN_ELEMENT_COUNT,
                                            MAX_ELEMENT_COUNT,
                                            MAX_EMPTY_NAME_PERCENTAGE,
                                            DUMMY_WORD_THRESHOLD,
                                            DUMMY_KEYWORDS)
from mcp4cm.bpmn.json_model import Shape
from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.generic.utils import join_texts, get_text_language
from tqdm.auto import tqdm

translation_table = str.maketrans({'\n': ' '})


def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> None:
    column = 'names'
    if use_types:
        column = 'names_with_types'

    name_extraction = partial(_extract_names_from_shape, use_types=use_types, empty_name_pattern=empty_name_pattern)

    dataset.models[column] = dataset.models['model_json'].apply(name_extraction)

    print(f"Extracting {column} from raw model done.")


def _extract_names_from_shape(model_json: List | Dict,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> list[str]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_with_types = list()

    stack = deque([bpmn_model_shape])

    if use_types:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern

            if element.stencil:
                if element.stencil.id:
                    names_with_types.append(f"{element.stencil.id}: {name}")
            else:
                names_with_types.append(f"unknown type: {name}")
        return names_with_types

    else:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)

            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                names.append(name)
            else:
                names.append(empty_name_pattern)
        return names


def extract_model_languages(dataset: BPMNDataset, key: str = 'names', empty_name: str = "empty name"):
    tqdm.pandas(desc='Language Extraction Progress')
    dataset.models['language'] = dataset.models[key].progress_apply(
        lambda text: get_text_language(join_texts(text, empty_name=empty_name)))


def filter_empty_models(dataset: BPMNDataset, key: str = 'names', inplace: bool = False,
                        empty_name: str = "empty name") -> BPMNDataset:
    empty_models = dataset.models[key].apply(lambda names: all(name == empty_name for name in names))
    non_empty_models = ~empty_models
    print(f'Found models with empty names: {sum(empty_models)}')
    if inplace:
        dataset.models = dataset.models[non_empty_models]
        return dataset
    return BPMNDataset(name=dataset.name, models=dataset.models[non_empty_models])

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
