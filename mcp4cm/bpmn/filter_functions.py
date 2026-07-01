import re
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from bpmn.dataloading.bpmn_dataset import BPMNDataset
from bpmn.filtering_patterns import MIN_ELEMENT_COUNT, MAX_ELEMENT_COUNT, MAX_EMPTY_NAME_PERCENTAGE, DUMMY_KEYWORDS, \
    DUMMY_WORD_THRESHOLD, MIN_MEDIAN_NAME_LENGTH, MINIMAL_ELEMENTS_DICT, DUPLICATE_ACTIVITY_NAME_THRESHOLD

from bpmn.filtering_patterns import (ACTIVITY_PATTERN,
                                            START_EVENT_PATTERN,
                                            END_EVENT_PATTERN,
                                            SEQUENCE_FLOW_PATTERN,
                                            EMPTY_NAME_PATTERN)

def filter_empty_models(dataset: BPMNDataset,
                        inplace: bool = False,
                        empty_name: str = "empty name") -> BPMNDataset:

    extracted_name_column = _get_extracted_name_column(dataset)

    if extracted_name_column == 'names_with_types':
        empty_models = dataset.models[extracted_name_column].apply(lambda type_names_dict: all(name == empty_name for names in type_names_dict.values() for name in names ))

    if extracted_name_column == 'names':
        empty_models = dataset.models[extracted_name_column].apply(lambda names: all(name == empty_name for name in names))

    non_empty_models = ~empty_models
    print(f'Filtered out models where every element is unnamed: {sum(empty_models)}')
    if inplace:
        dataset.models = dataset.models[non_empty_models]
        return dataset
    return BPMNDataset(name=dataset.name, models=dataset.models[non_empty_models])


def _get_extracted_name_column(dataset: BPMNDataset) -> str:
    filled_name_col = 'names_with_types' if dataset.models['names_with_types'].notna().any() else 'names'
    return filled_name_col


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
    total_count_col_name = 'total_element_count'

    models[total_count_col_name] = models['element_counts'].apply(_calculate_total_element_count)

    models.query(f'{total_count_col_name} >= {min_count}', inplace=inplace)
    models.drop(columns=[total_count_col_name], inplace=True)

    print(
        f"Filtered out models with element counts smaller than {min_count}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def _calculate_total_element_count(element_count_dict: dict) -> int:
    return sum(element_count_dict.values())

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

    n_models_before = len(dataset)
    models = dataset.models
    total_count_col_name = 'total_element_count'

    models[total_count_col_name] = models['element_counts'].apply(_calculate_total_element_count)

    models.query(f'{total_count_col_name} <= {max_count}', inplace=inplace)
    models.drop(columns=[total_count_col_name], inplace=True)

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


    n_models_before = len(dataset)
    models = dataset.models
    total_count_col_name = 'total_element_count'

    models[total_count_col_name] = models['element_counts'].apply(_calculate_total_element_count)

    models.query(f'{total_count_col_name} >= {min_count} and {total_count_col_name} <= {max_count}', inplace=inplace)

    models.drop(columns=[total_count_col_name], inplace=True)

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
    extracted_name_column = _get_extracted_name_column(dataset)
    models = dataset.models

    if extracted_name_column == 'names':
        models['name_count'] = models['names'].str.len()
        models['empty_name_count'] = models['names'].apply(
            lambda names: len([name for name in names if name == empty_name]))


    if extracted_name_column == 'names_with_types':

        models['name_count'] = models['names_with_types'].apply(_calculate_typed_name_count)
        models['empty_name_count'] = models['names_with_types'].apply(_calculate_typed_empty_name_counts, empty_name=empty_name)


    models['empty_name_percentage'] = models['empty_name_count'] / models['name_count']
    models.query(f'empty_name_percentage <= {empty_name_percentage}', inplace=inplace)

    models.drop(columns=['name_count','empty_name_count','empty_name_percentage'], inplace=True)

    print(
        f"Filtered out models with a empty name percentage higher than {empty_name_percentage}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)

def _calculate_typed_name_count(names_with_types_dict: dict) -> int:
    count = 0

    for key in names_with_types_dict.keys():
        count += len(names_with_types_dict[key])

    return count

def _calculate_typed_empty_name_counts(names_with_types_dict: dict, empty_name: str) -> int:
    count = 0

    for key in names_with_types_dict.keys():
        count += len([name for name in names_with_types_dict[key] if name == empty_name])
    return count




def _calculate_typed_dummy_name_percentage(names_with_types_dict: dict[str, list[str]], dummy_names: set[str]) -> float:
    element_count = 0
    dummy_word_count = 0

    for _, names in names_with_types_dict.items():
        for name in names:
            element_count += 1
            if name.casefold() in dummy_names:
                dummy_word_count += 1

    if element_count == 0:
        return 1

    return dummy_word_count / element_count

def filter_models_by_dummy_words(
        dataset: BPMNDataset,
        dummy_keywords: set[str] = DUMMY_KEYWORDS,
        dummy_word_threshold: float = DUMMY_WORD_THRESHOLD,
        inplace: bool = False,
) -> BPMNDataset:
    result_column_name = 'dummy_percentage'
    n_models_before = len(dataset)
    extracted_name_column = _get_extracted_name_column(dataset)
    models = dataset.models

    if extracted_name_column == 'names':
        models['element_count'] = models[extracted_name_column].str.len()
        models['dummy_word_count'] = models[extracted_name_column].apply(lambda names: sum([1 for name in names if name.casefold() in dummy_keywords]))
        models[result_column_name] = models['dummy_word_count'] / models['element_count']
        models.query(f'{result_column_name} <= {dummy_word_threshold}', inplace=inplace)
        models.drop(columns=['element_count', 'dummy_word_count', result_column_name], inplace=True)

    if extracted_name_column == 'names_with_types':
        dummy_percentage_extraction_fn = partial(_calculate_typed_dummy_name_percentage, dummy_names=DUMMY_KEYWORDS)
        models[result_column_name] = models[extracted_name_column].apply(dummy_percentage_extraction_fn)
        models.query(f'{result_column_name} <= {dummy_word_threshold}', inplace=inplace)
        models.drop(columns=[result_column_name], inplace=True)

    print(
        f"Filtered out models with a dummy_percentage higher than {dummy_word_threshold}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)


def filter_models_by_median_name_length(
        dataset: BPMNDataset,
        min_median_length: int = MIN_MEDIAN_NAME_LENGTH,
        inplace: bool = False
) -> BPMNDataset:

    n_models_before = len(dataset)
    extracted_name_column = _get_extracted_name_column(dataset)
    models = dataset.models

    if extracted_name_column == 'names':
        models['median_name_length'] = models[extracted_name_column].apply(lambda names: np.median([len(name) for name in names]))

    if extracted_name_column == 'names_with_types':
        models['median_name_length'] = models[extracted_name_column].apply(_calculate_typed_median_name_length)



    models.query(f'median_name_length >= {min_median_length}', inplace=inplace)
    models.drop(columns=['median_name_length'], inplace=True)
    print(f"Filtered out models with a median name length smaller than {min_median_length}: {n_models_before - len(models)}")
    return BPMNDataset(name=dataset.name, models=models)

def _calculate_typed_median_name_length(names_with_types_dict: dict):
    median_length = np.median([len(name) for name_list in names_with_types_dict.values() for name in name_list])
    return median_length


def _extract_count_of_pattern(counts: dict, pattern: re.Pattern):
    count = 0
    for key in counts.keys():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        count += counts[key]
    return count

def _validate_minimal_elements(row, min_counts: dict):
    valid_activities = row['n_activities'] >= min_counts['Activity']
    valid_start_events = row['n_start_events'] >= min_counts['StartEvent']
    valid_end_events = row['n_end_events'] >= min_counts['EndEvent']
    valid_sequence_flows = row['n_sequence_flows'] >= min_counts['SequenceFlow']

    return valid_activities and valid_start_events and valid_end_events and valid_sequence_flows


def filter_models_by_required_elements(
        dataset: BPMNDataset,
        minimal_elements_dict: dict = MINIMAL_ELEMENTS_DICT,
        inplace: bool = False
) -> BPMNDataset:
    models = dataset.models

    activity_extraction = partial(_extract_count_of_pattern, pattern=ACTIVITY_PATTERN)
    start_event_extraction = partial(_extract_count_of_pattern, pattern=START_EVENT_PATTERN)
    end_event_extraction = partial(_extract_count_of_pattern, pattern=END_EVENT_PATTERN)
    sequence_flow_extraction = partial(_extract_count_of_pattern, pattern=SEQUENCE_FLOW_PATTERN)

    models['n_activities'] = models['element_counts'].apply(activity_extraction)
    models['n_start_events'] = models['element_counts'].apply(start_event_extraction)
    models['n_end_events'] = models['element_counts'].apply(end_event_extraction)
    models['n_sequence_flows'] = models['element_counts'].apply(sequence_flow_extraction)

    minimal_element_validation = partial(_validate_minimal_elements, min_counts=minimal_elements_dict)
    models['has_minimal_elements'] = models.apply(minimal_element_validation, axis=1)

    total_models = len(models['has_minimal_elements'])
    valid_models = sum(models['has_minimal_elements'])
    invalid_models = total_models - valid_models

    models.query(f'has_minimal_elements', inplace=inplace)

    models.drop(columns=['n_activities', 'n_start_events', 'n_end_events', 'n_sequence_flows', 'has_minimal_elements'], inplace=True)
    print(f"Filtered out models which do not have minimally required elements: {invalid_models}")
    return BPMNDataset(name=dataset.name, models=models)

def _extract_names_of_pattern(names_with_types: dict, pattern: re.Pattern, empty_name: str = 'empty name'):
    relevant_names = []
    for key in names_with_types.keys():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        names_of_types = names_with_types[key]
        names_without_empty = [name for name in names_of_types if name != empty_name]
        relevant_names.extend(names_without_empty)
    return relevant_names

def _calculate_activity_name_duplicates(name_type_dict: dict[str, list[str]]) -> float:
    activity_names = _extract_names_of_pattern(name_type_dict, pattern=ACTIVITY_PATTERN)
    activity_names = pd.Series(activity_names)
    duplicated = activity_names.duplicated(keep=False)

    if len(duplicated) == 0:
        return 0.0
    percentage = sum(duplicated) / len(duplicated)

    return percentage

def filter_models_by_duplicate_activities(
        dataset: BPMNDataset,
        duplicate_activity_threshold: float = DUPLICATE_ACTIVITY_NAME_THRESHOLD,
        inplace: bool = False
) -> BPMNDataset:
    name_column = _get_extracted_name_column(dataset)
    if not name_column == 'names_with_types':
        raise ValueError("This analysis can only be done if name and types were extracted.")
    models = dataset.models

    name_duplicate_calculation = partial(_calculate_activity_name_duplicates)

    models['duplicate_activity_names_percentage'] = models['names_with_types'].apply(name_duplicate_calculation)

    n_models_before = len(models['duplicate_activity_names_percentage'])
    models.query(f'duplicate_activity_names_percentage <= {duplicate_activity_threshold}', inplace=inplace)

    models.drop(columns=['duplicate_activity_names_percentage'], inplace=True)
    print(
        f"Filtered out models with duplicate activity names higher than {duplicate_activity_threshold}: {n_models_before - len(models)}"
    )
    return BPMNDataset(name=dataset.name, models=models)


