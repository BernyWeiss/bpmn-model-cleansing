from typing import List

import numpy as np

from bpmn.dataloading.bpmn_dataset import BPMNDataset
from bpmn.filtering_patterns import MIN_ELEMENT_COUNT, MAX_ELEMENT_COUNT, MAX_EMPTY_NAME_PERCENTAGE, DUMMY_KEYWORDS, \
    DUMMY_WORD_THRESHOLD, MIN_MEDIAN_NAME_LENGTH


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