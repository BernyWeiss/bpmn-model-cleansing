import re
from collections import deque, Counter
from functools import partial
from typing import List, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from mcp4cm.bpmn.dataloading.json_model import Shape
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm.bpmn.filter_functions import _get_extracted_name_column

from mcp4cm.bpmn.data_extraction import translation_table

from mcp4cm.bpmn.filtering_patterns import (ACTIVITY_PATTERN,
                                            START_EVENT_PATTERN,
                                            END_EVENT_PATTERN,
                                            SEQUENCE_FLOW_PATTERN,
                                            EMPTY_NAME_PATTERN)




def get_all_names(dataset: BPMNDataset, key: str = 'names', casefold: bool = False) -> Iterable[str]:
    text_series = dataset.models[key].explode(ignore_index=True).dropna()
    if casefold:
        text_series = text_series.str.casefold()
    return text_series

def get_counts(dataset: BPMNDataset, key: str = 'names', casefold: bool = False):

    text_series = get_all_names(dataset, key, casefold)

    counts = text_series.value_counts()

    counts.sort_values(ascending=False, inplace=True)

    return counts



def extract_names_from_models(dataset: BPMNDataset,
                              use_types: bool = False,
                              empty_name_pattern: str = "empty name") -> None:
    column = 'names'
    if use_types:
        column = 'names_with_types'
    empty_counter = Counter()
    set_counter = Counter()
    text_counter = Counter()
    docu_counter = Counter()
    name_extraction = partial(_extract_names_from_shape_with_counts, set_counter=set_counter, empty_counter=empty_counter, text_counter=text_counter, docu_counter=docu_counter, use_types=use_types, empty_name_pattern=empty_name_pattern)
    dataset.models[column] = dataset.models['model_json'].apply(name_extraction)

    print("Text counter")
    print(text_counter)
    print("Documentation counter")
    print(docu_counter)
    print(f"Extracting {column} from raw model done.")
    print("Empty Counter")
    print(empty_counter)
    print("Set Counter")
    print(set_counter)
    print("Net names")
    set_counter.subtract(empty_counter)
    print(set_counter)


def _extract_names_from_shape_with_counts(model_json: List | Dict,
                                          empty_counter: Counter,
                                          set_counter: Counter,
                                          text_counter: Counter,
                                          docu_counter: Counter,
                                          use_types: bool = False,
                                          empty_name_pattern: str = "empty name",
                                          empty_type_pattern: str = "unknown type") -> list[str]:
    bpmn_model_shape = Shape.model_validate(model_json)
    names = list()
    names_with_types = list()

    stack = deque([bpmn_model_shape])

    if use_types:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            name = None
            if element.stencil and element.stencil.id:
                node_type = element.stencil.id
            else:
                node_type = empty_type_pattern
            if element.properties:
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                    empty_counter[node_type] += 1
                else:
                    set_counter[node_type] += 1

                if element.properties.documentation:
                    docu = element.properties.documentation.strip()
                    docu = docu.translate(translation_table)
                    if docu:
                        docu_counter[node_type] += 1

                if element.properties.text:
                    text = element.properties.text.strip()
                    text = text.translate(translation_table)
                    if text:
                        print(text)
                        text_counter[node_type] += 1


            else:
                empty_counter[node_type] +=1
                name = empty_name_pattern

            names_with_types.append(f"{node_type}: {name}")
        return names_with_types

    else:
        while len(stack) > 0:
            element = stack.pop()
            for child in element.childShapes:
                stack.append(child)
            node_type = (element.stencil and element.stencil.id) or empty_type_pattern
            if element.properties:
                name = None
                if element.properties.name:
                    name = element.properties.name.strip()
                    name = name.translate(translation_table)
                if not name:
                    name = empty_name_pattern
                    empty_counter[node_type] += 1
                else:
                    set_counter[node_type] += 1
                names.append(name)

                if element.properties.documentation:
                    docu = element.properties.documentation.strip()
                    docu = docu.translate(translation_table)
                    if docu:
                        docu_counter[node_type] += 1

                if element.properties.text:
                    text = element.properties.text.strip()
                    text = text.translate(translation_table)
                    if text:
                        text_counter[node_type] += 1
                        # if name == empty_name_pattern:
                        #     docu_counter[node_type] += 1

            else:
                empty_counter[node_type] +=1
                names.append(empty_name_pattern)
        return names

def create_wordcloud(counts):
    wordcloud = WordCloud(max_words=500, max_font_size=40, width=600, height=300).generate_from_frequencies(counts)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")


def print_all_types(dataset: BPMNDataset):
    models = dataset.models
    models['types'] = models['element_counts'].apply(lambda counts_dict: counts_dict.keys())
    all_types = models['types'].explode(ignore_index=True).unique()
    print(all_types.tolist())


def _extract_count_of_pattern(counts: dict, pattern: re.Pattern):
    count = 0
    for key in counts.keys():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        count += counts[key]
    return count

def extract_required_counts(dataset: BPMNDataset):
    models = dataset.models
    activity_extraction = partial(_extract_count_of_pattern, pattern=ACTIVITY_PATTERN)
    start_event_extraction = partial(_extract_count_of_pattern, pattern=START_EVENT_PATTERN)
    end_event_extraction = partial(_extract_count_of_pattern, pattern=END_EVENT_PATTERN)
    sequence_flow_extraction = partial(_extract_count_of_pattern, pattern=SEQUENCE_FLOW_PATTERN)

    models['n_activities'] = models['element_counts'].apply(activity_extraction)
    models['n_start_events'] = models['element_counts'].apply(start_event_extraction)
    models['n_end_events'] = models['element_counts'].apply(end_event_extraction)
    models['n_sequence_flows'] = models['element_counts'].apply(sequence_flow_extraction)

def _validate_minimal_elements(row):
    valid_activities = row['n_activities'] >= 1
    valid_start_events = row['n_start_events'] >= 1
    valid_end_events = row['n_end_events'] >= 1
    valid_sequence_flows = row['n_sequence_flows'] >=2

    return valid_activities and valid_start_events and valid_end_events and valid_sequence_flows

def check_minimal_model_elements(dataset: BPMNDataset):
    extract_required_counts(dataset)
    models = dataset.models
    models['has_minimal_elements'] = models.apply(_validate_minimal_elements, axis=1)

    total_models = len(models['has_minimal_elements'])
    valid_models = sum(models['has_minimal_elements'])
    invalid_models = total_models - valid_models

    print(f"Total models: {total_models}")
    print(f"Valid models: {valid_models}")
    print(f"Invalid models: {invalid_models}")

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

def analyse_duplicate_activity_names(dataset: BPMNDataset):
    name_column = _get_extracted_name_column(dataset)
    if not name_column == 'names_with_types':
        raise ValueError("This analysis can only be done if name and types were extracted.")
    models = dataset.models

    name_duplicate_calculation = partial(_calculate_activity_name_duplicates)

    models['duplicate_activity_names'] = models['names_with_types'].apply(name_duplicate_calculation)

    over_threshold = sum(models['duplicate_activity_names'] >= 0.4)
    all_models = len(models['duplicate_activity_names'])
    within_threshold = all_models - over_threshold

    print(f"All models: {all_models}")
    print(f"Over threshold: {over_threshold}")
    print(f"Within threshold: {within_threshold}")

    plt.hist(models['duplicate_activity_names'], bins=20, density=True),
    plt.title('Histogram of Percentages of duplicate activity names')
    plt.show()



