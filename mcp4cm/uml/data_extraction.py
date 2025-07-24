from collections import Counter
import re
from typing import List
import xml.etree.ElementTree as ET

import numpy as np
from mcp4cm.uml.dataloading import UMLDataset, UMLModel
from mcp4cm.uml.utils import ns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from mcp4cm.uml.filtering_patterns import (
    DUMMY_CLASSES_THRESHOLD,
    DUMMY_NAMES_THRESHOLD,
    DUMMY_WORD_THRESHOLD,
    FREQUENCT_NAMES,
    GENERIC_PATTERN_THRESHOLD_COUNT,
    MIN_MEDIAN_SHORT_NAME_LENGTH,
    MIN_NAMES_COUNT,
    MIN_SHORT_NAME_LENGTH,
    SEQUENTIAL_THRESHOLD,
    SHORT_DUMMY_WORD_THRESHOLD,
    SHORT_NAMES_LOWER_THRESHOLD,
    SHORT_NAMES_UPPER_THRESHOLD,
    STOPWORDS_THRESHOLD,
    VOCABULARY_UNIQUENESS_THRESHOLD,
    empty_name_pattern,
    empty_class_name_pattern,
    comment_pattern,
    dummy_name_pattern,
    dummy_class_pattern,
    general_class_pattern,
    myclass_pattern,
    numbered_pattern,
    two_char_pattern,
    letter_space_letter_pattern,
    DUMMY_KEYWORDS,
)
from mcp4cm.utils import split_name


def filter_empty_or_invalid_files(
    dataset: UMLDataset, inplace: bool = False
) -> UMLDataset:
    """
    Filter out empty or invalid files from the dataset.

    This function removes models with empty or unparseable XMI content from the dataset.
    It checks if each model's XMI content is present, non-empty, and valid XML.

    Args:
        dataset (UMLDataset): The dataset to filter.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only valid models.

    Example:
        >>> filtered_dataset = filter_empty_or_invalid_files(dataset, inplace=True)
        >>> print(f"Removed {len(dataset.models) - len(filtered_dataset.models)} invalid models")
    """
    filtered_models = []
    empty_models, invalid_models = [], []
    for model in tqdm(dataset.models, desc="Filtering models"):
        if not model.model_xmi:
            continue

        # Check if the file is empty
        if len(model.model_xmi) == 0:
            empty_models.append(model)
            continue

        try:
            # Parse the XML file
            tree = ET.ElementTree(ET.fromstring(model.model_xmi))
            tree.getroot()
            # If parsing is successful, add the model to the filtered list
            filtered_models.append(model)
        except ET.ParseError:
            invalid_models.append(model)

    # dataset.models = filtered_models
    print(
        f"Filtered out {len(empty_models)} empty models and {len(invalid_models)} invalid models."
    )
    # return {
    #     'empty': empty_models,
    #     'invalid': invalid_models
    # }
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def extract_names_from_model(
    model: UMLModel, 
    use_types: bool = False, 
    empty_name_pattern="empty name"
) -> UMLDataset:
    """
    Extract names from a single UML model, including all types of artifacts.

    This function parses the XMI content of a UML model and extracts names of all
    model elements (classes, attributes, operations, etc.). It can optionally
    include the element type with each name.

    Args:
        model (UMLModel): The UML model to extract names from.
        use_types (bool): If True, includes the element type with each name (e.g., "class: Customer").
            If False, returns only the names. Defaults to False.
        empty_name_pattern (str): The string to use for empty names. Defaults to 'empty name'.

    Returns:
        list: A list of extracted names, optionally with their types.

    Example:
        >>> names = extract_names_from_model(model)
        >>> print(f"Found {len(names)} named elements in the model")
        >>> names_with_types = extract_names_from_model(model, use_types=True)
        >>> class_names = [n.split(': ')[1] for n in names_with_types if n.startswith('class: ')]
    """

    extracted_info = []

    try:
        tree = ET.ElementTree(ET.fromstring(model.model_xmi))
        root = tree.getroot()

        for elem in root.iter():
            xsi_type = elem.get(f"{{{ns['xsi']}}}type", None)
            if not xsi_type:
                tag_type = elem.tag.split("}")[-1]
                xsi_type = f"uml:{tag_type}"

            artifact_type = xsi_type.split(":")[-1].lower()
            if "name" in elem.attrib:
                name = elem.attrib["name"].strip()
                name_entry = split_name(name) if name else empty_name_pattern
                formatted_name = (
                    f"{artifact_type}: {name_entry}" if use_types else name_entry
                )
                extracted_info.append(formatted_name)

            if elem.tag.endswith("ownedComment") and "body" in elem.attrib:
                comment = elem.attrib["body"].strip()
                formatted_comment = f"comment: {split_name(comment)}"
                content = formatted_comment if use_types else f"{split_name(comment)}"
                extracted_info.append(content)

        if use_types:
            model.names_with_types = extracted_info
        else:
            model.names = extracted_info

    except Exception as e:
        print(f"Error processing model {model.id}: {e}")
    return extracted_info


def filter_models_by_comments(
    dataset: UMLDataset, inplace: bool = False    
):
    files_with_comments = []
    for model in tqdm(dataset.models, desc="Filtering models by comments"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model, use_types=True)
        if any(comment_pattern in line for line in names):
            files_with_comments.append(model)
    
    print(f"Total files containing comments: {len(files_with_comments)}")
    if inplace:
        dataset.models = files_with_comments
        return dataset
    return UMLDataset(name=dataset.name, models=files_with_comments)


def filter_models_without_names(
    dataset: UMLDataset, inplace: bool = False
) -> UMLDataset:
    """
    Filter out models containing elements with empty or missing names.

    This function identifies and removes models that contain elements with empty names
    (represented as 'empty name' after extraction). Models with unnamed elements are
    often incomplete or not properly constructed.

    Args:
        dataset (UMLDataset): The dataset to filter.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with properly named elements.

    Example:
        >>> filtered_dataset = filter_models_without_names(dataset, inplace=True)
        >>> print(f"Retained {len(filtered_dataset.models)} models with all elements properly named")
    """
    filtered_models = list()
    empty_models = list()
    for model in tqdm(dataset.models, desc="Filtering models without names"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model)
        if any(re.fullmatch(empty_name_pattern, name) for name in names):
            empty_models.append(model)
            continue
        filtered_models.append(model)

    # dataset.models = filtered_models
    print(f"Filtered out {len(empty_models)} models with empty names.")
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_models_by_name_length(
    dataset: UMLDataset,
    min_length: int = MIN_MEDIAN_SHORT_NAME_LENGTH,
    inplace: bool = False,
) -> UMLDataset:
    """
    Filter models based on the median length of names.
    """

    filtered_models = []
    short_name_models = []

    for model in tqdm(dataset.models, desc="Filtering models by name length"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model)
        if not names:
            continue

        lengths = [len(name) for name in names]
        median_length = np.median(lengths)

        if median_length < min_length:
            short_name_models.append(model)
            continue

        filtered_models.append(model)

    print(f"Filtered out {len(short_name_models)} models with short names.")
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_models_by_name_count(
    dataset: UMLDataset,
    min_count: int = MIN_NAMES_COUNT,
    inplace: bool = False,
) -> UMLDataset:
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
    filtered_models = [
        model
        for model in dataset.models
        if min_count
        <= (
            len(extract_names_from_model(model))
            if not model.names
            else len(model.names)
        )
    ]

    print(
        f"Models After Filtering with name counts between {min_count}: {len(dataset.models) - len(filtered_models)}"
    )
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_models_by_empty_names(
    dataset: UMLDataset, inplace: bool = False
) -> UMLDataset:
    """
    Filter models that contain elements with empty names.
    """

    filtered_models = [
        m
        for m in tqdm(dataset)
        if "empty name" in "\n".join(extract_names_from_model(m))
    ]
    print(
        f"Filtered out {len(dataset.models) - len(filtered_models)} models with empty names."
    )
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_models_with_empty_class_names(
    dataset: UMLDataset, inplace: bool = False
) -> UMLDataset:
    """
    Identify and filter out models containing classes with empty names.

    This function scans through the dataset to find models that contain classes
    with no names or placeholder names like 'empty name'. Such models often indicate
    incomplete or auto-generated content and may not be suitable for analysis.

    Args:
        dataset (UMLDataset): The dataset to filter.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a list
            of models with empty class names. Defaults to False.

    Returns:
        If inplace=True: UMLDataset with problematic models removed.
        If inplace=False: List[UMLModel] containing the filtered-out models with empty class names.

    Example:
        >>> filtered_dataset = filter_models_with_empty_class_names(dataset, inplace=True)
        >>> print(f"Removed {len(dataset.models) - len(filtered_dataset.models)} models with empty class names")
    """
    files_with_empty_class_names = []
    filtered_models = []

    # Iterate over all files in the given directory
    for model in tqdm(
        dataset.models, desc="Searching for empty class names", unit="file"
    ):
        if not model.model_xmi:
            continue

        if any(
            re.fullmatch(empty_class_name_pattern, name)
            for name in extract_names_from_model(model, use_types=True)
        ):
            files_with_empty_class_names.append(model)
            continue

        filtered_models.append(model)

    # dataset.models = filtered_models

    print(f"Found {len(files_with_empty_class_names)} files with empty class names.")

    if inplace:
        dataset.models = filtered_models
        return dataset
    return dataset


def find_files_with_comments(dataset: UMLDataset) -> UMLDataset:
    """
    Identify models in the dataset that contain comments.

    This function scans through all models in the dataset to find those that
    contain comments in their XMI content. Comments can provide valuable metadata
    and documentation about the model's purpose and usage.

    Args:
        dataset (UMLDataset): The dataset to search for models with comments.

    Returns:
        List[UMLModel]: A list of models containing comments.

    Example:
        >>> models_with_comments = find_files_with_comments(dataset)
        >>> print(f"Found {len(models_with_comments)} models containing documentation comments")
        >>> for model in models_with_comments[:5]:  # Show first 5
        ...     print(f"Model ID: {model.id}")
    """
    files_with_comments = []

    # Iterate through all files in the directory
    for model in tqdm(dataset.models, desc="Searching for comments", unit="file"):
        if not model.model_xmi:
            continue

        if any(
            comment_pattern in line
            for line in extract_names_from_model(model, use_types=True)
        ):
            files_with_comments.append(model)

    print(f"Total files containing comments: {len(files_with_comments)}")
    return files_with_comments


def extract_names_counts_from_dataset(
    dataset: UMLDataset, ascending: bool = False, plt_figs: bool = False
) -> dict:
    """
    Extract names from all models and produce a dictionary of name counts.

    This function processes all models in a dataset, extracts names from each model,
    and returns a dictionary mapping model IDs to their extracted names. Optionally,
    it can also generate visualizations of the name count distribution.

    Args:
        dataset (UMLDataset): The dataset to process.
        ascending (bool): If True, sorts models by name count in ascending order.
            If False (default), sorts in descending order.
        plt_figs (bool): If True, displays boxplot and histogram visualizations
            of the name count distribution. Defaults to False.

    Returns:
        dict: A dictionary mapping model IDs to their corresponding extracted names,
            sorted by name count according to the ascending parameter.

    Example:
        >>> name_counts = extract_names_counts_from_dataset(dataset, plt_figs=True)
        >>> print(f"Model with most names: {list(name_counts.keys())[0]}")
        >>> print(f"Number of names in that model: {len(list(name_counts.values())[0])}")
    """

    for model in dataset.models:
        if not model.names:
            extract_names_from_model(model)

    file_counts = {model.id: len(model.names) for model in dataset.models}
    sorted_counts = dict(
        sorted(file_counts.items(), key=lambda item: item[1], reverse=not ascending)
    )
    if plt_figs:
        plt.figure(figsize=(10, 6))
        plt.boxplot(sorted_counts.values(), vert=False)
        plt.title("Boxplot of Name Counts of Models")
        plt.xlabel("Number of Names")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(sorted_counts.values(), bins=30, color="blue", alpha=0.7, log=True)
        plt.title("Histogram of Name Counts in Models (Log Scale)")
        plt.xlabel("Number of Names")
        plt.ylabel("Log Frequency of Models")
        plt.grid(True)
        plt.show()

    return sorted_counts


def get_word_counts_from_dataset(
    dataset: UMLDataset, plt_fig: bool = True, topk: int = 20
) -> dict:
    """
    Analyze the frequency of names across the entire dataset.

    This function extracts all unique names from models in the dataset, counts their
    frequency, and returns the most common names. Optionally, it can visualize the
    results as a bar chart.

    Args:
        dataset (UMLDataset): The dataset to analyze.
        plt_fig (bool): If True, displays a bar chart visualization of the most common names.
            Defaults to True.
        topk (int): The number of most frequent names to return. Defaults to 20.

    Returns:
        dict: A dictionary mapping names to their frequency counts, containing the
            topk most frequent names in the dataset.

    Example:
        >>> common_names = get_word_counts_from_dataset(dataset, topk=10)
        >>> print("Most common names in the dataset:")
        >>> for name, count in common_names.items():
        ...     print(f"{name}: {count} occurrences")
    """
    models: List[UMLModel] = dataset.models
    for model in models:
        if not model.names:
            extract_names_from_model(model)

    print(f"Total models: {len(models)}")
    names = sum(
        [
            list(set([n.strip().lower() for n in model.names if n.strip()]))
            for model in models
        ],
        [],
    )
    print(f"Total names: {len(names)}")
    name_counts = Counter(names)
    most_common_names = name_counts.most_common(
        topk
    )  # Get the top 20 most common names

    if plt_fig:
        plt.figure(figsize=(10, 8))
        names, counts = zip(*most_common_names)
        plt.bar(names, counts)
        plt.xlabel("Names")
        plt.ylabel("Frequency")
        plt.title(f"Top {topk} Most Frequent Names")
        plt.xticks(rotation=90)
        plt.show()

    return dict(most_common_names)


def get_name_length_distribution(dataset: UMLDataset, plt_fig: bool = True) -> dict:
    """
    Analyze the distribution of name lengths across models in the dataset.

    This function calculates the mean and median length of names in each model
    and provides statistical insights about name length patterns across the dataset.
    Optionally, it can visualize the distribution with a histogram.

    Args:
        dataset (UMLDataset): The dataset to analyze.
        plt_fig (bool): If True, displays a histogram visualization of name length
            distributions. Defaults to True.

    Returns:
        dict: A dictionary mapping model IDs to dictionaries containing the mean
            and median length of names in each model.

    Example:
        >>> length_stats = get_name_length_distribution(dataset)
        >>> # Get average of mean name lengths across all models
        >>> avg_mean_length = sum(stats['mean_length'] for stats in length_stats.values()) / len(length_stats)
        >>> print(f"Average mean name length across all models: {avg_mean_length:.2f} characters")
    """

    def get_model_name_lengths(m: UMLModel) -> tuple:
        """Retrieve the mean and median length of names from a file."""
        names = m.names if m.names else extract_names_from_model(m)

        mean_length, median_length = 0, 0
        if not names:
            print(f"No names found in model {m.id}.")
        else:
            lengths = [len(name) for name in names]  # Calculate lengths of each name
            mean_length = np.mean(lengths)
            median_length = np.median(lengths)

        return {"mean_length": mean_length, "median_length": median_length}

    name_lengths = {model.id: get_model_name_lengths(model) for model in dataset.models}

    mean_lengths = [v["mean_length"] for v in name_lengths.values()]
    median_lengths = [v["median_length"] for v in name_lengths.values()]

    if plt_fig:
        plt.figure(figsize=(10, 6))
        plt.hist(mean_lengths, bins=30, color="blue", alpha=0.7, label="Mean Lengths")
        plt.hist(
            median_lengths, bins=30, color="orange", alpha=0.7, label="Median Lengths"
        )
        plt.title("Distribution of Name Lengths in Models")
        plt.xlabel("Length of Names")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

    return name_lengths


def filter_models_by_name_length_or_stopwords(
    dataset: UMLDataset,
    length_upper_threshold: float = SHORT_NAMES_UPPER_THRESHOLD,
    length_lower_threshold: float = SHORT_NAMES_LOWER_THRESHOLD,
    stopword_threshold: float = STOPWORDS_THRESHOLD,
    min_name_length: int = MIN_SHORT_NAME_LENGTH,
    inplace: bool = False,
) -> UMLDataset:
    def analyze_names(names: List[str]) -> bool:
        """Analyze the names to categorize files based on criteria"""
        short_names = [name for name in names if len(name.strip()) <= min_name_length]
        stopwords_count = sum(
            1
            for name in names
            if any(stopword in name.lower() for stopword in FREQUENCT_NAMES)
        )

        short_name_ratio = len(short_names) / len(names)
        stopwords_ratio = stopwords_count / len(names)

        criteria_1 = short_name_ratio >= length_upper_threshold
        criteria_2 = (
            short_name_ratio >= length_lower_threshold
            and stopwords_ratio >= stopword_threshold
        )

        return criteria_1, criteria_2

    criteria_one_models, criteria_two_models = list(), list()
    filtered_models = list()
    # Iterate over files in the directory
    models: List[UMLModel] = dataset.models
    for model in models:
        names = model.names if model.names else extract_names_from_model(model)

        c1, c2 = analyze_names(names)
        if c1:
            criteria_one_models.append(model.id)
        if c2:
            criteria_two_models.append(model.id)

        if not c1 and not c2:
            filtered_models.append(model)

    # Output results
    print(
        f"Flagged models with > {length_upper_threshold * 100}% short names: {len(criteria_one_models)}"
    )
    print(
        f"Flagged models with >= {length_lower_threshold * 100}% short names and >= {stopword_threshold * 100}% of stopwords: {len(criteria_two_models)}"
    )

    print(
        f"Models After Filtering: {len(filtered_models)}. Filtered: {len(dataset.models) - len(filtered_models)}"
    )
    # dataset.models = filtered_models
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_dummy_names(
    dataset: UMLDataset, 
    threshold: float = DUMMY_NAMES_THRESHOLD, 
    inplace: bool = False
) -> UMLDataset:
    """
    Filter out models containing a high proportion of dummy or placeholder names.

    This function identifies models where a significant percentage of names match
    common placeholder patterns (like 'test', 'dummy', 'foo', etc.) and removes
    them from the dataset. These models often represent test data or auto-generated
    content rather than meaningful UML diagrams.

    Args:
        dataset (UMLDataset): The dataset to filter.
        threshold (float): The maximum allowed proportion of dummy names. Models with
            a higher proportion will be filtered out. Defaults to DUMMY_NAMES_THRESHOLD.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an acceptable proportion of real names.

    Example:
        >>> filtered_dataset = filter_dummy_names(dataset, threshold=0.3, inplace=True)
        >>> print(f"Kept {len(filtered_dataset.models)} models with mostly meaningful names")
    """
    filtered_models = []
    dummy_models = []

    for model in tqdm(dataset.models, desc="Filtering dummy names"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model) if model.names is None else model.names
        dummy_names_count = sum(
            1 for name in names if dummy_name_pattern.match(name.strip())
        )
        if dummy_names_count:
            dummy_models.append((model.id, len(names), dummy_names_count))
        else:
            filtered_models.append(model)

        # dummy_ratio = dummy_names_count / len(names) if names else 0

        # if dummy_ratio >= threshold:
        #     dummy_models.append((model.id, len(names), dummy_names_count, dummy_ratio))
        # else:
        #     filtered_models.append(model)

    # Output results
    dummy_models.sort(key=lambda x: x[2], reverse=True)
    print(
        f"Flagged {len(dummy_models)} models based on dummy name percentage (Threshold: {threshold * 100}%)\nShowing Top 10"
    )
    # for file, total, dummy, ratio in dummy_models[:10]:  # Show first 10 for preview
    #     print(f"{file} - {dummy}/{total} names ({ratio:.2%} dummy)")

    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_dummy_short_names(
    dataset: UMLDataset,
    threshold: float = SHORT_DUMMY_WORD_THRESHOLD,
    inplace: bool = False,
):
    """
    Filter out models containing a high proportion of dummy or placeholder names that are shorter than a specified length.
    This function identifies models where a significant percentage of names match common placeholder patterns (like 'a1', 'b2', etc.) and removes them from the dataset.
    """

    filtered_models = []
    dummy_short_names = []

    for model in tqdm(dataset.models, desc="Filtering dummy short names"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model) if not model.names else model.names
        dummy_count = sum(
            1
            for name in names
            if two_char_pattern.match(name) or letter_space_letter_pattern.match(name)
        )
        total_names = len(names)

        # Compute the percentage of dummy-like names
        dummy_ratio = dummy_count / total_names
        if dummy_ratio >= threshold:
            dummy_short_names.append((model.id, total_names, dummy_count, dummy_ratio))
        else:
            filtered_models.append(model)

    # Output results
    print(
        f"Flagged {len(dummy_short_names)} models based on short dummy name percentage (Threshold: {threshold * 100}%)\nShowing Top 10"
    )
    # for file, total, dummy, ratio in dummy_short_names[:10]:  # Show first 10 for preview
    #     print(f"{file} - {dummy}/{total} names ({ratio:.2%} short dummy)")

    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_dummy_classes(
    dataset: UMLDataset,
    threshold: float = DUMMY_CLASSES_THRESHOLD,
    inplace: bool = False,
) -> UMLDataset:
    """
    Filter out models containing a high proportion of classes with dummy or generic names.

    This function focuses specifically on class names (rather than all element names)
    and identifies models where a significant percentage of classes have placeholder names
    like 'Class1', 'TestClass', etc. Models with many generic class names often
    represent low-quality or auto-generated UML content.

    Args:
        dataset (UMLDataset): The dataset to filter.
        threshold (float): The maximum allowed proportion of dummy class names. Models with
            a higher proportion will be filtered out. Defaults to DUMMY_CLASSES_THRESHOLD.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an acceptable proportion of
            meaningfully named classes.

    Example:
        >>> filtered_dataset = filter_dummy_classes(dataset, threshold=0.25, inplace=True)
        >>> print(f"Kept {len(filtered_dataset.models)} models with mostly meaningful class names")
    """
    filtered_models = []
    files_fully_dummy = []
    files_mostly_valid = []
    files_mixed_classes = []

    for model in tqdm(dataset.models, desc="Filtering dummy classes"):
        if not model.model_xmi:
            continue

        names = (
            extract_names_from_model(model, use_types=True)
            if not model.names_with_types
            else model.names_with_types
        )

        dummy_count, valid_count, dummy_found = 0, 0, False
        for name in names:
            type_name = name.strip().split(":")  # Splitting line into type and name
            if len(type_name) < 2:  # Ensure there is a type and a name
                continue

            artifact_type, name = type_name[0].strip().lower(), type_name[1].strip()
            if artifact_type == "class":  # Only process class types
                if dummy_class_pattern.match(name):
                    dummy_count += 1
                    dummy_found = True  # Set flag on finding a dummy name
                elif general_class_pattern.match(name):
                    valid_count += 1

        addable = True
        if dummy_found:  # Only process further if a dummy name was found
            # Evaluate file based on counts
            total_classes = dummy_count + valid_count
            if total_classes == 0:
                continue  # Avoid division by zero, handle files with no class definitions

            dummy_ratio = dummy_count / total_classes
            # Define thresholds
            if dummy_ratio > threshold:
                files_fully_dummy.append(model.id)
                addable = False
            elif dummy_count > 0 and dummy_ratio <= 0.13:  # Less than 13% dummy names
                files_mixed_classes.append(model.id)
            else:
                files_mostly_valid.append(model.id)
                addable = False
        if addable:
            filtered_models.append(model)

    # Output results
    print(
        f"Flagged models based on dummy class percentage (Threshold: {threshold * 100}%)"
    )
    print(f"Files fully dummy: {len(files_fully_dummy)}")
    print(f"Files mostly valid (with few dummy classes): {len(files_mostly_valid)}")
    print(
        f"Files with a mix of dummy and non-dummy classes: {len(files_mixed_classes)}"
    )
    print(
        f"Models After Filtering: {len(filtered_models)}. Filtered: {len(dataset.models) - len(filtered_models)}"
    )

    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_classes_by_generic_pattern(
    dataset: UMLDataset,
    threshold_count: int = GENERIC_PATTERN_THRESHOLD_COUNT,
    inplace: bool = False,
) -> UMLDataset:
    """
    Filter out models containing classes with overly generic naming patterns.

    This function identifies models that contain classes with names following patterns
    like 'MyClass', 'MyClass1', 'MyClass2', etc. These patterns often indicate placeholder
    or example models rather than real-world UML diagrams with domain-specific naming.

    Args:
        dataset (UMLDataset): The dataset to filter.
        threshold_count (int): The maximum allowed number of generic class names. Models with
            more than this number will be filtered out. Defaults to GENERIC_PATTERN_THRESHOLD_COUNT.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models with an acceptable number of specific class names.

    Example:
        >>> filtered_dataset = filter_classes_by_generic_pattern(dataset, threshold_count=2, inplace=True)
        >>> print(f"Kept {len(filtered_dataset.models)} models with specific class names")
    """
    filtered_models = []
    generic_classes = []

    for model in tqdm(dataset.models, desc="Filtering generic classes"):
        if not model.model_xmi:
            continue

        names = (
            extract_names_from_model(model, use_types=True)
            if not model.names_with_types
            else model.names_with_types
        )

        name_count = sum(1 for name in names if myclass_pattern.match(name))
        if name_count > threshold_count:
            generic_classes.append((model.id, len(names), name_count))
            continue
        filtered_models.append(model)

    # Output results
    print(
        "Files containing more than one 'class: my class' or 'class: my class' followed by a number:"
    )
    print(
        f"Models After Filtering based on generic class names (Threshold: {threshold_count})"
    )
    # for file, total, count in generic_classes:
    #     print(f"{file} - {count}/{total} names")

    print(
        f"Models After Filtering: {len(filtered_models)}. Filtered: {len(dataset.models) - len(filtered_models)}"
    )

    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)


def filter_models_by_sequential_and_dummy_words(
    dataset: UMLDataset,
    sequential_threshold: float = SEQUENTIAL_THRESHOLD,
    dummy_word_threshold: float = DUMMY_WORD_THRESHOLD,
    vocabulary_uniqueness_threshold: int = VOCABULARY_UNIQUENESS_THRESHOLD,
    inplace: bool = False,
) -> UMLDataset:
    """
    Filter models by sequential and dummy word patterns.

    This function identifies and filters out models that contain a high percentage
    of sequentially named elements (e.g., Class1, Class2, Class3), dummy keywords,
    or have low vocabulary diversity. These patterns often indicate auto-generated
    or low-quality models.

    Args:
        dataset (UMLDataset): The dataset to filter.
        sequential_threshold (float): Threshold for proportion of sequentially named elements.
            Models with more than this proportion will be filtered out. Defaults to SEQUENTIAL_THRESHOLD.
        dummy_word_threshold (float): Threshold for proportion of names containing dummy words.
            Models with more than this proportion will be filtered out. Defaults to DUMMY_WORD_THRESHOLD.
        vocabulary_uniqueness_threshold (int): Minimum number of unique words required.
            Models with fewer unique words will be filtered out. Defaults to VOCABULARY_UNIQUENESS_THRESHOLD.
        inplace (bool): If True, modifies the dataset in-place. If False, returns a new dataset.
            Defaults to False.

    Returns:
        UMLDataset: The filtered dataset, either the original dataset modified in-place
            or a new dataset containing only models that pass the filtering criteria.

    Example:
        >>> filtered_dataset = filter_models_by_sequential_and_dummy_words(
        ...     dataset,
        ...     sequential_threshold=0.6,
        ...     dummy_word_threshold=0.7,
        ...     inplace=True
        ... )
        >>> print(f"Kept {len(filtered_dataset.models)} high-quality models")
    """
    filtered_models = []
    flagged_models = []

    for model in tqdm(dataset.models, desc="Filtering sequential and dummy words"):
        if not model.model_xmi:
            continue

        names = extract_names_from_model(model)

        # Check for sequential patterns
        sequential_count = sum(1 for name in names if numbered_pattern.match(name))
        sequential_ratio = sequential_count / len(names)

        # Check for dummy words
        dummy_count = sum(
            1 for name in names if any(dw in name for dw in DUMMY_KEYWORDS)
        )
        dummy_ratio = dummy_count / len(names)

        # Check for vocabulary uniqueness
        words = [word for name in names for word in name.split()]
        unique_words = set(words)
        # Flagging condition
        if (
            sequential_ratio >= sequential_threshold  # Too many numbered names
            or dummy_ratio >= dummy_word_threshold  # Too many generic words
            or len(unique_words)
            <= vocabulary_uniqueness_threshold  # Low vocabulary richness
        ):
            flagged_models.append(
                (model.id, len(names), sequential_ratio, dummy_ratio, len(unique_words))
            )
        else:
            filtered_models.append(model)

    # Output results
    print(
        f"Flagged models based on sequential patterns (Threshold: {sequential_threshold * 100}%)"
    )
    print(
        f"Flagged models based on dummy word percentage (Threshold: {dummy_word_threshold * 100}%)"
    )
    print(
        f"Flagged models based on vocabulary uniqueness (Threshold: {vocabulary_uniqueness_threshold})"
    )

    # for file, total, seq_ratio, dummy_ratio, vocab_count in flagged_models:
    #     print(f"{file} - {total} names, {seq_ratio:.2%} sequential, {dummy_ratio:.2%} dummy words, {vocab_count} unique words")
    print(
        f"Models After Filtering: {len(filtered_models)}. Filtered: {len(dataset.models) - len(filtered_models)}"
    )
    if inplace:
        dataset.models = filtered_models
        return dataset
    return UMLDataset(name=dataset.name, models=filtered_models)
