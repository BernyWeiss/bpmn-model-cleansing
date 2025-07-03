from collections import Counter
from typing import List

import numpy as np
from mcp4cm.archimate.dataloading import ArchimateDataset, ArchimateModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from mcp4cm.archimate.filtering_patterns import (
    DUMMY_NAMES_THRESHOLD,
    MIN_SHORT_NAME_LENGTH,
    SHORT_NAMES_UPPER_THRESHOLD,
    dummy_name_pattern
)

def extract_names_counts_from_dataset(dataset: ArchimateDataset, ascending: bool=False, plt_figs: bool= False) -> dict:
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
        
    
    file_counts = {
        model.id: len(model.names) 
        for model in dataset.models
    }
    sorted_counts = dict(sorted(file_counts.items(), key=lambda item: item[1], reverse=not ascending))
    
    if plt_figs:
        plt.figure(figsize=(10, 6))
        plt.boxplot(sorted_counts.values(), vert=False)
        plt.title('Boxplot of Name Counts of Models')
        plt.xlabel('Number of Names')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.hist(sorted_counts.values(), bins=30, color='blue', alpha=0.7, log=True)
        plt.title('Histogram of Name Counts in Models (Log Scale)')
        plt.xlabel('Number of Names')
        plt.ylabel('Log Frequency of Models')
        plt.grid(True)
        plt.show()
    
    return sorted_counts


def get_word_counts_from_dataset(dataset: ArchimateDataset, plt_fig: bool = True, topk: int = 20) -> dict:
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
    models: List[ArchimateModel] = dataset.models
    print(f"Total models: {len(models)}")
    names = sum([list(set([n.strip().lower() for n in model.names if n.strip()])) for model in models], [])
    print(f"Total names: {len(names)}")
    name_counts = Counter(names)
    most_common_names = name_counts.most_common(topk)  # Get the top 20 most common names

    if plt_fig:
        plt.figure(figsize=(10, 8))
        names, counts = zip(*most_common_names)
        plt.bar(names, counts)
        plt.xlabel('Names')
        plt.ylabel('Frequency')
        plt.title(f'Top {topk} Most Frequent Names')
        plt.xticks(rotation=90)
        plt.show()

    return dict(most_common_names)


def get_name_length_distribution(dataset: ArchimateDataset, plt_fig: bool = True) -> dict:
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
    
    def get_model_name_lengths(model: ArchimateModel) -> tuple:
        """Retrieve the mean and median length of names from a file."""
        names = model.names
        mean_length, median_length = 0, 0
        lengths = [len(name) for name in names]  # Calculate lengths of each name
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)

        return {
            'mean_length': mean_length,
            'median_length': median_length
        }

    
    name_lengths = {
        model.id: get_model_name_lengths(model)
        for model in dataset.models
    }
    
    mean_lengths = [stats['mean_length'] for stats in name_lengths.values()]
    median_lengths = [stats['median_length'] for stats in name_lengths.values()]

    if plt_fig:
        plt.figure(figsize=(10, 6))
        plt.hist(mean_lengths, bins=30, color='blue', alpha=0.7, label='Mean Lengths')
        plt.hist(median_lengths, bins=30, color='orange', alpha=0.7, label='Median Lengths')
        plt.title('Distribution of Name Lengths in Models')
        plt.xlabel('Length of Names')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    return name_lengths


def filter_models_by_name_length_or_stopwords(
    dataset: ArchimateDataset, 
    length_upper_threshold: float = SHORT_NAMES_UPPER_THRESHOLD, 
    min_name_length: int = MIN_SHORT_NAME_LENGTH,
    inplace: bool = False
) -> list:
    
    def analyze_names(names: List[str]) -> bool:
        """ Analyze the names to categorize files based on criteria """
        short_names = [name for name in names if len(name.strip()) <= min_name_length]
        short_name_ratio = len(short_names) / len(names)
        return short_name_ratio > length_upper_threshold

    criteria_one_models = list()
    filtered_models = list()
    # Iterate over files in the directory
    models: List[ArchimateModel] = dataset.models
    for model in models:
        if analyze_names(model.names):
            criteria_one_models.append(model.id)
        else:
            filtered_models.append(model)
            
    # Output results
    print(f"Flagged models with > {length_upper_threshold*100}% short names: {len(criteria_one_models)}")
    print(f"Filtered models: {len(filtered_models)}")
    # dataset.models = filtered_models
    if inplace:
        dataset.models = filtered_models
        return dataset
    return ArchimateDataset(name=dataset.name, models=filtered_models)


def filter_dummy_names(dataset: ArchimateDataset, threshold: float = DUMMY_NAMES_THRESHOLD, inplace: bool = False) -> list:
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
        names = model.names
        dummy_names_count = sum(1 for name in names if dummy_name_pattern.match(name))
        dummy_ratio = dummy_names_count / len(names) if names else 0
        
        if dummy_ratio >= threshold:
            dummy_models.append((model.id, len(names), dummy_names_count, dummy_ratio))
            print(f"Filtered out {model.id} with {dummy_names_count}/{len(names)} dummy names ({dummy_ratio:.2%})")
            continue
        filtered_models.append(model)
    
    # Output results
    dummy_models.sort(key=lambda x: x[3], reverse=True)
    print(f"Flagged {len(dummy_models)} models based on dummy name percentage (Threshold: {threshold*100}%)\nShowing Top 10")
    for file, total, dummy, ratio in dummy_models[:10]:  # Show first 10 for preview
        print(f"{file} - {dummy}/{total} names ({ratio:.2%} dummy)")

    if inplace:
        dataset.models = filtered_models
        return dataset
    return ArchimateDataset(name=dataset.name, models=filtered_models)