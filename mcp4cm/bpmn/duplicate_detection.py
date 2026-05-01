import time

from collections import Counter
from functools import partial

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components

from mcp4cm.util.text_util import join_texts

from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.bpmn.filtering_patterns import TFIDF_DUPLICATE_THRESHOLD
from mcp4cm.util.plotting_util import plot_duplicate_pie_chart


def _generate_tf_idf_matrix(dataset: BPMNDataset, key: str = 'names'):

    content_join_partial = partial(join_texts, delim=' ', empty_name=None)

    content_series = dataset.models[key].apply(content_join_partial)

    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(content_series)
    return content_series, tf_idf_matrix


def detect_duplicates_by_hash(
        dataset: BPMNDataset,
        inplace: bool = False,
        plt_fig: bool = False,
        print_results: bool = False
):
    """
    Detect duplicate models in the BPMNDataset based on their hash values.

    This function identifies exact duplicates in a dataset by using the hash values stored for each value.
    Because hash values for the BPMNDataset are required upon loading, hashes are not recomputed.
    It can optionally modify the dataset inplace to remove duplicates, visualize the duplicate distribution and print statistics for the deduplication.

    Args:
        dataset (Dataset): The dataset containing models.
        inplace (bool): If True, removes duplicates from the dataset. Defaults to False.
        plt_fig (bool): If True, displays a pie chart of unique vs. duplicate files. Defaults to False.
        print_results (bool): If True, prints statistics about unique and duplicate files in the Dataset. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - BPMNDataset: A BPMNDataset containing all unqiue models
            - BPMNDataset: A BPMNDataset containing all duplicate models, hash can be used to determine duplicate groups.

    Example:
        >>> unique_models, duplicate_groups = detect_duplicates_by_hash(dataset, inplace=True)
        >>> print(f"Found {len(duplicate_groups)} duplicate groups")
    """
    starttime = time.time()

    duplicated_mask = dataset.models.duplicated(subset=['hash'], keep=False)

    unique_models = dataset.models[~duplicated_mask]
    duplicate_models = dataset.models[duplicated_mask]

    group_mask = duplicate_models.duplicated(subset=['hash'], keep='first')

    end_time = time.time()

    total_number_of_models = len(dataset)
    unique_model_count = len(unique_models)
    duplicate_count = len(duplicate_models)
    number_of_duplicate_groups = sum(~group_mask)

    if print_results:
        print("\n=== Dataset Statistics ===")
        print(f"Duplicate Detection on already computed hashes took {end_time - starttime:.2f} seconds.")
        print(f"Total number of models: {total_number_of_models}")
        print(f"Total unique files: {unique_model_count}")
        print(f"Total duplicate files: {duplicate_count}")
        print(f"Number of duplicate groups: {number_of_duplicate_groups}")

    if inplace:
        dataset.models = unique_models

    if plt_fig:
        labels = ('Unique Files', 'Duplicate Files')
        sizes = (unique_model_count, duplicate_count)
        colors = ('green', 'red')

        plot_duplicate_pie_chart(labels, sizes,colors, "Proportion of Unique vs. Duplicate Files")

    unique_dataset = BPMNDataset(name=f'{dataset.name}_unique', models=unique_models)
    duplicate_dataset = BPMNDataset(name=f'{dataset.name}_duplicates', models=duplicate_models)

    return unique_dataset, duplicate_dataset


def tfidf_near_duplicate_detector(
        dataset: BPMNDataset,
        key='names',
        threshold: float = TFIDF_DUPLICATE_THRESHOLD,
        inplace: bool = False,
        plt_fig: bool = False,
        print_results: bool = False,
):
    """
    Detect near-duplicate models in the BPMNDataset based on TF-IDF vectorization and cosine similarity.

    This function identifies near-duplicate models by computing TF-IDF vectors
    for model text content and measuring their cosine similarity. Models with
    similarity above the threshold are considered near-duplicates.

    Args:
        dataset (BPMNDataset): The dataset containing Models.
        key (str): The key to the text content which is used to calculate TF-IDF vectors. Defaults to 'names'.
        threshold (float): The similarity threshold for considering two models as near-duplicates.
            Values range from 0 to 1, with 1 being identical and 0 being completely different. Defaults to TFIDF_DUPLICATE_THRESHOLD.
        inplace (bool): If True, removes near-duplicates from the dataset. Defaults to False.
        plt_fig (bool): If True, displays a pie chart of unique vs. near-duplicate files. Defaults to False.
        print_results (bool): If True, prints statistics about unique and duplicate files in the Dataset. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - BPMNDataset: A BPMNDataset containing all unqiue models
            - BPMNDataset: A BPMNDataset containing all duplicate models, where the field 'duplicate_group' a group number for duplicates.

    Example:
        >>> unique_models, near_duplicate_groups = tfidf_near_duplicate_detector(dataset, threshold=0.85)
        >>> print(f"Found {len(near_duplicate_groups)} near-duplicate groups with threshold {threshold}")
    """
    # Extract the text content from the models
    start_time = time.time()

    model_df_index = dataset.models.index

    content_series, tfidf_matrix = _generate_tf_idf_matrix(dataset, key)
    cosine_distance_threshold = 1 - threshold

    print('Generating Connectivity Graph')

    connectivity_graph = radius_neighbors_graph(tfidf_matrix,
                                                radius=cosine_distance_threshold,
                                                mode='distance',
                                                metric='cosine')
    print('Connectivity Graph Done')

    print('Calculating connected Components')
    n_components, labels = connected_components(connectivity_graph, directed=False, return_labels=True)
    print('Calculating connected Components Done')
    print(f'Number of components: {n_components}')

    print('Finding unique files:')
    label_counts = Counter(labels)

    unique_file_mask = [label_counts[label] == 1 for label in labels]
    duplicate_files_mask = [not is_unique for is_unique in unique_file_mask]
    indices_of_unique_files = content_series.index[unique_file_mask]


    print('Creating Duplicate Groups')

    duplicate_group_col_name = 'duplicate_group'
    duplicate_group_series = pd.Series(labels, index=model_df_index, name=duplicate_group_col_name)
    duplicate_group_series = duplicate_group_series.loc[duplicate_files_mask]
    duplicate_group_df = pd.merge(dataset.models, duplicate_group_series,
                                  left_index=True, right_index=True,
                                  how='right', validate='one_to_one')

    total_files_processed = len(dataset)
    unique_file_count = len(indices_of_unique_files)
    near_duplicate_count = total_files_processed - unique_file_count
    number_of_duplicate_groups = duplicate_group_df[duplicate_group_col_name].nunique()

    unique_model_df = dataset.models.loc[indices_of_unique_files,:]

    if inplace:
        dataset.models = unique_model_df

    if print_results:
        print("\n=== Dataset Statistics ===")
        print(f"Total files processed: {total_files_processed}")
        print(f"Total unique files: {unique_file_count}")
        print(f"Total duplicate files: {near_duplicate_count}")
        print(f"Number of duplicate groups: {number_of_duplicate_groups}")

    if plt_fig:
        labels = ('Unique Files', 'Near Duplicate Files')
        sizes = (unique_file_count, total_files_processed - unique_file_count)
        colors = ('green', 'red')
        plot_duplicate_pie_chart(labels, sizes, colors,"Proportion of Unique vs. Near Duplicate Files")

    unique_dataset = BPMNDataset(name=f'{dataset.name}_unique', models=unique_model_df)
    duplicate_dataset = BPMNDataset(name=f'{dataset.name}_duplicates', models=duplicate_group_df)

    return unique_dataset, duplicate_dataset


