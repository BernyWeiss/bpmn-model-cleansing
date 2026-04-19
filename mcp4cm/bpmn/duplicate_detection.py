import time

from collections import defaultdict, Counter
from functools import partial

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components

from mcp4cm.generic.utils import join_texts

from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.bpmn.filtering_patterns import TFIDF_DUPLICATE_THRESHOLD
from mcp4cm.bpmn.plotting_util import plot_duplicate_pie_chart


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

    Args:
        dataset:
        inplace:
        plt_fig:
        print_results:

    Returns:

    """
    starttime = time.time()

    duplicated_mask = dataset.models.duplicated(subset=['hash'], keep=False)

    unique_models = dataset.models[~duplicated_mask]
    duplicate_models = dataset.models[duplicated_mask]

    group_mask = duplicate_models.duplicated(subset=['hash'], keep='first')

    end_time = time.time()

    if print_results:
        print(f"Duplicate Detection on already computed hashes took {end_time - starttime:.2f} seconds.")
        print(f"Total number of models: {len(dataset)}")
        print(f"Total unique files: {len(unique_models)}")
        print(f"Total duplicate files: {len(dataset) - len(unique_models)}")
        print(f"Duplicate groups: {sum(~group_mask)}")

    if inplace:
        dataset.models = unique_models

    if plt_fig:
        labels = ('Unique Files', 'Duplicate Files')
        sizes = (len(unique_models), len(dataset) - len(unique_models))
        plot_duplicate_pie_chart(labels, sizes, "Proportion of Unique vs. Duplicate Files")


def tfidf_graph_near_duplicate_detector(
        dataset: BPMNDataset,
        key='names',
        threshold: float = TFIDF_DUPLICATE_THRESHOLD,
        inplace: bool = False,
        plt_fig: bool = False,
        print_results: bool = False,
):
    """

    Args:
        dataset:
        key:
        threshold:
        inplace:
        plt_fig:
        print_results:

    Returns:

    """
    # Extract the text content from the models
    start_time = time.time()
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

    print('Finding unique files:')
    label_counts = Counter(labels)
    unique_file_mask = [label_counts[label] == 1 for label in labels]
    indices_of_unique_files = content_series.index[unique_file_mask]

    print('Creating Duplicate Groups')

    total_files_processed = len(dataset)
    unique_file_count = len(indices_of_unique_files)
    near_duplicate_count = total_files_processed - unique_file_count
    number_of_duplicate_groups = 0  # TODO: Change after generation of duplicate groups

    if inplace:
        dataset.models = dataset.models.loc[indices_of_unique_files,:]

    if print_results:
        print("\n=== Dataset Statistics ===")
        print(f"Total files processed: {total_files_processed}")
        print(f"Total unique files: {unique_file_count}")
        print(f"Total duplicate files: {near_duplicate_count}")
        print(f"Number of duplicate groups: {number_of_duplicate_groups}")

    if plt_fig:
        labels = ('Unique Files', 'Near Duplicate Files')
        sizes = (unique_file_count, total_files_processed - unique_file_count)
        plot_duplicate_pie_chart(labels, sizes, "Proportion of Unique vs. Near Duplicate Files")


def tfidf_cluster_near_duplicate_detector(
        dataset: BPMNDataset,
        key='names',
        threshold: float = TFIDF_DUPLICATE_THRESHOLD,
        inplace: bool = False,
        plt_fig: bool = False
):
    # TODO: This method can be removed.
    #  Agglomerative clustering might be an option when it works on sparse inputs (to improve creation of duplicate groups)

    # Extract the text content from the models
    start_time = time.time()
    text_data, tfidf_matrix = _generate_tf_idf_matrix(dataset, key)
    cosine_distance_threshold = 1 - threshold

    print('Generating Connectivity Graph')

    connectivity_graph = radius_neighbors_graph(tfidf_matrix,
                                                radius=cosine_distance_threshold,
                                                mode='distance',
                                                metric='cosine')
    print('Connectivity Graph Done')

    print('Reducing Matrix Features')
    reduced_matrix = TruncatedSVD(n_components=50, random_state=0).fit_transform(tfidf_matrix)
    normalized_reduced_matrix = normalize(reduced_matrix)
    print('Reducing Matrix Features Done')

    print('Creating Clustering on Features.')
    clustering_algorithm = AgglomerativeClustering(n_clusters=None,
                                                   metric='cosine',
                                                   distance_threshold=cosine_distance_threshold,
                                                   linkage='average')

    cluster_labels = clustering_algorithm.fit_predict(normalized_reduced_matrix)
    print('Clustering Done')
    print('Number of clusters:')
    print(clustering_algorithm.n_clusters)
    print(cluster_labels)
