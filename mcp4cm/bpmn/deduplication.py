import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp4cm.bpmn.dataloading import BPMNDataset
from mcp4cm.bpmn.plotting_util import plot_duplicate_piechart


def detect_duplicates_by_hash(
        dataset: BPMNDataset,
        inplace: bool =False,
        plt_fig: bool =False
):

    starttime = time.time()

    duplicated_mask = dataset.models.duplicated(subset=['hash'], keep=False)

    unique_models = dataset.models[~duplicated_mask]
    duplicate_models = dataset.models[duplicated_mask]

    group_mask = duplicate_models.duplicated(subset=['hash'], keep='first')


    end_time = time.time()
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
        plot_duplicate_piechart(labels, sizes, "Proportion of Unique vs. Duplicate Files")



def detect_near_duplicates_by_tfidf(dataset: BPMNDataset,
    key='names',
    threshold: float=0.8,
    inplace: bool=False,
    plt_fig: bool=False):


    start_time = time.time()
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(dataset.models['names'])


    similarity_matrix = cosine_similarity(tf_idf_matrix)

    similarity_threshold = threshold



    end_time = time.time()


    print(f"Duplicate Detection and text-extraction took {end_time - start_time:.2f} seconds.")

