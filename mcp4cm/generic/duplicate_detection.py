import hashlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from mcp4cm.base import Dataset
import time
from mcp4cm.generic.filtering_patterns import TFIDF_DUPLICATE_THRESHOLD
from mcp4cm.generic.utils import get_model_text

def get_file_hash(string: str) -> str:
    """
    Compute a SHA-256 hash for the content of a file.
    
    Args:
        string (str): The content of the file to hash.
    
    Returns:
        str: The SHA-256 hash of the file content.
    
    Example:
        >>> hash_value = get_file_hash("example content")
        >>> print(hash_value)
        '5d41402abc4b2a76b9719d911017c592'
    """    
    
    return hashlib.sha256(string.encode()).hexdigest()  # Hash the content
    


def detect_duplicates_by_hash(
    dataset: Dataset, 
    hash_function=get_file_hash, 
    key: str='names', 
    inplace: bool=False, 
    plt_fig: bool=False
):
    """
    Detect duplicate UML models based on their hash values.
    
    This function identifies exact duplicates in a dataset by computing hash values
    for each model's content. It can optionally modify the dataset in-place to remove
    duplicates and visualize the duplicate distribution.
    
    Args:
        dataset (UMLDataset): The dataset containing UML models.
        hash_function (callable): Function to compute hash values. Defaults to get_file_hash.
        inplace (bool): If True, removes duplicates from the dataset. Defaults to False.
        plt_fig (bool): If True, displays a pie chart of unique vs. duplicate files. Defaults to False.
    
    Returns:
        tuple: A tuple containing:
            - List[UMLModel]: A list of unique UML models.
            - List[tuple]: A list of duplicate groups, where each group is a tuple of (original, duplicate).
    
    Example:
        >>> unique_models, duplicate_groups = detect_duplicates_by_hash(dataset, inplace=True)
        >>> print(f"Found {len(duplicate_groups)} duplicate groups")
    """
    
    hash_dict = {}
    unique_files = []
    duplicate_files = []
    
    assert all(hasattr(model, key) for model in dataset.models), f"All models must have a '{key}' attribute"
    
    start_time = time.time()
    for model in dataset.models:
        if key in ['names', 'names_with_types']:
            content = "\n".join(model.names_with_types if hasattr(model, 'names_with_types') else model.names) + "\n"
        else:
            content = get_model_text(model, key) + "\n"
            
        file_hash = hash_function(content)
        if file_hash is not None:
            if file_hash not in hash_dict:
                hash_dict[file_hash] = list()
                
            hash_dict[file_hash].append(model.file_path)
            # print(model.file_path, file_hash)  # Print the file path and its hash for debugging
    
    duplicate_files = {h: files for h, files in hash_dict.items() if len(files) > 1}
    unique_files = {h: files[0] for h, files in hash_dict.items() if len(files) == 1}
    
    
    end_time = time.time()
    print(f"Hashing and duplicate detection took {end_time - start_time:.2f} seconds.")
    print(f"Total files processed: {len(dataset)}")
    print(f"Total unique files: {len(unique_files)}")
    print(f"Total duplicate files: {len(dataset) - len(unique_files)}")
    print(f"Duplicate groups: {len(duplicate_files)}")
    
    
    if inplace:
        dataset.models = unique_files
    
    if plt_fig:
        labels = ['Unique Files', 'Duplicate Files']
        sizes = [len(unique_files), len(dataset) - len(unique_files)]
        colors = ['green', 'red']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title("Proportion of Unique vs. Duplicate Files")
        plt.show()


def tfidf_near_duplicate_detector(
    dataset: Dataset, 
    key='names', 
    threshold: float=TFIDF_DUPLICATE_THRESHOLD, 
    inplace: bool=False, 
    plt_fig: bool=False
):
    """
    Detect near-duplicate UML models based on TF-IDF vectorization and cosine similarity.
    
    This function identifies near-duplicate models by computing TF-IDF vectors
    for model text content and measuring their cosine similarity. Models with
    similarity above the threshold are considered near-duplicates.
    
    Args:
        dataset (UMLDataset): The dataset containing UML models.
        threshold (float): The similarity threshold for considering two models as near-duplicates.
            Values range from 0 to 1, with 1 being identical. Defaults to TFIDF_DUPLICATE_THRESHOLD.
        inplace (bool): If True, removes near-duplicates from the dataset. Defaults to False.
        plt_fig (bool): If True, displays a pie chart of unique vs. near-duplicate files. Defaults to False.
    
    Returns:
        tuple: A tuple containing:
            - List[UMLModel]: A list of unique UML models.
            - List[tuple]: A list of near-duplicate groups, where each group is a tuple of (model1, model2).
    
    Example:
        >>> unique_models, near_duplicate_groups = tfidf_near_duplicate_detector(dataset, threshold=0.85)
        >>> print(f"Found {len(near_duplicate_groups)} near-duplicate groups with threshold {threshold}")
    """
    
    
    # Extract the text content from the models
    start_time = time.time()
    text_data = list()
    for model in dataset:
        if key in ['names', 'names_with_types']:
            content = "\n".join(model.names_with_types if hasattr(model, 'names_with_types') else model.names) + "\n"
        else:
            content = get_model_text(model, key) + "\n"
        text_data.append(content)
        
    print(f"Extracted text data from {len(text_data)} models.")
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Detect duplicates with similarity threshold
    threshold = 0.8
    duplicate_groups = []
    visited = set()

    def find_duplicates():
        for i in range(len(text_data)):
            if i in visited:
                continue
            group = [i]
            for j in range(i + 1, len(text_data)):
                if cosine_sim[i, j] >= threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) > 1:
                duplicate_groups.append(group)
            visited.add(i)

    find_duplicates()
    
    total_duplicate_groups = len(duplicate_groups)
    total_duplicate_files = sum(len(group) for group in duplicate_groups)
    
    unique_indices = set(range(len(text_data))) - set(idx for group in duplicate_groups for idx in group[1:])
    unique_files = [text_data[i] for i in unique_indices]
    total_unique_files = len(unique_files)

    # Display results
    similarity_df = pd.DataFrame(cosine_sim, columns=[f"Doc {i+1}" for i in range(len(text_data))], 
                                index=[f"Doc {i+1}" for i in range(len(text_data))])
    
    end_time = time.time()
    print(f"TF-IDF vectorization took {end_time - start_time:.2f} seconds.")
    print("Cosine Similarity Matrix:")
    print(similarity_df)
    print(f"Total Duplicate Groups: {total_duplicate_groups}")
    print(f"Total Duplicate Files: {total_duplicate_files}")
    print(f"Total Unique Files: {total_unique_files}")


    
    if inplace:
        unique_files = [model for model in dataset.models if model not in [dup[1] for dup in duplicate_groups]]
        dataset.models = unique_files
    
    if plt_fig:
        labels = ['Unique Files', 'Near-Duplicate Files']
        sizes = [total_unique_files, total_duplicate_groups]
        colors = ['green', 'red']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title("Proportion of Unique vs. Near-Duplicate Files")
        plt.show()
    
    return unique_files, duplicate_groups