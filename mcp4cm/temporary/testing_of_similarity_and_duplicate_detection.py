from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from collections import Counter


doc1 = 'Hello'
doc2 = 'Hello Good Morning'
doc3 = 'Morning'
doc4 = 'this is different, group i am not a part, some words'
doc5 = 'hello I am not part of any good group even though I have some words'
doc6 = doc2 + ' ' + doc3

text_data = [doc1, doc2, doc4, doc3,doc5,  doc6]
content_series = pd.Series(text_data)

tfidf = TfidfVectorizer()

features = tfidf.fit_transform(content_series)

print('Cosine Similarity:')
print(cosine_similarity)

threshold = 0.5
cosine_distance_threshold = 1-threshold

connectivity_graph = radius_neighbors_graph(features,
                                            radius=cosine_distance_threshold,
                                            mode='distance',
                                            metric='cosine')
print('Connectivity Graph Done')
print(connectivity_graph)

print('Calculating connected Components')

n_components, labels = connected_components(connectivity_graph, directed=False, return_labels=True)

print('Calculation of connected components Done')
print(n_components)
print(labels)

counts = Counter(labels)

u, index, inverse, np_counts = np.unique(labels,return_index=True,return_inverse=True, return_counts=True)

print(counts)

print(np_counts)


labels==u[np_counts>1]

unique_file_mask = [counts[label]==1 for label in labels]

indices_of_unique_files = content_series.index[unique_file_mask]

df[indices_of_unique_files]