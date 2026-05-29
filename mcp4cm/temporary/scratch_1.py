from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd


c1 = Counter(['Text' , 'Title'])
c2 = Counter(['Text',  'Title'])
c3 = Counter(['Picture', 'Title'])


doc1 = 'Hello Good Bye'
doc2 = 'Hello Good Morning'
doc3 = 'Morning A long content series which is not present in another document'

content_series = pd.Series([doc1, doc2, doc3])
types_series = pd.Series([" ".join(c1.elements()), " ".join(c2.elements()), " ".join(c3.elements())])


finished_series = pd.Series([' '.join(texts) for texts in zip(content_series, types_series)])

print(finished_series)
tfidf = TfidfVectorizer()

features = tfidf.fit_transform(finished_series)

threshold = 0.5
cosine_distance_threshold = 1-threshold

similarity = cosine_similarity(features)


print("Cosine Similarity Matrix:")
# Display results
similarity_df = pd.DataFrame(similarity, columns=[f"Doc {i + 1}" for i in range(len(finished_series))],
                             index=[f"Doc {i + 1}" for i in range(len(finished_series))])
print(similarity_df)


