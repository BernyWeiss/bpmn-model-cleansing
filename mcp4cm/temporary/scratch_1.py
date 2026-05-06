from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd



#a = np.array([1,1,0])
#b = np.array([0,1,1])
#c = np.array([1,1,1])

#matrix = cosine_similarity([a,c,b])
#print(matrix)

#x = np.average([a,b], axis=0)
#print(x)
#y = np.average([x,c],weights=[2,1], axis=0)
#print (y)


doc1 = 'Hello'
doc2 = 'Hello Good Morning'
doc3 = 'Morning'


text_data = [doc2, doc1, doc3]

tfidf = TfidfVectorizer()

features = tfidf.fit_transform(text_data)

threshold = 0.5
cosine_distance_threshold = 1-threshold

similarity = cosine_similarity(features)

print(similarity)

# Detect duplicates with similarity threshold
duplicate_groups = []
visited = set()


def find_duplicates():
    for i in range(len(text_data)):
        if i in visited:
            continue
        group = [i]
        for j in range(i + 1, len(text_data)):
            if similarity[i, j] >= threshold:
                print(f'Found {j} to be a duplicate of {i}')
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
similarity_df = pd.DataFrame(similarity, columns=[f"Doc {i + 1}" for i in range(len(text_data))],
                             index=[f"Doc {i + 1}" for i in range(len(text_data))])

print("Cosine Similarity Matrix:")
print(similarity_df)
print(f"Total Duplicate Groups: {total_duplicate_groups}")
print(f"Total Duplicate Files: {total_duplicate_files}")
print(f"Total Unique Files: {total_unique_files}")