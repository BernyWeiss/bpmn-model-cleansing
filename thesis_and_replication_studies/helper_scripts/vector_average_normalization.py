import numpy as np

# Example vectors
vectors = np.array([
    [1.0, 2.0, 3.0],
    [0.5, 0.1, 0.0],
    [4.0, 0.0, 1.0]
])

# L2-normalize each vector
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / norms

# Average the normalized vectors
avg_vector = normalized.mean(axis=0)

print("Normalized vectors:\n", normalized)
print("\nAverage vector:\n", avg_vector)

norm_avg = np.linalg.norm(avg_vector, keepdims=True)

print("\nNormalized average vector:\n", norm_avg)

norm_all  = np.linalg.norm(normalized, axis=1, keepdims=True)

print("\nNormalized all vector:\n", norm_all)
print("\nAverage normalized vector:\n", normalized)