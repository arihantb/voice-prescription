from dtw import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def train_model(patient_patterns, clustering_algorithm='kmeans'):
    """
    Train a model using clustering on the patient patterns.

    Parameters:
    - patient_patterns (list): List of preprocessed patient patterns.
    - clustering_algorithm (str): Clustering algorithm to use ('kmeans' or 'gmm').

    Returns:
    - model (list): Trained model.
    """
    if not patient_patterns:
        raise ValueError("No patient patterns provided for training.")

    # Combine patient patterns into a single matrix
    combined_patterns = np.concatenate(patient_patterns, axis=1).T

    # Standardize the data
    scaler = StandardScaler()
    scaled_patterns = scaler.fit_transform(combined_patterns)

    # Apply clustering algorithm
    if clustering_algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=len(patient_patterns))
        kmeans.fit(scaled_patterns)
        labels = kmeans.labels_
    elif clustering_algorithm == 'gmm':
        gmm = GaussianMixture(n_components=len(patient_patterns))
        gmm.fit(scaled_patterns)
        labels = gmm.predict(scaled_patterns)
    else:
        raise ValueError("Invalid clustering algorithm. Use 'kmeans' or 'gmm'.")

    # Assign each pattern to a cluster
    clusters = {}
    max_length = max(pattern.shape[1] for pattern in patient_patterns)
    
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        if i < len(patient_patterns):
            # Reshape to have a consistent size along the time axis
            reshaped_pattern = np.zeros((patient_patterns[i].shape[0], max_length))
            reshaped_pattern[:, :patient_patterns[i].shape[1]] = patient_patterns[i]
            clusters[label].append(reshaped_pattern)
        else:
            print(f"Warning: Index {i} exceeds the length of patient_patterns.")

    # Calculate the centroid of each cluster
    centroids = []
    for cluster in clusters.values():
        if cluster:
            # Take the mean along the last axis (axis=-1) assuming it's the time axis
            cluster_mean = np.mean(np.array(cluster), axis=-1)
            centroids.append(cluster_mean)

    return centroids
