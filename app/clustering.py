from sklearn.cluster import KMeans
import numpy as np

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Cluster the embeddings using KMeans.

    Args:
        embeddings (np.ndarray): Array of embeddings
        num_clusters (int): Number of clusters

    Returns:
        np.ndarray: Array of cluster labels
    """

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    return clusters
