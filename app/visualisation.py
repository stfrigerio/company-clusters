import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_clusters(df: pd.DataFrame, embeddings: np.ndarray):
    """
    Plot the clusters using t-SNE.

    Args:
        df (pd.DataFrame): DataFrame containing the company data
        embeddings (np.ndarray): Array of embeddings

    Returns:
        None, it simply displays the plot
    """

    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50)

    for i, company in enumerate(df['Company']):
        plt.annotate(company, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.75)
    
    plt.colorbar()
    plt.title("Company Clusters")
    plt.show()
