import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import textwrap

def plot_clusters(df: pd.DataFrame, embeddings: np.ndarray, cluster_info: dict, save_as_image: bool = False):
    """
    Plot the clusters using t-SNE with descriptions.

    Args:
        df (pd.DataFrame): DataFrame containing the company data.
        embeddings (np.ndarray): Array of embeddings.
        cluster_info (dict): Dictionary with cluster descriptions and themes.
        save_as_image (bool): Whether to save the plot as an image.

    Returns:
        None. It displays the plot or saves it as an image.
    """
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    unique_clusters = df['cluster'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    
    n_clusters = len(unique_clusters)
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    fig, (ax_scatter, ax_legend) = plt.subplots(1, 2, figsize=(15, 8), 
        gridspec_kw={'width_ratios': [2, 1]})
    
    for cluster in unique_clusters:
        mask = df['cluster'] == cluster
        ax_scatter.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=[cluster_colors[cluster]],
            label=f"Cluster {cluster}",
            s=50
        )

    # Add company annotations
    for i, company in enumerate(df['Company']):
        ax_scatter.annotate(
            company, 
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
            fontsize=8,
            alpha=0.75
        )

    ax_scatter.set_title("Company Clustering Analysis", fontsize=14)
    ax_scatter.set_xlabel("Similarity Dimension 1", fontsize=10)
    ax_scatter.set_ylabel("Similarity Dimension 2", fontsize=10)

    ax_legend.axis('off')
    y_start = 0.95
    y_spacing = 0.12
    
    for i, cluster in enumerate(unique_clusters):
        info = cluster_info[str(cluster)]
        y_pos = y_start - (i * y_spacing)
        
        # Add colored dot to understand which cluster is which
        ax_legend.scatter(0.05, y_pos, 
            c=[cluster_colors[cluster]], 
            s=100,  
            transform=ax_legend.transAxes)
        
        wrapped_theme = textwrap.fill(info['cluster_theme'], width=80)
        cluster_text = (f"Cluster {cluster}:\n"
                    f"{info['cluster_name']}\n" # this should be bold
                    f"• {wrapped_theme}")
        
        ax_legend.text(0.15, y_pos,
            cluster_text,
            transform=ax_legend.transAxes,
            verticalalignment='center',
            fontsize=9,  
            linespacing=1.5)  

    fig.set_size_inches(20, 12)  
    plt.tight_layout()

    if save_as_image:
        plt.savefig("company_clusters.png", dpi=300, bbox_inches='tight')
    
    plt.show()