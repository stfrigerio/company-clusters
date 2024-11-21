from app.data_loaders import load_json_data, create_df
from app.semantic_embeddings import create_embeddings
from app.clustering import cluster_embeddings
from app.visualisation import plot_clusters

data = load_json_data('jsons/BioTech_data.json')
df = create_df(data)

embeddings = create_embeddings(df)
clusters = cluster_embeddings(embeddings, 3)
df['cluster'] = clusters

# plot_clusters(df, embeddings)
