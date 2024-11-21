from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Create embeddings for the combined text in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the combined text

    Returns:
        np.ndarray: Array of embeddings
    """
    
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    return embeddings