import pandas as pd
import numpy as np
from models import Product


class ProductDatabase:
    def __init__(self):
        self.df_products = pd.read_json(
            "./data/sample_data_with_embeddings.jsonl", lines=True
        )

    def find_similar_products(
        self, query_embeddings: list[list[float]], top_k: int = 5
    ) -> list[Product]:
        # Calculate cosine similarity between query embeddings and product embeddings
        similarities = np.dot(query_embeddings, self.df_products["embedding"].T)

        # Get the top 5 most similar products for each query
        top_indices = np.argsort(similarities, axis=1)[:, -1 * top_k :]

        # Get the products for the top indices
        products = self.df_products.iloc[top_indices]

        return products
