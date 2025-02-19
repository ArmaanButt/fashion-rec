import pandas as pd
import numpy as np
from models import Product


class ProductDatabase:
    def __init__(self):
        self.df_products = pd.read_json(
            "./data/processed_data_with_embeddings.jsonl", lines=True
        )

    def find_similar_products(
        self, query_embeddings: list[list[float]], top_k: int = 3
    ) -> list[Product]:
        product_embeddings = np.array(
            list(self.df_products["embedding"].apply(lambda x: list(x)))
        )

        # Calculate cosine similarity between query embeddings and product embeddings
        similarities = np.dot(query_embeddings, product_embeddings.T)

        # Get the top 5 most similar products for each query
        top_indices = np.argsort(similarities, axis=1)[:, -1 * top_k :]

        top_indices = top_indices.ravel()

        # Get the products for the top indices
        products = self.df_products.iloc[top_indices]

        return products
