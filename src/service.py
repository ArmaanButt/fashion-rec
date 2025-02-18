import numpy as np

from openai import OpenAI

from models import Product, QueryList

from .config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY_PERSONAL)


def expand_query(query: str) -> QueryList:
    response = client.beta.chat.completions.parse(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                                You are a helpful assistant that expands queries. 
                                Based on the user query, you will expand the query to include more relevant products. 
                                Your output will be used to do a similarity search on a product database.
                                Return a list of expanded queries with only three queries.
                                """,
            },
            {"role": "user", "content": query},
        ],
        response_format=QueryList,
    )
    return response.choices[0].message.parsed


# Simple function to take in a list of text objects and return them as a list of embeddings
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input):
    print(input)
    response = client.embeddings.create(input=input, model=settings.EMBEDDING_MODEL)
    return [data.embedding for data in response.data]


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
