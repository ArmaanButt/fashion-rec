import pandas as pd
import numpy as np

from fastapi import FastAPI

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from openai import OpenAI


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "OPENAI_API_KEY"
    OPENAI_API_KEY_PERSONAL: str = "OPENAI_API_KEY_PERSONAL"
    LLM_MODEL: str = "LLM_MODEL"
    EMBEDDING_MODEL: str = "EMBEDDING_MODEL"

    class Config:
        env_file = ".env"


class Product(BaseModel):
    title: str
    average_rating: float
    rating_number: int
    price: float
    store: str
    thumbnail: str


class QueryList(BaseModel):
    queries: list[str]


class RecommendationRequest(BaseModel):
    query: str


class RecommendationResponse(BaseModel):
    response: str
    products: list[Product] | None = None


settings = Settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY_PERSONAL)
app = FastAPI()

df_products = pd.read_json("./data/sample_data_with_embeddings.jsonl", lines=True)


class ProductDatabase:
    def __init__(self):
        self.products = []


class QueryExpansion:
    @staticmethod
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
    query_embeddings: list[list[float]], top_k: int = 5
) -> list[Product]:
    # Calculate cosine similarity between query embeddings and product embeddings
    similarities = np.dot(query_embeddings, df_products["embedding"].T)

    # Get the top 5 most similar products for each query
    top_indices = np.argsort(similarities, axis=1)[:, -1 * top_k :]

    # Get the products for the top indices
    products = df_products.iloc[top_indices]

    return products


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recommendations")
async def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    queries = QueryExpansion.expand_query(request.query).queries

    embeddings = get_embeddings(queries)

    products = find_similar_products(embeddings)

    return RecommendationResponse(response=request.query, products=products)
