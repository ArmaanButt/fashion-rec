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


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recommendations")
async def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    queries = QueryExpansion.expand_query(request.query).queries

    embeddings = get_embeddings(queries)

    return RecommendationResponse(response=request.query)
