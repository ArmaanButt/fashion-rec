from pydantic import BaseModel


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
