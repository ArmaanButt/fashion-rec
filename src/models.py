from pydantic import BaseModel


class Product(BaseModel):
    title: str
    average_rating: float
    rating_number: int
    price: float
    store: str
    thumbnail: str

class ProductValidationResponse(BaseModel):
    answer: bool
    # reason: str

class ValidQuery(BaseModel):
    is_valid: bool

class QueryList(BaseModel):
    queries: list[str]

class RecommendationRequest(BaseModel):
    query: str
    llmResponse: bool | None = None

class RecommendationResponse(BaseModel):
    response: str
    products: list[Product] | None = None
