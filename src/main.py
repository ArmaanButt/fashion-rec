from fastapi import FastAPI

from .service import expand_query, get_embeddings
from .models import RecommendationRequest, RecommendationResponse
from .database import ProductDatabase


db = ProductDatabase()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post(
    "/recommendations",
    summary="Parses a user's natural-language query and finds relevant products.",
)
async def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    queries = expand_query(request.query).queries

    embeddings = get_embeddings(queries)

    products = db.find_similar_products(embeddings)

    return RecommendationResponse(response=request.query, products=products)
