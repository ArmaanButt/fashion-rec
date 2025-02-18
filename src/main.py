from fastapi import FastAPI

from service import (
    expand_query,
    get_embeddings,
    validate_single_product,
    generate_recommendation_response,
)
from models import RecommendationRequest, RecommendationResponse
from database import ProductDatabase


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
    user_query = request.query

    queries = expand_query(user_query).queries

    embeddings = get_embeddings(queries)

    product_results = db.find_similar_products(embeddings)

    # Filter out products with average rating less than 3
    product_results = product_results[product_results["average_rating"] >= 3]

    # Filter products to only those that match the query according to the vision model
    validated_products = product_results[
        product_results.apply(validate_single_product, axis=1, query=user_query).map(
            lambda x: x.answer
        )
    ]

    recommendation_response = generate_recommendation_response(
        validated_products, user_query
    )

    return RecommendationResponse(response=recommendation_response, products=None)
