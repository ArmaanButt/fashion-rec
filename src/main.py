import time

from fastapi import FastAPI

from pandarallel import pandarallel

from service import (
    expand_query,
    get_embeddings,
    validate_single_product,
    generate_recommendation_response,
    map_dataframe_to_products,
)
from models import RecommendationRequest, RecommendationResponse
from database import ProductDatabase

pandarallel.initialize()

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
    start_time = time.time()

    user_query = request.query

    queries = expand_query(user_query).queries

    print("Expanded Query")
    print(time.time() - start_time)

    embeddings = get_embeddings(queries)
    print("Generated embeddings")
    print(time.time() - start_time)

    product_results = db.find_similar_products(embeddings)
    print("Found similar products")
    print(time.time() - start_time)
    # Filter out products with average rating less than 3
    product_results = product_results[product_results["average_rating"] >= 3]

    # Filter products to only those that match the query according to the vision model
    validated_products = product_results[
        product_results.parallel_apply(
            validate_single_product, axis=1, query=user_query
        ).map(lambda x: x.answer)
    ]

    print("Validate with images")
    print(time.time() - start_time)

    # recommendation_response = generate_recommendation_response(
    #     validated_products, user_query
    # )

    # print("Generating answer")
    # print(time.time() - start_time)

    validated_products_list = map_dataframe_to_products(validated_products)

    print("Generating answer")
    print(time.time() - start_time)

    # TODO
    return RecommendationResponse(
        response="Placeholder", products=validated_products_list
    )
