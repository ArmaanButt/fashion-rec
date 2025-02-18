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
    # from concurrent.futures import ThreadPoolExecutor
    # import pandas as pd

    # def validate_row(row, query):
    #     # This function wraps your vision model validation
    #     # It returns the boolean answer (assumed to be in result.answer)
    #     result = validate_single_product(row, query=query)
    #     return result.answer

    # # Convert DataFrame rows to a list of dictionaries.
    # rows = product_results.to_dict(orient="records")

    # # Use ThreadPoolExecutor to concurrently validate each row.
    # with ThreadPoolExecutor() as executor:
    #     # executor.map returns results in the order of the input rows.
    #     validation_results = list(
    #         executor.map(lambda row: validate_row(row, user_query), rows)
    #     )

    # # Create a boolean mask using the validation results.
    # mask = pd.Series(validation_results, index=product_results.index)

    # # Filter the DataFrame with the mask.
    # validated_products = product_results[mask]
    print("Validate with images")
    print(time.time() - start_time)

    # recommendation_response = generate_recommendation_response(
    #     validated_products, user_query
    # )

    # print("Generated answer")
    # print(time.time() - start_time)

    validated_products_list = map_dataframe_to_products(validated_products)

    # TODO
    return RecommendationResponse(
        response="Placeholder", products=validated_products_list
    )
