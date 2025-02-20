import time
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAIError

from service import (
    expand_query,
    get_embeddings,
    validate_product_with_query,
    generate_recommendation_response,
    map_dataframe_to_products,
)
from models import RecommendationRequest, RecommendationResponse
from database import ProductDatabase

# Loads processed data with embeddings into memory
db = ProductDatabase()

# Create FastAPI application instance
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serves the main HTML interface for the fashion recommendation system.
    
    Returns:
        HTMLResponse: A simple web interface for testing the recommendation system
    """
    html_content = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Fashion Product Recommendation</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          .container { max-width: 600px; margin: auto; }
          input[type="text"] { width: 100%; padding: 10px; margin-bottom: 10px; }
          button { padding: 10px 20px; }
          .product { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
          .thumbnail { max-width: 100px; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Fashion Product Recommendation</h1>
          <input type="text" id="query" placeholder="Enter your search...">
          <br>
          <label>
            <input type="checkbox" id="llmResponse"> Include natural language response
          </label>
          <br>
          <button onclick="submitQuery()">Search</button>
          <div id="results"></div>
        </div>
        <script>
          async function submitQuery() {
            const query = document.getElementById('query').value;
            const llmResponse = document.getElementById('llmResponse').checked;
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = "<p>Loading...</p>";
    
            const payload = {
              query: query,
              llmResponse: llmResponse
            };
    
            try {
              const res = await fetch("/recommendations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
              });
              const data = await res.json();
              let html = "<h2>Recommendations</h2>";
              html += `<p>${data.response}</p>`;
              if(data.products && data.products.length > 0) {
                data.products.forEach(product => {
                  html += `<div class="product">
                            <img src="${product.thumbnail}" alt="${product.title}" class="thumbnail">
                            <h3>${product.title}</h3>
                            <p>Store: ${product.store}</p>
                            <p>Rating: ${product.average_rating} (${product.rating_number} reviews)</p>
                            <p>Price: $${product.price}</p>
                          </div>`;
                });
              } else {
                html += "<p>No products found.</p>";
              }
              resultsDiv.innerHTML = html;
            } catch (error) {
              resultsDiv.innerHTML = "<p>Error retrieving recommendations.</p>";
              console.error(error);
            }
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post(
    "/recommendations",
    summary="Parses a user's natural-language query and finds relevant products.",
)
async def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Process a natural language query and return relevant fashion recommendations.
    
    Args:
        request (RecommendationRequest): Contains the user's query and response preferences
        
    Returns:
        RecommendationResponse: Product recommendations and optional natural language response

    """

    start_time = time.time()
    user_query = request.query
    llm_response_requested = request.llmResponse
    recommendation_response = ""

    # Expand the user's query into related search terms
    try:
        queries = expand_query(user_query)
    except OpenAIError as e:
        raise HTTPException(
            status_code=503,
            detail="Error expanding query. Please try again later."
        ) from e

    # Validate the expanded queries
    if len(queries) == 0 or (len(queries) == 1 and queries[0] == ""):
        return RecommendationResponse(
            response="Sorry I can't help with that. Please rephrase your query to focus on fashion items.",
            products=[]
        )

    print(f"Query expansion completed in {time.time() - start_time:.2f}s")

    # Generate embeddings for semantic search
    try:
        embeddings = get_embeddings(queries)
    except OpenAIError as e:
        raise HTTPException(
            status_code=503,
            detail="Error generating embeddings. Please try again later."
        ) from e

    print(f"Embeddings generated in {time.time() - start_time:.2f}s")

    # Find similar products using vector search
    try:
        product_results = db.find_similar_products(embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error searching product database."
        ) from e

    print(f"Similar products found in {time.time() - start_time:.2f}s")

    # Sort products by rating for initial ranking
    product_results.sort_values("average_rating", ascending=False, inplace=True)

    # Validate products against the original query using concurrent processing
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor(max_workers=15) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    validate_product_with_query,
                    user_query,
                    row
                )
                for row in product_results.itertuples(index=False)
            ]
            results = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Error validating products. Please try again later."
        ) from e

    # Filter products based on validation results
    validated_products = product_results[results]
    print(f"Products validated in {time.time() - start_time:.2f}s")

    # Generate natural language response if requested
    if llm_response_requested:
        try:
            recommendation_response = generate_recommendation_response(
                validated_products,
                user_query
            )
        except OpenAIError as e:
            # Don't fail the request if response generation fails
            recommendation_response = (
                "I've found some products that match your request. "
                "Please take a look and see if any meet your needs."
            )
            print(f"Error generating response: {e}")

        print(f"Response generated in {time.time() - start_time:.2f}s")

    # Convert DataFrame to Product models
    try:
        validated_products_list = map_dataframe_to_products(validated_products)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error processing product data."
        ) from e

    return RecommendationResponse(
        response=recommendation_response,
        products=validated_products_list
    )

