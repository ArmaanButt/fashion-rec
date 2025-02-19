import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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

pandarallel.initialize(nb_workers=4, verbose=0)

db = ProductDatabase()

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
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
            <input type="checkbox" id="llmResponse"> Include LLM explanation
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
    start_time = time.time()

    user_query = request.query
    llm_response_requested = request.llmResponse

    recommendation_response = ""

    queries = expand_query(user_query).queries

    # Validate user query
    if len(queries) == 0:
        raise Exception

    print("Expanded Query")
    print(time.time() - start_time)

    embeddings = get_embeddings(queries)
    print("Generated embeddings")
    print(time.time() - start_time)

    product_results = db.find_similar_products(embeddings)
    print("Found similar products")
    print(time.time() - start_time)

    product_results.sort_values('average_rating', ascending=False, inplace=True)
    # Filter products to only those that match the query according to the vision model
    validated_products = product_results[
        product_results.parallel_apply(
            validate_single_product, axis=1, query=user_query
        )
    ]

    print("Validate with images")
    print(time.time() - start_time)

    if llm_response_requested:
        recommendation_response = generate_recommendation_response(
            validated_products, user_query
        )

        print("Generated answer")
        print(time.time() - start_time)

    validated_products_list = map_dataframe_to_products(validated_products)

    return RecommendationResponse(
        response=recommendation_response, products=validated_products_list
    )
