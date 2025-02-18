from openai import OpenAI

from models import Product, QueryList, ProductValidationResponse
from utils import get_and_encode_image
from config import settings

from pydantic import ValidationError

client = OpenAI(api_key=settings.OPENAI_API_KEY_PERSONAL)


def expand_query(query: str) -> QueryList:
    response = client.beta.chat.completions.parse(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                          You are a helpful assistant that expands queries. 
                          Based on the user query, you will expand the query to include more relevant products.
                          If you are asked to help with an outfit, you will create queries
                          that expand the number of products.

                          Example: 
                          
                          Query: "I need a suit for prom."

                          Expanded Queries: ['Suit formal', 'Dress pants', 'Tied formal', 'Dress shoes black']

                          Your output will be used to do a similarity search on a product database.
                          Return a list of expanded queries with only 5 expanded queries.
                          """,
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format=QueryList,
    )
    return response.choices[0].message.parsed


def validate_product_with_query(query, product_title, product_image_base64):
    response = client.beta.chat.completions.parse(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f""" You are a fashion expert analyzing a clothing item.
                            You will be shown an image of a clothing item and given a text query.
                            Evaluate if this item matches the description in the query.
                            If the user was specific about what they were looking for, make sure the item matches that description.
                            Example: If the user is looking for a "mens wedding outfit", make sure the item is for an adult male and not a child.
                            
                            Provide your analysis in JSON format with two fields:
                            - "answer": Must be "True" or "False" indicating if the item matches the query
                            - "reason": A concise explanation of your decision if the item matches the query. Leave blank if the item does not match the query.
                            
                            Do not describe the item itself. Focus only on its relevance to the query.
                            
                            <query>
                            {query}
                            </query>

                            <product_title>
                            {product_title}
                            </product_title>
                            """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{product_image_base64}",
                        },
                    },
                ],
            }
        ],
        temperature=0,
        response_format=ProductValidationResponse,
    )

    return response.choices[0].message.parsed


def validate_single_product(product_row, query):
    """
    Validates a single product against the original query using its thumbnail image.
    Returns the validation response.
    """
    product_image_base64 = get_and_encode_image(product_row["thumbnail"])
    return validate_product_with_query(
        query, product_row["title"], product_image_base64
    )


def generate_recommendation_response(validated_products, original_query):
    """
    Generates a natural language response summarizing the validated product results.

    Args:
        validated_products (pd.DataFrame): DataFrame containing the validated products
        original_query (str): The original search query

    Returns:
        str: A natural language response describing the search results
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful shopping assistant. Summarize the search results in a natural, 
            conversational way. Include key details like number of results, price ranges if available, 
            and notable brands or features. Avoid any negative language when describing the products.

            Do not list the number of products that you are showing. Mention that you have found a few options that you think will work.
            encourage the user to look at the products and see if they like any of them and say there are more options if they want to see more.

            Here is a search query: {original_query}
            
            Here are the matching products:
            {validated_products.to_string()}
            
            Please provide a natural language summary of these results.
            """,
        }
    ]

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content


# Simple function to take in a list of text objects and return them as a list of embeddings
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input):
    print(input)
    response = client.embeddings.create(input=input, model=settings.EMBEDDING_MODEL)
    return [data.embedding for data in response.data]


def map_dataframe_to_products(df) -> list[Product]:
    products = []
    # Valid fields based on the Product model
    valid_fields = set(Product.model_fields.keys())

    for index, row in df.iterrows():
        row_data = row.to_dict()
        # Only keep the keys that match the Product model fields
        filtered_data = {
            key: value for key, value in row_data.items() if key in valid_fields
        }
        try:
            product = Product(**filtered_data)
            products.append(product)
        except ValidationError as e:
            print(f"Validation error on row {index}: {e}")
    return products
