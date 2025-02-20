import re
import json

from openai import OpenAI
from pydantic import ValidationError
from tenacity import retry, wait_random_exponential, stop_after_attempt

from models import Product
from config import settings


client = OpenAI(api_key=settings.OPENAI_API_KEY)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def expand_query(query: str):
    """
    Expands a user's fashion query into multiple related search queries using gpt-3.5-turbo.

    Takes a single query string and returns up to 5 related search queries to help find
    relevant fashion products. For example, a query about a prom suit might expand to
    include dress pants and shoes.

    Args:
        query (str): The user's original search query

    Returns:
        list[str]: A list of expanded search queries, or empty list if query is invalid
    """

    prompt = """
            You are a helpful assistant that expands user queries to help find products for a fashion product line. 
            Based on the user query, you will expand the query to include more relevant products and return a list
            of new queries, up to 5.

            Your output will be used to do a similarity search on a product database.
            If you are asked to help with an outfit, you will create queries
            that expand the number of products.

            Example: 
            
            Query: "I need a suit for prom."

            Expanded Queries: ["Suit formal", "Dress pants", "Dress shoes black"]
            
            If the input is not appropriate or has nothing to do with searching for fashion products,
            or has a part of the query that has nothing to do with products return an empty list.

            Example: Show me car parts
            Response: [""]

            Do not include JSON tags. Use double quotes for the list.
            Remember If the input is not appropriate or has nothing to do with searching for fashion products
            return an empty list.
            User Query:
            """
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    queries_str = response.choices[0].message.content.strip()

    print(queries_str)

    try:
        queries = json.loads(queries_str)
        print(queries)
        return queries
    except Exception as e:
        print(e)
        return []


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def validate_product_with_query(query, product_row):
    """
    Validates if a product matches the user's search query using gpt-3.5-turbo.

    Analyzes product details against the query to determine relevance, considering factors
    like gender, formality, price range, and appropriateness for specific events.

    Args:
        query: The search query to validate against
        product_row: Product details including title, price, and rating

    Returns:
        bool: True if product matches query criteria, False otherwise
    """

    prompt = f"""
        You are a fashion expert analyzing a clothing item.
        You will be given a text query about a clothing item along with its details.
        Evaluate if this item matches the description in the query.

        Example: If the user is looking for a "mens wedding outfit", ensure the item is for an adult male and not a child.
        If the query mentions an event or location, consider the appropriate formality.
        If the query provides a price or range, take that into account when determining relevance.

        Provide answer as "True" or "False" only.

        Do not describe the item itself; focus solely on its relevance to the query.
        Do not include JSON tags.

        Query: {query}
        Product Title: {product_row.title}
        Price: {product_row.price}
        Average Rating: {product_row.average_rating}
    """

    messages = [{"role": "user", "content": prompt.strip()}]

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0,
    )

    answer_text = response.choices[0].message.content.strip()

    try:
        match = re.search(r"^(True|False)$", answer_text, re.IGNORECASE)
        if match:
            answer_str = match.group(0).lower()
            return True if answer_str == "true" else False
    except Exception:
        return False


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def generate_recommendation_response(validated_products, original_query):
    """
    Generates a natural language response summarizing the validated product results.

    Creates a conversational summary of search results, highlighting key product features,
    comparing options, and providing helpful suggestions.

    Args:
        validated_products (pd.DataFrame): DataFrame containing the validated products
        original_query (str): The original search query

    Returns:
        str: A natural language response describing the search results
    """
    messages = [
        {
            "role": "user",
            "content": f"""You are a helpful shopping assistant. Summarize the search results in a natural, 
            conversational way. Include key details like number of results, price ranges if available, 
            and notable brands or features. Avoid any negative language when describing the products.

            Do not list the number of products that you are showing. Mention that you have found a few options that you think will work.
            encourage the user to look at the products and see if they like any of them and say there are more options if they want to see more.

            Compare and contrast the results. 
            
            Example: This shirt is a great option but this other shirt has a higher rating.

            Search Query: {original_query}
            
            Here are the matching products:
            {validated_products.to_string()}
            
            Please provide a natural language summary of these results. If there are no products,
            Suggest ways the user can improve their search query.
            """,
        }
    ]

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.5,
    )

    return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def get_embeddings(input):
    """
    Generates embeddings for a list of text inputs using text-embedding-3-small.

    Takes text inputs and converts them into vector embeddings that are used for
    similarity search.

    Args:
        input: List of text strings to generate embeddings for

    Returns:
        list: List of embedding vectors for each input text
    """
    print(input)
    response = client.embeddings.create(input=input, model=settings.EMBEDDING_MODEL)
    return [data.embedding for data in response.data]


def map_dataframe_to_products(df) -> list[Product]:
    """
    Converts a pandas DataFrame into a list of Product model instances.

    Maps DataFrame rows to Product objects, validating the data against the
    Product model schema.

    Args:
        df (pd.DataFrame): DataFrame containing product data

    Returns:
        list[Product]: List of validated Product model instances
    """
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
