from pydantic import BaseModel


class Product(BaseModel):
    """
    Represents a fashion product with its key attributes.
    
    Attributes:
        title (str): The name/description of the product
        average_rating (float): Average customer rating (0-5 scale)
        rating_number (int): Number of customer ratings
        price (float): Product price in USD
        store (str): Name of the store selling the product
        thumbnail (str): URL to product image thumbnail
    """
    title: str
    average_rating: float
    rating_number: int
    price: float
    store: str
    thumbnail: str

class ProductValidationResponse(BaseModel):
    """
    Response from product validation against a query.
    
    Attributes:
        answer (bool): True if product matches query criteria, False otherwise
    """
    answer: bool
    # reason: str  # Commented out but could be used for validation explanation


class RecommendationRequest(BaseModel):
    """
    Request model for fashion product recommendations.
    
    Attributes:
        query (str): User's natural language search query
        llmResponse (bool | None): Whether to include AI-generated natural language response.
                                 Optional, defaults to None.
    """
    query: str
    llmResponse: bool | None = None

class RecommendationResponse(BaseModel):
    """
    Response model containing product recommendations.
    
    Attributes:
        response (str): Natural language response describing recommendations
        products (list[Product] | None): List of recommended products matching the query.
                                       Optional, defaults to None.
    """
    response: str
    products: list[Product] | None = None
