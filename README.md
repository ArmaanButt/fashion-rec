# Fashion Product Recommendation

A small project to learn FastAPI and OpenAI APIs for building a fashion product recommendation system. 
The system leverages OpenAI's capabilities to provide personalized fashion recommendations based on user preferences and descriptions. 
Built with FastAPI for the backend API service, this project serves as a learning exercise in integrating modern AI services with web APIs.

## Project Structure

```
fashion-rec/
├── src/                    # Main application source code
│   ├── main.py            # FastAPI application entry point
│
├── data/                  # Processed dataset
│
├── docs/                  # Documentation and diagrams
│   ├── Armaan_Butt_Architecture_Diagram.pdf  # System architecture diagram
│
├── notebooks/            # Jupyter notebooks for experiments
├── requirements.txt      # Python dependencies
├── .env.sample          # Environment variables template
└── README.md            # Project documentation
```

## Prerequisites 

Python 3.12.8

## Quick Start

1. Unzip and cd into the repository:
```bash
cd fashion-rec
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables and add OpenAI API key:
```bash
cp .env.sample .env
# Add your OpenAI API key to .env
```

5. Run the service:
```bash
fastapi run src/main.py
```
6. Open a browser and navigate to:
```bash
http://localhost:8000/
```

## API Usage

### Example Request
```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{"query": "I need an outfit for a winter ball", "llmResponse": true}'
```

### Example Response
```json
{
    "recommendations": "I've found some elegant options for your winter ball...",
    "products": [
        {
            "title": "Evening Gown with Sequins",
            "price": 299.99,
            "rating": 4.8
        }
        // Additional products...
    ]
}
```

## Example Queries

### Query: "I need an outfit for a winter ball"
Response: "I found a few options for your winter ball outfit search. There is a stunning satin crepe gown in ultramarine from Halston priced at $89.99. Adrianna Papell offers a lace long sleeve evening gown for $179.99. For a tuxedo look, Neil Allyn has a set including a shirt, cummerbund, bow tie, cufflinks, and studs for $18.55. Another option from Neil Allyn is a tuxedo shirt set with a laydown collar in white for $26.00. There is also a tuxedo shirt set with a wing collar for $26.00. Sportoli offers a midlength down alternative puffer coat with fur trim and a detachable hood for $34.39. Lastly, Bangunlah has an elegant bridesmaid dress in lilac with double V-neck, long sleeves, and a high slit for $85.99. Take a look at these options and see if any catch your eye. If you need more choices, feel free to explore further!"

### Query: "I need pants for men between $10 - $20"
Response: "I found a few options for men's pants within the $10 - $20 price range. One option is the "Men Slim Fit Ripped Destroyed Jeans" by Fanteecy priced at $18.98. These jeans are described as vintage, stylish, and comfortable with a slim fit cut. Another option is the "Drawstring Pant Fashion Men's Sport Pure Color Bandage Casual Loose Sweatpants" by SERYU priced at $13.79. These sweatpants are made of cotton and polyester, feature a slim fit design, and are suitable for various casual occasions. Take a look at these options and see if any catch your eye. If you want more options, feel free to explore further."

### Query: "Give me chiptole"
Response: "Sorry I can't help with that. Please rephrase your query to focus on fashion items.

## Key Design Decisions and Next Steps

### Features for Vectors
- **Embedding Strategy:**  
  I used the `text-embedding-3-small` model to vectorize key product fields—**title, features, and description**. Although the features and description fields were sparse, they improved the semantic relevance of the search.
- **Image-Based Metadata (Experimental):**  
  I experimented with extracting metadata (formality, gender, and description) from product thumbnail images. Due to API key limitations, this feature wasn't incorporated into the prototype. In production, I would leverage the OpenAI batch API to generate image captions and embed them, further enhancing relevancy.

### Query Expansion
- **Handling High-Level Queries:**  
  Some user queries, such as "A men's outfit to the beach," are abstract and may miss related items such as sunglasses and flip flops.  
- **gpt-3.5-turbo for Expansion:**  
  To cover these gaps, I use gpt‑3.5‑turbo to generate more specific query variations ("Men's swim trunks," "Sunglasses," "Flip flops," "Sun hat," "Beach towel"). This improves search coverage and meets the need for a fast, customer-facing storefront (higher token throughput).

### Similarity Search
- **Prototype Implementation:**  
  The system computes semantic similarity using NumPy to measure distances between the query embedding and the precomputed product embeddings.
- **Scaling Considerations:**  
  For production, a vector database (such as PgVector or Pinecone) would be used to store and index embeddings efficiently, significantly improving retrieval performance as the dataset grows.

### Product Validation
- **Relevance Filtering via LLM:**  
  After retrieving candidate products from the expanded queries, the system validates each product's relevance by passing the original user query and product information to an LLM. This step is particularly useful for handling queries with specific constraints such as price ranges or nuanced formality requirements such as "date night at an aquarium".
- **Parallelization with ThreadPoolExecutor:**  
  Since this validation involves network-bound I/O calls (sending requests to the OpenAI API) that are not CPU intensive, I parallelized the process using `concurrent.futures.ThreadPoolExecutor` with 15 workers. This approach significantly mitigates network latency by processing multiple validations concurrently, ensuring that the overall validation step is more efficient.
- **Production Enhancement:**  
  In production, I would further optimize this process by pre-processing or batching image data to extract captions before passing them to the LLM, thereby reducing real-time network overhead.

### Summarization
- **Enhanced User Feedback:**  
  In addition to validating product relevance, the system uses an LLM to generate a natural language summary explaining the recommendation. For example:  
  > "These recommendations were selected for their lightweight fabrics and summer-friendly styles, matching your request for a casual beach outfit."
- **Transparency and Trust:**  
  This summarization builds user trust by clearly communicating the rationale behind the recommendations, aligning with a customer-focused approach.


## Next Steps for the Application

- **Performance Optimization:**  
    - **Caching & Precomputation:**  
    Implement prompt caching for query expansion and image processing steps to reduce latency.  
    - **Asynchronous & Parallel Processing:**  
    Continue optimizing I/O-bound tasks (like product validation) with asynchronous processing and ensure that the ThreadPoolExecutor configuration scales with the load.

- **Vector Database Integration:**  
    Transition from a NumPy-based similarity search to a dedicated vector database (PgVector or Pinecone) for efficient retrieval as product data scales.

- **Enhanced Image Processing:**  
  - Develop a batch processing pipeline to pre-generate image captions and feature embeddings, reducing real-time processing delays.

- **Improve the User Experience:**  
    - Creating a chat interface and leverage the Assistant API instead the search interface.
    - The assistant API can be used to maintain a conversation history and context and be passed user information to provide a more personalized experience.
    - Utilizing streaming to reduce the perceived latency of the response.
- **Robust Error Handling & Monitoring:**  
  - Expand error handling for external API calls and integrate logging and monitoring systems to track performance and detect issues.

- **Customer Integration & Collaboration:**  
    - Gather user and customer feedback to iteratively refine the semantic search and validation algorithms.
    - Prepare detailed scopes of work and project plans for transitioning from the MVP to full production deployments.
    - Engage with strategic customers and internal teams (Sales, Technical Success, Applied) to refine the solution based on real-world requirements.
