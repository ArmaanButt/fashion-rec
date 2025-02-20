# Fashion Product Recommendation API
Submission By: Armaan Butt

## Technical Stack

- Python 3.12.8
- FastAPI for API endpoints
- OpenAI GPT-3.5-turbo for natural language processing
- text-embedding-3-small for semantic search
- Pydantic for data validation
- Pandas for data processing

## Quick Start

1. Unzip and cd into the repository:
```bash
cd semantic-fashion-rec
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Run the service:
```bash
fastapi dev src/main.py
```
5. Open a browser and navigate to:
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

## Example Use Cases

The system handles a wide range of fashion queries, including:

### Formal Events
- "I need an outfit for a winter ball"
- "I'm looking for a dress that's perfect for a wedding"
- "I need a smart outfit for an important business meeting"

### Casual Wear
- "I need something light and breezy for a hot summer day"
- "What can I wear to a beach party?"
- "I'm after a trendy, casual look for hanging out with friends"

### Themed Outfits
- "I want to wear something that is like a halloween costume but professional to a work party"
- "Show me some vintage-inspired outfits"
- "Date night outfit at the aquarium"

### Seasonal Fashion
- "What should I wear to stay warm this winter?"
- "I need comfortable activewear for my morning jog"

## Design Decisions

### Query Processing
- Uses GPT-3.5-turbo for query understanding and expansion
- Maintains conversation context for better recommendations
- Implements retry logic for API reliability

### Search Strategy
- Semantic search using embeddings for better matching
- Product validation to ensure relevance
- Natural language response generation for user-friendly output

### Performance Considerations
- Stateless design for scalability
- Efficient embedding storage and retrieval
- Optimized API calls to minimize latency

## Development

### Local Development
```bash
fastapi dev src/main.py
```

## Future Improvements

- [ ] Frontend chat interface
- [ ] API key authentication
- [ ] Expanded product metadata
- [ ] Caching layer for frequent queries
- [ ] User preference learning
- [ ] Performance optimization
- [ ] Additional product attributes
