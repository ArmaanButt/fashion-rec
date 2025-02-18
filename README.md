# FDE Takehome Assignment

Semantic Fashion Recommendation

## Backlog

- [ ] Diagram for system
- [ ] Frontend chat
- [ ] Generate API Key
- [ ] Embed product titles
- [ ] Extract additional metadata

You will prototype a new semantic recommendation feature for an e-commerce website's fashion product line using the OpenAI API. The service will provide an interactive chat experience where users can have natural conversations about their fashion needs, rather than just receiving search results.
:

- Natural language chat interface using OpenAI's GPT models
- Contextual product recommendations based on user's needs and preferences
- Ability to ask follow-up questions and refine recommendations
- Integration with product catalog for accurate, real-time suggestions

Example conversation:
User: "I need an outfit to go to the beach this summer"
Assistant: "I'll help you find the perfect beach outfit! Are you looking for something casual or dressy?"
User: "Casual and comfortable"
Assistant: _Recommends relevant beach-appropriate items with explanations_

You will expose this functionality through a chat-based API endpoint, with an optional front-end demo showing the conversational shopping experience. The system should maintain context throughout the conversation to provide increasingly personalized recommendations.

## Example User Queries

- "I need something light and breezy for a hot summer day."
- "What should I wear to stay warm this winter?"
- "I’m looking for a dress that’s perfect for a wedding."
- "I need a smart outfit for an important business meeting.":
- "I need comfortable activewear for my morning jog."
- "What can I wear to a beach party?":
- "I'm after a trendy, casual look for hanging out with friends."
- "Show me some vintage-inspired outfits.":
- "I need an outfit that makes me feel confident and unique."
- "Help me find something stylish to wear tonight."

## Project Setup

### Prerequisites

- Python 3.12.7
- pip (Python package installer)

### Installation Steps

1. Clone the repository

```
fastapi dev src/main.py
```
