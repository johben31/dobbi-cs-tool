from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

RESPONSE_PROMPT = """You are a helpful customer service assistant for Dobbi, a Dutch dry cleaning company with pickup and delivery service.

GUIDELINES:
- Tone: Friendly, helpful, professional but not stiff
- Use "we" when referring to Dobbi
- Match the customer's language (Dutch or English)
- Be specific with prices and details when available
- If info is not in the knowledge base, say you'll look into it
- Keep responses concise but complete
- Sign off with "Groetjes, Team Dobbi" (Dutch) or "Best regards, Team Dobbi" (English)

CUSTOMER MESSAGE:
{customer_message}

CATEGORY: {category}

RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{retrieved_context}

Write a helpful response to the customer."""

class ResponseGenerator:
    def __init__(self):
        self.client = Anthropic()
    
    def generate(self, customer_message: str, category: str, retrieved_docs: list[dict]) -> dict:
        context = "\n\n".join([
            f"[Source: {doc['metadata']['source']}]\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        if not context:
            context = "No specific information found in knowledge base."
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": RESPONSE_PROMPT.format(
                    customer_message=customer_message,
                    category=category,
                    retrieved_context=context
                )
            }]
        )
        
        confidence = self._estimate_confidence(retrieved_docs)
        
        return {
            "draft_response": response.content[0].text,
            "sources_used": [doc['metadata']['source'] for doc in retrieved_docs],
            "confidence": confidence
        }
    
    def _estimate_confidence(self, docs: list[dict]) -> float:
        if not docs:
            return 0.3
        distances = [d.get('distance', 0.5) for d in docs if d.get('distance') is not None]
        if not distances:
            return 0.6
        avg_distance = sum(distances) / len(distances)
        confidence = max(0.3, min(0.95, 1 - avg_distance))
        return round(confidence, 2)


if __name__ == "__main__":
    generator = ResponseGenerator()
    
    sample_docs = [
        {
            "content": "Winter coat (Winterjas): €26.50",
            "metadata": {"source": "price_list"},
            "distance": 0.25
        },
        {
            "content": "Shirt (Overhemd): €4.95 - Washed and pressed",
            "metadata": {"source": "price_list"},
            "distance": 0.30
        }
    ]
    
    result = generator.generate(
        customer_message="Hoeveel kost het om een winterjas en 2 overhemden te laten reinigen?",
        category="pricing",
        retrieved_docs=sample_docs
    )
    
    print("Generated response:")
    print("-" * 50)
    print(result["draft_response"])
    print("-" * 50)
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Sources: {result['sources_used']}")