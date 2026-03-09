from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

RESPONSE_PROMPT = """You are a customer service assistant for Dobbi, a Dutch dry cleaning company.

RULES:
- Be concise. Only answer what the customer asked.
- Do NOT mention services or information the customer didn't ask about.
- Do NOT ask follow-up questions unless absolutely necessary.
- Use prices from the knowledge base. If a price is not available, say so briefly.
- Match the customer's language (Dutch or English).
- Keep responses short: 3-6 sentences max for simple questions.
- Sign off with "Groetjes, Team Dobbi" (Dutch) or "Best regards, Team Dobbi" (English).

CUSTOMER MESSAGE:
{customer_message}

CATEGORY: {category}

KNOWLEDGE BASE:
{retrieved_context}

Write a short, helpful response answering ONLY what the customer asked."""

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
            max_tokens=500,
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
        }
    ]
    
    result = generator.generate(
        customer_message="Hoeveel kost het om een winterjas te laten reinigen?",
        category="pricing",
        retrieved_docs=sample_docs
    )
    
    print("Generated response:")
    print(result["draft_response"])
