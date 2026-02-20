from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

CLASSIFICATION_PROMPT = """Classify this customer service message for Dobbi, a Dutch dry cleaning company.

Categories:
- pricing: Questions about costs, quotes, minimum order
- order_status: Questions about where an order is, tracking, delays
- service_info: Questions about what services exist, what can be cleaned
- pickup_delivery: Scheduling, changing times, missed pickups
- complaints: Damage, quality issues, lost items, stains not removed
- payment: Payment methods, invoices, refunds
- starter_kit: Laundry bag issues, not received, lost bag
- new_customer: Signing up, how to start, business inquiries
- other: Doesn't fit above categories

Customer message:
{message}

Respond ONLY with valid JSON (no other text):
{{
    "category": "one of the categories above",
    "confidence": 0.0 to 1.0,
    "entities": {{
        "order_id": null or "the order number if mentioned",
        "garment_types": ["list", "of", "items"],
        "language": "nl" or "en"
    }},
    "sentiment": "positive" or "neutral" or "negative" or "frustrated"
}}"""

class QuestionClassifier:
    def __init__(self):
        self.client = Anthropic()
    
    def classify(self, message: str) -> dict:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(message=message)
            }]
        )
        
        response_text = response.content[0].text
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = {
                "category": "other",
                "confidence": 0.5,
                "entities": {"language": "unknown"},
                "sentiment": "neutral"
            }
        
        return result


if __name__ == "__main__":
    classifier = QuestionClassifier()
    
    test_messages = [
        "Hoeveel kost het om een winterjas te reinigen?",
        "Where is my order #12345? It's been a week!",
        "I received my shirt back but the stain is still there!",
        "Can I pay by invoice?"
    ]
    
    for msg in test_messages:
        print(f"\nMessage: {msg}")
        result = classifier.classify(msg)
        print(f"Category: {result['category']} ({result['confidence']:.0%})")
        print(f"Sentiment: {result['sentiment']}")