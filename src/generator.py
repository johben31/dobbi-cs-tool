from anthropic import Anthropic
from dotenv import load_dotenv
import re
load_dotenv()

def anonymize_message(text: str) -> tuple[str, str]:
    """Remove personal data before sending to Claude, but keep first name.
    Returns: (anonymized_text, first_name or None)"""
    
    # Try to extract first name from common patterns
    first_name = None
    
    # Pattern: "I'm [Name]" or "I am [Name]" or "My name is [Name]"
    name_match = re.search(r"(?:I'm|I am|my name is|this is|it's|ik ben|mijn naam is)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    if name_match:
        first_name = name_match.group(1).capitalize()
    
    # Pattern: "Hi, [Name] here" or signature at end
    if not first_name:
        name_match = re.search(r"(?:^|\n)(?:groetjes|groeten|regards|cheers|thanks|mvg|met vriendelijke groet)[,\s]*([A-Z][a-z]+)", text, re.IGNORECASE)
        if name_match:
            first_name = name_match.group(1).capitalize()
    
    # Pattern: Name at very end of message (common in emails)
    if not first_name:
        name_match = re.search(r"\n([A-Z][a-z]+)\s*$", text)
        if name_match:
            first_name = name_match.group(1).capitalize()
    
    anonymized = text
    
    # Remove email addresses
    anonymized = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', anonymized)
    
    # Remove phone numbers (Dutch and international formats)
    anonymized = re.sub(r'(\+31|0031|0)[1-9][0-9]{8}', '[PHONE]', anonymized)
    anonymized = re.sub(r'\b06[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b', '[PHONE]', anonymized)
    
    # Remove postcodes (Dutch format: 1234 AB)
    anonymized = re.sub(r'\b[1-9][0-9]{3}\s?[A-Za-z]{2}\b', '[POSTCODE]', anonymized)
    
    # Remove street addresses (common Dutch patterns)
    anonymized = re.sub(r'\b[A-Z][a-z]+(?:straat|laan|weg|plein|gracht|kade|singel|dreef|hof|park)\s+\d+[a-zA-Z]?\b', '[ADDRESS]', anonymized, flags=re.IGNORECASE)
    
    # Remove full names but keep first name (remove "van/de/der + lastname" patterns)
    if first_name:
        anonymized = re.sub(r'\b' + first_name + r'\s+(?:van\s+(?:de|der|den)\s+)?[A-Z][a-z]+\b', first_name, anonymized)
    
    return anonymized, first_name

RESPONSE_PROMPT = """You are a customer service assistant for Dobbi, a Dutch dry cleaning company.

RULES:
- Be concise. Only answer what the customer asked.
- Do NOT mention services or information the customer didn't ask about.
- Do NOT ask follow-up questions unless absolutely necessary.
- Use prices from the knowledge base. If a price is not available, say so briefly.
- Match the customer's language (Dutch or English).
- Keep responses short: 3-6 sentences max for simple questions.
- If you see [EMAIL], [PHONE], [ADDRESS], or [POSTCODE], ignore them - they are placeholders for privacy.
- Address the customer by their first name if provided: {first_name}
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
        # Anonymize before sending to API, but keep first name
        anonymized_message, first_name = anonymize_message(customer_message)
        
        context = "\n\n".join([
            f"[Source: {doc['metadata']['source']}]\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        if not context:
            context = "No specific information found in knowledge base."
        
        first_name_instruction = first_name if first_name else "Not provided - use a general greeting"
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": RESPONSE_PROMPT.format(
                    customer_message=anonymized_message,
                    category=category,
                    retrieved_context=context,
                    first_name=first_name_instruction
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
    # Test anonymization
    test = "Hi, I'm Sophie van der Berg, email sophie@gmail.com, phone 0612345678, Hoofdstraat 42, 1234 AB Amsterdam"
    anon, name = anonymize_message(test)
    print("Original:", test)
    print("Anonymized:", anon)
    print("First name kept:", name)
