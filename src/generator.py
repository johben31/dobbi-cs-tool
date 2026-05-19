from anthropic import Anthropic
from dotenv import load_dotenv
from dobbi_api import get_order_status, format_order_status, extract_order_number

load_dotenv()

RESPONSE_PROMPT = """You are a friendly customer service assistant for Dobbi, a Dutch dry cleaning company.

RULES:
- Be warm, positive, and helpful.
- Be concise. Only answer what the customer asked.
- IMPORTANT: Match the customer's language exactly. Dutch question = Dutch response. English question = English response.
- Do NOT mention disclaimers, risks, or warnings unless the customer specifically asks about them.
- Do NOT mention terms and conditions unless the customer asks about policies.
- Do NOT say "contact customer service" or "neem contact op met onze klantenservice" - the customer is ALREADY talking to customer service. You are helping the CS employee respond to them.
- Do NOT ask follow-up questions unless absolutely necessary.
- Use prices from the knowledge base. If a price is not available, say so briefly.
- Keep responses short: 2-4 sentences max for simple questions.
- If order information is provided below, use it to answer the customer's question about their order.
- For pricing questions, just give the price and delivery time. Keep it simple and positive.
- Start with a brief greeting like "Hoi!" (Dutch) or "Hi!" (English).
- Sign off with "Groetjes, Team Dobbi" (Dutch) or "Best regards, Team Dobbi" (English).

CUSTOMER MESSAGE:
{customer_message}

CATEGORY: {category}

{order_info_section}

KNOWLEDGE BASE:
{retrieved_context}

Write a short, friendly response in the SAME LANGUAGE as the customer's message. Do not include warnings, disclaimers, or suggestions to contact customer service."""


class ResponseGenerator:
    def __init__(self):
        self.client = Anthropic()
    
    def generate(self, customer_message: str, category: str, retrieved_docs: list[dict]) -> dict:
        # Check if message contains an order number
        order_number = extract_order_number(customer_message)
        order_info_section = ""
        
        if order_number:
            order_data = get_order_status(order_number)
            if order_data:
                formatted_order = format_order_status(order_data)
                order_info_section = f"ORDER INFORMATION (for order {order_number}):\n{formatted_order}"
            else:
                order_info_section = f"ORDER INFORMATION: Order {order_number} not found in system."
        
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
                    order_info_section=order_info_section,
                    retrieved_context=context
                )
            }]
        )
        
        confidence = self._estimate_confidence(retrieved_docs)
        
        return {
            "draft_response": response.content[0].text,
            "sources_used": [doc['metadata']['source'] for doc in retrieved_docs],
            "confidence": confidence,
            "order_number": order_number
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
