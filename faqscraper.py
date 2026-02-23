import re
import json
from bs4 import BeautifulSoup
import requests


def scrape_dobbi_faq(url, language="en"):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    category_map = {
        "How does the service work": "services",
        "What types of services": "services",
        "don't know what service": "services",
        "lost my laundry bag": "services",
        "didn't receive a laundry bag": "services",
        "Can I use my own bag": "services",
        "Can I keep the laundry bag": "services",
        "can't place an order": "services",
        "placed an order by mistake": "services",
        "quickly can you process": "services",
        "change my pickup or delivery": "pickup_delivery",
        "miss my pickup or delivery": "pickup_delivery",
        "options for pickup and delivery": "pickup_delivery",
        "Who picks up and delivers": "pickup_delivery",
        "specify what items": "cleaning",
        "sort my clothes": "cleaning",
        "order doesn't fit in one": "cleaning",
        "starter kit": "starter_kit",
        "receive my dobbi starter kit": "starter_kit",
        "return the dobbi laundry bag": "starter_kit",
        "lose my dobbi laundry bag": "starter_kit",
        "How much fits": "starter_kit",
        "How can I pay": "payment",
        "minimum order value": "payment",
        "order is under": "payment",
        "pay by invoice": "payment",
        "pay via direct debit": "payment",
        "When do I need to pay": "payment",
    }

    def get_category(question_text):
        for keyword, cat in category_map.items():
            if keyword.lower() in question_text.lower():
                return cat
        return "general"

    faq_data = []
    counter = 1

    for h2 in soup.find_all("h2"):
        question = h2.get_text(strip=True)
        if not question:
            continue

        # Get the parent faq-toggle div
        faq_toggle_div = h2.parent

        # Find the next sibling that contains the answer (should be a div with text)
        answer_div = faq_toggle_div.find_next_sibling()
        answer = ""

        if answer_div:
            answer = answer_div.get_text(separator=" ", strip=True)
            # Clean up any extra whitespace
            answer = re.sub(r'\s+', ' ', answer).strip()

        if question and answer:
            faq_data.append({
                "id": f"faq_{language}_{str(counter).zfill(3)}",
                "question": question,
                "answer": answer,
                "category": get_category(question),
                "language": language
            })
            counter += 1

    return faq_data


faq = scrape_dobbi_faq("https://dobbi.com/nl/faq", language="nl")

with open("faq.nl.json", "w", encoding="utf-8") as f:
    json.dump(faq, f, indent=2, ensure_ascii=False)

print(f"Successfully exported {len(faq)} FAQ items to faq.nl.json")
print(json.dumps(faq[:2], indent=2))
