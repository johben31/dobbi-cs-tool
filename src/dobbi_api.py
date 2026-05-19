import os
import requests
from dotenv import load_dotenv

load_dotenv()

DOBBI_API_BASE = "https://midlayer.dobbi.com/api-cs"
DOBBI_API_KEY = os.getenv("DOBBI_API_KEY")


def get_order_status(barcode: str) -> dict | None:
    """
    Get order track and trace info from Dobbi API.
    
    Args:
        barcode: Order ID (e.g., '393143') or bag barcode (e.g., '22836100004737')
    
    Returns:
        Order info dict or None if not found
    """
    if not DOBBI_API_KEY:
        print("Warning: DOBBI_API_KEY not set")
        return None
    
    try:
        response = requests.get(
            f"{DOBBI_API_BASE}/orders/track-and-trace",
            params={"barcode": barcode},
            headers={
                "Content-Type": "application/json",
                "dobbi-api-key": DOBBI_API_KEY,
                "Accept-Language": "nl-NL"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def format_order_status(order_data: dict) -> str:
    """
    Format order data into a readable string for the response.
    """
    if not order_data:
        return "Order niet gevonden."
    
    status = order_data.get("status", {}).get("name", "Onbekend")
    
    pickup = order_data.get("pickup", {})
    pickup_date = pickup.get("time", {}).get("formatted", "Onbekend")
    pickup_window = pickup.get("windowFormatted", "")
    
    delivery = order_data.get("delivery", {})
    delivery_date = delivery.get("time", {}).get("formatted", "Onbekend")
    delivery_window = delivery.get("windowFormatted", "")
    
    result = f"Status: {status}\n"
    result += f"Ophalen: {pickup_date}"
    if pickup_window:
        result += f" ({pickup_window})"
    result += f"\nBezorging: {delivery_date}"
    if delivery_window:
        result += f" ({delivery_window})"
    
    return result

def get_order_details(order_data: dict) -> dict | None:
    """
    Extract internal details for CS agents from order data.
    """
    if not order_data:
        return None
    
    carrier = order_data.get("type", "Unknown")
    
    # Get shipment info from pickup or delivery
    shipment = None
    packages = []
    
    pickup = order_data.get("pickup", {})
    delivery = order_data.get("delivery", {})
    
    if pickup.get("shipment"):
        shipment = pickup["shipment"]
    if delivery.get("shipment"):
        shipment = delivery["shipment"]
    
    shipment_id = shipment.get("id") if shipment else None
    
    if shipment and shipment.get("barcodes"):
        for barcode in shipment["barcodes"]:
            packages.append({
                "barcode": barcode.get("barcode"),
                "status": barcode.get("status") or "Pending"
            })
    
    return {
        "carrier": carrier.title(),
        "shipment_id": shipment_id,
        "package_count": len(packages),
        "packages": packages
    }

def extract_order_number(text: str) -> str | None:
    """
    Extract order number or barcode from customer message.
    Looks for patterns like #123456, order 123456, barcode 22836100004737
    """
    import re
    
    # Pattern for bag barcode (long number, possibly with dash)
    bag_pattern = r'\b(2283\d{10,}(?:-\d+)?)\b'
    bag_match = re.search(bag_pattern, text)
    if bag_match:
        return bag_match.group(1)
    
    # Pattern for order ID with # prefix
    hash_pattern = r'#(\d{5,7})\b'
    hash_match = re.search(hash_pattern, text)
    if hash_match:
        return hash_match.group(1)
    
    # Pattern for order/bestelling followed by number
    order_pattern = r'(?:order|bestelling|ordernummer)\s*[:#]?\s*(\d{5,7})\b'
    order_match = re.search(order_pattern, text, re.IGNORECASE)
    if order_match:
        return order_match.group(1)
    
    # Pattern for standalone 6-digit number (likely order ID)
    standalone_pattern = r'\b(\d{6})\b'
    standalone_match = re.search(standalone_pattern, text)
    if standalone_match:
        return standalone_match.group(1)
    
    return None


if __name__ == "__main__":
    # Test the API
    test_barcode = "393143"
    print(f"Testing with barcode: {test_barcode}")
    
    result = get_order_status(test_barcode)
    if result:
        print("\nRaw response:")
        print(result)
        print("\nFormatted:")
        print(format_order_status(result))
    else:
        print("No result")
    
    # Test extraction
    test_messages = [
        "Waar is mijn order #393143?",
        "Bestelling 393143 status",
        "Barcode 22836100004737",
        "Ik heb niks ontvangen, order nummer: 395163"
    ]
    print("\n\nTesting order extraction:")
    for msg in test_messages:
        extracted = extract_order_number(msg)
        print(f"  '{msg}' -> {extracted}")
