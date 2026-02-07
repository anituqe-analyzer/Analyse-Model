import os
import requests
from apify_client import ApifyClient
from dotenv import load_dotenv
import json

# --- CONFIGURATION ---
# Load environment variables from the .env file (if it exists)
load_dotenv()

ACTOR_ID = "vulnv/ebay-product-scraper"

# --- HELPER FUNCTIONS ---
def sanitize_name(text):
    """Sanitizes text by removing Polish characters and special symbols for a folder name."""
    polish_chars = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
    }

    text = text.lower()
    result = ""

    for char in text:
        if char in polish_chars:
            result += polish_chars[char]
        elif char.isalnum():
            result += char
        else:
            result += "_"

    # Remove double underscores
    while "__" in result:
        result = result.replace("__", "_")

    return result.strip("_")

def get_high_res_ebay_image(url):
    """Replaces the size code in the eBay link with s-l1600 (Max quality)."""
    if not url: return None
    sizes = ["s-l64", "s-l140", "s-l300", "s-l400", "s-l500", "s-l960"]
    
    for size in sizes:
        if size in url:
            return url.replace(size, "s-l1600")
            
    if "ebayimg.com" in url and "s-l1600" not in url:
        parts = url.split("/")
        last_part = parts[-1] 
        if "s-l" in last_part:
            return url.replace(last_part[:last_part.find(".")], "s-l1600")
            
    return url

def get_api_token():
    """Retrieves token from .env or asks the user."""
    token = os.getenv("APIFY_TOKEN")
    if token:
        print("Info: API Token loaded from .env file.")
        return token
    
    return AttributeError("API Token is required but not provided.")


def get_ebay_data(url):
    apify_token = get_api_token()
    if not apify_token:
        print("ERROR: API Token is required.")
        return

    print(f"\n--- SENDING REQUEST TO APIFY ---")
    client = ApifyClient(apify_token)
    
    run_input = { "product_urls": [url] }

    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
        
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        if not dataset_items:
            print("Apify finished the job but returned no data.")
            return

        item = dataset_items[0]

        # --- DATA MAPPING ---
        
        # TITLE
        title = item.get("name") or item.get("title") or "untitled_ebay"

        # PRICE
        price = item.get("price", "N/A")
        currency = item.get("currency", "")
        
        # DESCRIPTION
        description = item.get("description", "No text description available.")

        # PARAMETERS
        parameter_list = []
        raw_props = item.get("additionalProperties", [])
        if isinstance(raw_props, list):
            for prop in raw_props:
                p_name = prop.get("name")
                p_val = prop.get("value")
                if p_name and p_val:
                    parameter_list.append(f"{p_name}: {p_val}")
        
        if item.get("sku"): parameter_list.insert(0, f"SKU: {item.get('sku')}")

        # IMAGES
        unique_links = set()
        
        main_img = item.get("mainImage", {}).get("url")
        if main_img:
            unique_links.add(get_high_res_ebay_image(main_img))
            
        raw_images = item.get("images", [])
        for img_entry in raw_images:
            if isinstance(img_entry, dict):
                raw_url = img_entry.get("url")
                if raw_url:
                    unique_links.add(get_high_res_ebay_image(raw_url))
            elif isinstance(img_entry, str):
                unique_links.add(get_high_res_ebay_image(img_entry))

        print(f"Found {len(unique_links)} unique images (High-Res).")

        return {
            "platform": "Ebay",
            "title": title,
            "sanitized_title": sanitize_name(title),
            "url": url,
            "description": description,
            "parameters": parameter_list,
            "image_urls": list(unique_links),
            "image_count": len(unique_links),
            "price": f"{price} {currency}"
        }

    except Exception as e:
        print(f"Critical error occurred: {e}")

