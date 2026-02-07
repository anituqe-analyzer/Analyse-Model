import os
import requests
from apify_client import ApifyClient
from dotenv import load_dotenv
import json

# --- CONFIGURATION ---
# Load environment variables from the .env file (if it exists)
load_dotenv()

ACTOR_ID = "e-commerce/allegro-product-detail-scraper"

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

def get_high_res_image(url):
    """Converts a thumbnail/resized link to the original high-resolution Allegro link."""
    if not url: return None
    sizes = ["/s128/", "/s360/", "/s720/", "/s1024/", "/s1440/"]
    for size in sizes:
        if size in url:
            return url.replace(size, "/original/")
    return url

def get_api_token():
    """
    Retrieves API token.
    Priority 1: from .env file (environment variable).
    Priority 2: prompts user input in the console.
    """
    token = os.getenv("APIFY_TOKEN")
    
    if token:
        print("Info: API Token loaded from .env file.")
        return token
    
    return AttributeError("API Token is required but not provided.")

def get_allegro_data(url):
    apify_token = get_api_token()
    
    if not apify_token:
        print("ERROR: API Token is required to run the script.")
        return

    client = ApifyClient(apify_token)
    
    run_input = { "startUrls": [url] }

    try:
        print("--- GATHERING DATA ---")
        
        run = client.actor(ACTOR_ID).call(run_input=run_input)
        
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        if not dataset_items:
            print("Apify finished the job but returned no data.")
            return

        item = dataset_items[0]

        # --- DATA MAPPING ---
        
        # TITLE
        title = item.get("productTitle") or item.get("title") or "untitled"

        # DESCRIPTION
        description = item.get("description", "No description")

        # PARAMETERS
        parameter_list = []
        specs = item.get("productSpecifications", {})
        
        if isinstance(specs, dict):
            for key, value in specs.items():
                parameter_list.append(f"{key}: {value}")
        elif not specs:
            raw_params = item.get("parameters") or item.get("attributes", [])
            for p in raw_params:
                name = p.get("name") or p.get("key")
                val = p.get("value")
                if name and val:
                    parameter_list.append(f"{name}: {val}")

        # IMAGES
        unique_links = set()
        
        raw_images = item.get("images", [])
        for img in raw_images:
            if isinstance(img, str): unique_links.add(get_high_res_image(img))
            elif isinstance(img, dict): unique_links.add(get_high_res_image(img.get("url")))

        if not unique_links:
            thumb = item.get("thumbnail")
            if thumb:
                high_res = get_high_res_image(thumb)
                unique_links.add(high_res)
                print("Info: Retrieved main image from thumbnail (gallery was empty in API).")

        print(f"Found {len(unique_links)} images.")

        return {
                    "title": title,
                    "sanitized_title": sanitize_name(title),
                    "platform": "Allegro",
                    "url": url,
                    "description": description,
                    "parameters": parameter_list,
                    "image_urls": list(unique_links),
                    "image_count": len(unique_links),
                    "price": f"{item.get('price')} {item.get('currency')}"
                }

    except Exception as e:
        print(f"Main error occurred: {e}")

