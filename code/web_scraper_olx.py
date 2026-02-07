import requests
from bs4 import BeautifulSoup
import json


# --- FUNCTIONS ---
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


def get_olx_data(url):
    """Fetches OLX offer data and returns it as a dictionary with title, description, parameters, and image URLs."""
    # --- CONFIGURATION ---
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"error": f"Connection error. Status code: {response.status_code}"}

    soup = BeautifulSoup(response.content, "html.parser")

    # --- GATHERING DATA ---

    # Title
    title_element = soup.find("h4", class_="css-1au435n")
    title = title_element.get_text().strip() if title_element else "untitled"

    # Price
    price = "No price"
    price_container = soup.find("div", attrs={"data-testid": "ad-price-container"})
    if price_container:
        price_header = price_container.find("h3")
        if price_header:
            price = price_header.get_text(strip=True)

    # Description
    description_element = soup.find("div", class_="css-19duwlz")
    description = (
        description_element.get_text(separator="\n").strip()
        if description_element
        else "No description"
    )

    # Parameters
    parameter_list = []
    parameters_container = soup.find(
        "div", attrs={"data-testid": "ad-parameters-container"}
    )
    if parameters_container:
        params = parameters_container.find_all("p", class_="css-13x8d99")
        for p in params:
            parameter_list.append(p.get_text().strip())

    # Image Links
    images = soup.select('img[data-testid^="swiper-image"]')
    unique_links = list(set(img.get("src") for img in images if img.get("src")))

    # --- RETURNING DICTIONARY ---
    return {
        "platform": "OLX",
        "title": title,
        "sanitized_title": sanitize_name(title),
        "price": price,
        "url": url,
        "description": description,
        "parameters": parameter_list,
        "image_urls": unique_links,
        "image_count": len(unique_links),
    }


