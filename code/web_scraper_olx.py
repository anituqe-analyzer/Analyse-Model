# scrape_olx_offer.py
import requests
from bs4 import BeautifulSoup

def scrape_olx_offer(url: str):
    """Zwraca dane aukcji bez zapisywania na dysk"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    print(f"🔍 OLX: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise ValueError(f"OLX error: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # TITLE
    title_element = soup.find("h4", class_="css-1au435n")
    title = title_element.get_text().strip() if title_element else "untitled"
    
    # DESCRIPTION
    description_element = soup.find("div", class_="css-19duwlz")
    description = description_element.get_text(separator="\n").strip() if description_element else "No description"
    
    # PARAMETERS
    parameter_list = []
    parameters_container = soup.find("div", attrs={"data-testid": "ad-parameters-container"})
    if parameters_container:
        params = parameters_container.find_all("p", class_="css-13x8d99")
        for p in params:
            parameter_list.append(p.get_text().strip())
    
    # IMAGES
    images = soup.select('img[data-testid^="swiper-image"]')
    unique_links = set()
    for img in images:
        link = img.get("src")
        if link:
            unique_links.add(link)
    
    return {
        "platform": "olx",
        "url": url,
        "title": title,
        "description": description,
        "parameters": parameter_list,
        "image_urls": list(unique_links)
    }

if __name__ == "__main__":
    url = input("OLX URL: ")
    result = scrape_olx_offer(url)
    print(result)
