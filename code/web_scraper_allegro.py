# scrape_allegro_offer.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import requests

def sanitize_folder_name(text):  # helper function
    polish_chars = {
        "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
        "ó": "o", "ś": "s", "ź": "z", "ż": "z"
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
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")

def scrape_allegro_offer(url: str):
    """Zwraca dane aukcji bez zapisywania na dysk"""
    options = uc.ChromeOptions()
    options.add_argument("--window-position=-3000,0")
    driver = uc.Chrome(use_subprocess=True, options=options)
    
    try:
        print(f"🔍 Allegro: {url}")
        driver.get(url)
        time.sleep(10)
        
        # TITLE
        try:
            title_element = driver.find_element(By.TAG_NAME, "h1")
            title_str = title_element.text.strip()
        except:
            title_str = "untitled"
        
        # PARAMETERS
        parameter_list = []
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) == 2:
                    name = cells[0].text.strip()
                    value = cells[1].text.strip()
                    if name and value:
                        parameter_list.append(f"{name}: {value}")
        except:
            pass
        
        # DESCRIPTION
        try:
            description_element = driver.find_element(By.CSS_SELECTOR, "div._0d3bd_am0a-")
            description_content = description_element.text
        except:
            description_content = "No description"
        
        # IMAGES
        unique_links = set()
        try:
            images = driver.find_elements(By.CSS_SELECTOR, ".msub_80.m9tr_5r._07951_IOf8s")
            allowed_sizes = ["/s128/", "/s360/", "/s512/", "/s720/", "/s1024/", "/s1440/", "/original/"]
            for img in images:
                src = img.get_attribute("src")
                if src and "allegroimg.com" in src:
                    if not any(size in src for size in allowed_sizes):
                        continue
                    for size in allowed_sizes:
                        src = src.replace(size, "/original/")
                    unique_links.add(src)
        except Exception as e:
            print(f"Image error: {e}")
        
        return {
            "platform": "allegro",
            "url": url,
            "title": title_str,
            "description": description_content,
            "parameters": parameter_list,
            "image_urls": list(unique_links)
        }
    
    finally:
        driver.quit()

if __name__ == "__main__":
    url = input("Allegro URL: ")
    result = scrape_allegro_offer(url)
    print(result)
