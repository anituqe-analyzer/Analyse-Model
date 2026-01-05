# scrape_ebay_offer.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import requests

def scrape_ebay_offer(url: str):
    """Zwraca dane aukcji bez zapisywania na dysk"""
    print(f"🔍 eBay: {url}")
    options = uc.ChromeOptions()
    options.add_argument("--window-position=-3000,0")
    driver = uc.Chrome(use_subprocess=True, options=options)
    
    try:
        driver.get(url)
        time.sleep(4)
        
        # TITLE
        try:
            title_element = driver.find_element(By.CSS_SELECTOR, "h1.x-item-title__mainTitle")
            title_str = title_element.text.strip()
        except:
            title_str = "untitled_ebay"
        
        # PARAMETERS
        parameter_list = []
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, ".ux-labels-values")
            for row in rows:
                try:
                    label = row.find_element(By.CSS_SELECTOR, ".ux-labels-values__labels").text.strip()
                    value = row.find_element(By.CSS_SELECTOR, ".ux-labels-values__values").text.strip()
                    if label and value:
                        parameter_list.append(f"{label}: {value}")
                except:
                    continue
        except:
            pass
        
        # DESCRIPTION
        description_content = "No description"
        try:
            frame = driver.find_element(By.ID, "desc_ifr")
            driver.switch_to.frame(frame)
            description_content = driver.find_element(By.TAG_NAME, "body").text.strip()
            driver.switch_to.default_content()
        except:
            pass
        
        # IMAGES
        unique_links = set()
        try:
            thumbnails = driver.find_elements(By.CSS_SELECTOR, ".ux-image-grid-item img")
            for img in thumbnails:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                if src and "ebayimg.com" in src:
                    # Zamień na HD
                    hd_link = src.replace("/s-l64/", "/s-l1600").replace("/s-l140/", "/s-l1600")
                    unique_links.add(hd_link)
        except:
            pass
        
        return {
            "platform": "ebay",
            "url": url,
            "title": title_str,
            "description": description_content,
            "parameters": parameter_list,
            "image_urls": list(unique_links)
        }
    
    finally:
        driver.quit()

if __name__ == "__main__":
    url = input("eBay URL: ")
    result = scrape_ebay_offer(url)
    print(result)
