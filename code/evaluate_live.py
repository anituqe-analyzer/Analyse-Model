# evaluate_live.py
import requests
from io import BytesIO

# Import Twoich scraperów
from web_scraper_allegro import scrape_allegro_offer
from web_scraper_olx import scrape_olx_offer
from web_scraper_ebay import scrape_ebay_offer

API_URL = "http://localhost:7860/predict"

def call_model(auction):
    if not auction.get("image_urls"):
        return {"error": "No images found"}
    
    img_url = auction["image_urls"][0]
    print(f"📸 Pobieram zdjęcie: {img_url}")
    
    img_resp = requests.get(img_url, timeout=20)
    img_resp.raise_for_status()
    
    files = {
        "image": ("image.jpg", BytesIO(img_resp.content), "image/jpeg")
    }
    data = {
        "title": auction.get("title", ""),
        "description": auction.get("description", "")
    }
    
    r = requests.post(API_URL, files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

def scrape_offer(url: str):
    """Automatycznie wybiera scraper na podstawie domeny"""
    if "allegro.pl" in url:
        return scrape_allegro_offer(url)
    elif "olx.pl" in url:
        return scrape_olx_offer(url)
    elif "ebay." in url:
        return scrape_ebay_offer(url)
    else:
        raise ValueError("Nieobsługiwana platforma")

def evaluate_url(url: str):
    """Pełny pipeline: scrape → model → wynik"""
    print(f"🔍 Analizuję: {url}")
    
    auction = scrape_offer(url)
    print(f"📋 Zebrane: {auction['title'][:50]}...")
    
    model_result = call_model(auction)
    
    return {
        "url": url,
        "platform": auction["platform"],
        "title": auction["title"],
        "model_result": model_result,
    }

if __name__ == "__main__":
    while True:
        url = input("\nPodaj link do aukcji (lub 'q' do wyjścia): ")
        if url.lower() == 'q':
            break
        
        try:
            result = evaluate_url(url)
            print("\n" + "="*80)
            print(f"VERDICT: {result['model_result'].get('verdict')}")
            print(f"CONFIDENCE: {result['model_result'].get('confidence')}")
            print("="*80)
        except Exception as e:
            print(f"❌ Błąd: {e}")
