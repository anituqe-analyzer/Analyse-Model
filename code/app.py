# app.py
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from model import AuctionAuthenticityModel
from torchvision import transforms
import os
import numpy as np


app = FastAPI(
    title="Antique Auction Authenticity API",
    description="AI model do oceny autentyczności aukcji antyków",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cpu')
MODEL_PATH = '../weights/auction_model.pt'

model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.on_event("startup")
async def load_model():
    global model
    print("🚀 Ładowanie modelu...")
    model = AuctionAuthenticityModel(num_classes=3, device=DEVICE).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Model załadowany z {MODEL_PATH}")
    else:
        print("⚠️  Brak wag - pretrained")
    model.eval()
    print("✓ Model gotowy")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(...)
):
    try:
        img_data = await image.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        text = f"{title} {description}"
        
        with torch.no_grad():
            logits = model(img_tensor, [text])
            probs = torch.softmax(logits, dim=1)[0]
        
        orig_prob = float(probs[0])  # label 0
        scam_prob = float(probs[1])  # label 1
        repl_prob = float(probs[2])  # label 2
        
        probs_dict = {
            "ORIGINAL": orig_prob,
            "SCAM": scam_prob,
            "REPLICA": repl_prob
        }
        best_label = max(probs_dict, key=probs_dict.get)
        best_prob = probs_dict[best_label]
        
        # Niepewny: max prob < 0.6 LUB margin < 0.15
        sorted_probs = sorted(probs_dict.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        
        if best_prob < 0.6 or margin < 0.15:
            verdict = "UNCERTAIN"
        else:
            verdict = best_label
        
        return JSONResponse({
            "status": "success",
            "original_probability": round(orig_prob, 3),
            "scam_probability": round(scam_prob, 3),
            "replica_probability": round(repl_prob, 3),
            "verdict": verdict,
            "confidence": round(best_prob, 3),
            "margin": round(margin, 3),
            "message": f"Aukcja ma {best_prob*100:.1f}% pewności: {verdict}"
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=400
        )

@app.post("/predict_ensemble")
async def predict_ensemble(
    images: list[UploadFile] = File(...),  # wiele plików!
    title: str = Form(...),
    description: str = Form(...)
):
    predictions = []
    
    for i, img_file in enumerate(images):
        img_data = await img_file.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        text = f"{title} {description}"
        
        with torch.no_grad():
            logits = model(img_tensor, [text])
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predictions.append(probs)
    
    # Średnia z wszystkich zdjęć
    avg_probs = np.mean(predictions, axis=0)
    
    orig_prob = float(avg_probs[0])
    scam_prob = float(avg_probs[1])
    repl_prob = float(avg_probs[2])
    
    probs_dict = {"ORIGINAL": orig_prob, "SCAM": scam_prob, "REPLICA": repl_prob}
    best_label = max(probs_dict, key=probs_dict.get)
    best_prob = probs_dict[best_label]
    
    sorted_probs = sorted(probs_dict.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    
    if best_prob < 0.6 or margin < 0.15:
        verdict = "UNCERTAIN"
    else:
        verdict = best_label
    
    return JSONResponse({
        "status": "success",
        "image_count": len(images),
        "original_probability": round(orig_prob, 3),
        "scam_probability": round(scam_prob, 3),
        "replica_probability": round(repl_prob, 3),
        "verdict": verdict,
        "confidence": round(best_prob, 3),
        "margin": round(margin, 3),
        "per_image_probs": [p.tolist() for p in predictions]  # dla debug
    })

@app.post("/validate_url")
async def validate_url(
    url: str = Form(...),
    max_images: int = Form(3)
):
    try:
        import numpy as np
        from io import BytesIO
        import requests
        
        max_images = max(1, min(max_images, 10))
        
        # 1. Scraper
        if "allegro.pl" in url:
            from web_scraper_allegro import scrape_allegro_offer
            auction = scrape_allegro_offer(url)
        elif "olx.pl" in url:
            from web_scraper_olx import scrape_olx_offer
            auction = scrape_olx_offer(url)
        elif "ebay." in url:
            from web_scraper_ebay import scrape_ebay_offer
            auction = scrape_ebay_offer(url)
        else:
            return JSONResponse({"error": "Unsupported platform"}, status_code=400)
        
        if not auction.get("image_urls"):
            return JSONResponse({"error": "No images"}, status_code=400)
        
        # 2. Ile zdjęć
        total_available = len(auction["image_urls"])
        images_to_use = min(max_images, total_available)
        
        # 3. Model BEZ HTTP (bezpośrednio!)
        img_probs = []
        text = auction["title"] + " " + auction["description"]
        
        for i, img_url in enumerate(auction["image_urls"][:images_to_use]):
            print(f"📸 {i+1}/{images_to_use}")
            
            img_resp = requests.get(img_url, timeout=15)
            img_resp.raise_for_status()
            
            img = Image.open(BytesIO(img_resp.content)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(img_tensor, [text])
                probs = torch.softmax(logits, dim=1)[0]
                
            img_probs.append({
                "original_probability": float(probs[0]),
                "scam_probability": float(probs[1]),
                "replica_probability": float(probs[2])
            })
        
        # 4. Średnia
        avg_orig = np.mean([p["original_probability"] for p in img_probs])
        avg_scam = np.mean([p["scam_probability"] for p in img_probs])
        avg_repl = np.mean([p["replica_probability"] for p in img_probs])
        
        probs_dict = {"ORIGINAL": avg_orig, "SCAM": avg_scam, "REPLICA": avg_repl}
        best_label = max(probs_dict, key=probs_dict.get)
        best_prob = float(probs_dict[best_label])
        
        sorted_probs = sorted(probs_dict.values(), reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1])
        
        if best_prob < 0.6 or margin < 0.15:
            verdict = "UNCERTAIN"
        else:
            verdict = best_label
        
        return {
            "status": "success",
            "url": url,
            "title": auction["title"][:100] + "...",
            "platform": auction["platform"],
            "total_images_available": total_available,
            "requested_max_images": max_images,
            "image_count_used": images_to_use,
            "original_probability": round(avg_orig, 3),
            "scam_probability": round(avg_scam, 3),
            "replica_probability": round(avg_repl, 3),
            "verdict": verdict,
            "confidence": round(best_prob, 3),
            "margin": round(margin, 3)
        }
    
    except Exception as e:
        import traceback
        return JSONResponse({
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok", "message": "API running"}

@app.get("/")
def root():
    return {
        "name": "Antique Auction Authenticity API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Oceń aukcję",
            "GET /health": "Health check"
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
