from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from model import AuctionAuthenticityModel
from config import (
    AUTHENTICITY_CLASSES,
    CATEGORIES,
    UNCERTAINTY_CONFIDENCE_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAIN_CATEGORY,
)
from torchvision import transforms
import os
import numpy as np

app = FastAPI(
    title="Antique Auction Authenticity API",
    description="AI model for antique auction authenticity evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
MODEL_PATH = "../weights/auction_model.pt"

authenticity_model = None

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.on_event("startup")
async def load_model():
    global authenticity_model
    print("🚀 Loading model...")

    authenticity_model = AuctionAuthenticityModel(device=DEVICE).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        authenticity_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print("⚠️ No model weights found – using pretrained backbone")

    authenticity_model.eval()
    print("✓ Model ready")


def predict_single(img_tensor, text):
    with torch.no_grad():
        outputs = authenticity_model(img_tensor, [text])
        auth_probs = outputs["auth_probs"][0].cpu().numpy()
        cat_probs = outputs["cat_probs"][0].cpu().numpy()
    return auth_probs, cat_probs


def build_verdict(probs, labels):
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    best_label = max(probs_dict, key=probs_dict.get)
    best_prob = probs_dict[best_label]

    sorted_probs = sorted(probs_dict.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]

    uncertain = (
        best_prob < UNCERTAINTY_CONFIDENCE_THRESHOLD
        or margin < UNCERTAINTY_MARGIN_THRESHOLD
    )

    return probs_dict, best_label, best_prob, margin, uncertain



@app.post("/validate_url")
async def validate_url(url: str = Form(...), max_images: int = Form(3)):
    try:
        from io import BytesIO
        import requests

        max_images = max(1, min(max_images, 10))

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

        images_to_use = min(max_images, len(auction["image_urls"]))

        auth_probs_list = []
        cat_probs_list = []

        text = auction["title"] + " " + auction.get("description", "")

        for img_url in auction["image_urls"][:images_to_use]:
            img_resp = requests.get(img_url, timeout=15)
            img_resp.raise_for_status()

            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            auth_probs, cat_probs = predict_single(img_tensor, text)

            auth_probs_list.append(auth_probs)
            cat_probs_list.append(cat_probs)

        avg_auth_probs = np.mean(auth_probs_list, axis=0)
        avg_cat_probs = np.mean(cat_probs_list, axis=0)

        auth_dict, best_auth, best_auth_prob, auth_margin, auth_uncertain = build_verdict(
            avg_auth_probs, AUTHENTICITY_CLASSES
        )

        cat_dict, best_cat, best_cat_prob, cat_margin, cat_uncertain = build_verdict(
            avg_cat_probs, CATEGORIES
        )

        auth_verdict = "UNCERTAIN" if auth_uncertain else best_auth
        category_verdict = UNCERTAIN_CATEGORY if cat_uncertain else best_cat

        return JSONResponse(
            {
                "status": "success",
                "evaluation": {
                    "title": auction["title"],
                    "image_urls": auction["image_urls"][:images_to_use],
                    "price": auction["price"],
                    "category": None
                    if category_verdict == UNCERTAIN_CATEGORY
                    else category_verdict,
                    "evaluation_status": auth_verdict,
                    "confidence": round(best_auth_prob, 3),
                },
                "details": {
                    "url": url,
                    "platform": auction["platform"],
                    "image_count_used": images_to_use,
                    "authenticity": {
                        "verdict": auth_verdict,
                        "confidence": round(best_auth_prob, 3),
                        "margin": round(auth_margin, 3),
                        "probabilities": {
                            k: round(v, 3) for k, v in auth_dict.items()
                        },
                    },
                    "category": {
                        "verdict": category_verdict,
                        "label": best_cat,
                        "confidence": round(best_cat_prob, 3),
                        "margin": round(cat_margin, 3),
                        "probabilities": {
                            k: round(v, 3) for k, v in cat_dict.items()
                        },
                    },
                },
            }
        )

    except Exception as e:
        import traceback

        return JSONResponse(
            {"status": "error", "error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/health")
def health():
    return {"status": "ok", "message": "API running"}


@app.get("/")
def root():
    return {
        "name": "Antique Auction Authenticity API",
        "version": "1.0.0",
        "endpoints": {"POST /predict": "Evaluate auction", "GET /health": "Health check"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
