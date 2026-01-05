from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from model import AuctionAuthenticityModel
from torchvision import transforms
import os

app = FastAPI(
    title="Antique Auction Authenticity API",
    description="AI model do oceny autentyczności aukcji antyków",
    version="1.0.0"
)

# CORS - zezwól na żądania z dowolnych źródeł
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfiguracja
DEVICE = torch.device('cpu')
MODEL_PATH = '../weights/auction_model.pt'

# Załaduj model na startup
@app.on_event("startup")
def load_model():
    global model
    print("🚀 Ładowanie modelu...")
    
    model = AuctionAuthenticityModel(device=DEVICE).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Model załadowany z {MODEL_PATH}")
    else:
        print(f"⚠️  Model nie znaleziony! Używam wagi z pretrain.")
    
    model.eval()
    print("✓ Model gotowy do inference")

# Transformacje
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(...)
):
    """
    Ocena autentyczności aukcji
    
    Parametry:
    - image: Zdjęcie aukcji (JPG/PNG)
    - title: Tytuł aukcji
    - description: Opis aukcji
    
    Zwraca:
    {
        "authentic_probability": 0.87,
        "suspicious_probability": 0.13,
        "verdict": "LIKELY_AUTHENTIC",
        "confidence": 0.87
    }
    """
    try:
        # Wczytaj zdjęcie
        img_data = await image.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Przygotuj tekst
        text = f"{title} {description}"
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor, [text])
            probs = torch.softmax(logits, dim=1)[0]
        
        authentic_prob = float(probs[0])
        suspicious_prob = float(probs[1])
        confidence = max(authentic_prob, suspicious_prob)
        
        # Logika verdyktu
        if authentic_prob > 0.7:
            verdict = "LIKELY_AUTHENTIC"
        elif suspicious_prob > 0.6:
            verdict = "LIKELY_SUSPICIOUS"
        else:
            verdict = "UNCERTAIN"
        
        return JSONResponse({
            "status": "success",
            "authentic_probability": round(authentic_prob, 3),
            "suspicious_probability": round(suspicious_prob, 3),
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "message": f"Aukcja ma {confidence*100:.1f}% pewności, że jest {verdict.lower()}"
        })
    
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "error": str(e)
            },
            status_code=400
        )

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "ok",
        "message": "API is running"
    }

@app.get("/")
def root():
    """Info o API"""
    return {
        "name": "Antique Auction Authenticity API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Oceń autentyczność aukcji",
            "GET /health": "Health check",
            "GET /docs": "Dokumentacja Swagger"
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)