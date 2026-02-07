# Antique Auth API

API do oceny autentyczności antyków z portali aukcyjnych. Model wykorzystuje multimodalne uczenie maszynowe (obraz + tekst) do klasyfikacji ofert jako oryginalne, repliki lub oszustwa.

## Funkcjonalności

- **Weryfikacja URL** – analiza ofert z Allegro, OLX i eBay na podstawie linku
- **Klasyfikacja autentyczności** – ORIGINAL, SCAM, REPLICA
- **Klasyfikacja kategorii** – np. zegary, meble, numizmatyka, szable, zastawa stołowa
- **Detekcja niepewności** – zwraca `UNCERTAIN` gdy model ma niskie zaufanie do predykcji

## Architektura modelu

- **Vision**: EfficientNet-B0 (ekstrakcja cech z obrazu)
- **Text**: DistilBERT (język wielojęzyczny, obsługa polskiego)
- **Fuzja**: połączone cechy → wspólny encoder → dwie głowy (autentyczność + kategoria)

## Wymagania

- Python 3.10+
- CUDA (opcjonalnie, dla szybszego treningu)

## Instalacja

```bash
# Klonowanie repozytorium
git clone <repo_url>
cd antique-auth-api

# Tworzenie środowiska wirtualnego (zalecane)
python -m venv venv
venv\Scripts\activate   # Windows

# Instalacja zależności
pip install -r requirements.txt
```

## Uruchomienie API

```bash
# Z głównego katalogu projektu
python app.py
```

API startuje na `http://localhost:7860`.

## Endpointy

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| GET | `/` | Informacje o API |
| GET | `/health` | Health check |
| POST | `/validate_url` | Walidacja oferty po URL |

### Walidacja URL (`POST /validate_url`)

**Parametry (form-data):**
- `url` (string) – link do oferty na Allegro, OLX lub eBay
- `max_images` (int, opcjonalnie) – max. liczba zdjęć do analizy (1–10, domyślnie 3)

**Przykład (curl):**
```bash
curl -X POST "http://localhost:7860/validate_url" \
  -F "url=https://allegro.pl/oferta/..." \
  -F "max_images=5"
```

**Przykładowa odpowiedź:**
```json
{
  "status": "success",
  "evaluation": {
    "title": "Antyczna srebrna łyżka XIX w.",
    "evaluation_status": "ORIGINAL",
    "confidence": 0.87,
    "category": "Tableware"
  }
}
```

## Struktura projektu

```
antique-auth-api/
├── app.py              # Entry point (Hugging Face Spaces / uvicorn)
├── code/
│   ├── app.py          # FastAPI – główna logika API
│   ├── config.py       # Konfiguracja (klasy, kategorie, progi)
│   ├── model.py        # AuctionAuthenticityModel
│   ├── train.py        # Skrypt treningu
│   ├── dataset_loader.py
│   ├── web_scraper_allegro.py
│   ├── web_scraper_olx.py
│   ├── web_scraper_ebay.py
│   └── labeling_app/   # Narzędzie do etykietowania danych
├── weights/
│   ├── auction_model.pt      # Wagi modelu
│   └── training_history.json
├── requirements.txt
└── Dockerfile
```

## Trening modelu

1. Przygotuj dataset w formacie JSON (`dataset/dataset.json`) z polami m.in.: `title`, `description`, `images`, `folder_path`, `label`, `platform`.

2. Uruchom trening:
```bash
cd code
python train.py
```

Model zostanie zapisany w `weights/auction_model.pt`.

## Etykietowanie danych

Aplikacja Flask do ręcznego etykietowania aukcji (autentyczność + kategoria):

```bash
cd code/labeling_app
python labeling_app.py
```

## Konfiguracja

Plik `code/config.py` pozwala m.in.:
- Zmieniać klasy autentyczności (ORIGINAL, SCAM, REPLICA)
- Rozszerzać kategorie (Clocks, Furniture, Numismatics, Sabers, Tableware)
- Ustawiać progi niepewności (`UNCERTAINTY_CONFIDENCE_THRESHOLD`, `UNCERTAINTY_MARGIN_THRESHOLD`)

## Docker

```bash
docker build -t antique-auth-api .
docker run -p 7860:7860 antique-auth-api
```

## Licencja

Projekt tworzony na potrzeby zajęć zespołowych.
