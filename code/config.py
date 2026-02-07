# config.py
"""
Configuration for authenticity classes and categories
This allows elastic modification without code changes
"""

# Authenticity Status (main model - requires retraining if changed)
AUTHENTICITY_CLASSES = {
    0: "ORIGINAL",
    1: "SCAM",
    2: "REPLICA"
}

AUTHENTICITY_TO_ID = {v: k for k, v in AUTHENTICITY_CLASSES.items()}

# Category/Type (separate classifier - can be extended without retraining main model)
# 5 specific categories + UNCERTAIN added automatically when confidence is low
# Add/remove categories here as needed (edit these 5, UNCERTAIN is automatic)
CATEGORIES = {
    0: "Clocks",
    1: "Furniture",
    2: "Numismatics",
    3: "Sabers",
    4: "Tableware"
}

CATEGORY_TO_ID = {v: k for k, v in CATEGORIES.items()}

# Special uncertainty category (added automatically, not in model output)
UNCERTAIN_CATEGORY = "Uncertain"

# Uncertainty thresholds
UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.6
UNCERTAINTY_MARGIN_THRESHOLD = 0.15
