# app.py - Main entry point for Hugging Face Spaces
import os
import sys
import importlib.util

# Load the FastAPI app from code/app.py via importlib to avoid
# conflicts with the standard-library module named `code`.
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.join(HERE, "code")
app_path = os.path.join(CODE_DIR, "app.py")

# Ensure the `code/` directory is on sys.path so relative imports like
# `from model import ...` inside `code/app.py` resolve correctly.
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

spec = importlib.util.spec_from_file_location("antique_auth_code_app", app_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Optionally: remove CODE_DIR from sys.path after loading to avoid side effects
try:
    # remove the first occurrence we added
    if sys.path[0] == CODE_DIR:
        sys.path.pop(0)
except Exception:
    pass

# The FastAPI `app` object expected inside code/app.py
app = getattr(module, "app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
