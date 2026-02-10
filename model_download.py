from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "./sentence-transformers_all-MiniLM-L6-v2"

# Download & Save SentenceTransformer locally
# ==============================
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.save(MODEL_PATH)
    print("✅ Model downloaded & saved")
else:
    print("✅ Model already exists. Skipping download.")
