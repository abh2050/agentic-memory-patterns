import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = "gpt-4o"
CONFIDENCE_THRESHOLD = 0.7
MAX_FACTS = 100
TOKEN_BUDGET = 2000
DEBOUNCE_SECONDS = 3        # set to 30 in production
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MEMORY_FILE = "memory.json"
MEMORY_TMP_FILE = "memory.json.tmp"
