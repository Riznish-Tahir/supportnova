"""Central config — env-driven, no secrets committed."""
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GENAI_MODE = os.environ.get("SUPPORTNOVA_GENAI_MODE", "mock")  # "mock" | "live"
DATABASE_PATH = os.environ.get("SUPPORTNOVA_DB_PATH", "database/supportnova.db")
MAX_FILE_SIZE_MB = int(os.environ.get("SUPPORTNOVA_MAX_FILE_MB", "20"))
NEAR_DUPLICATE_THRESHOLD = float(os.environ.get("SUPPORTNOVA_DUP_THRESHOLD", "0.85"))
