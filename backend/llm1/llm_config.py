# llm1/llm_config.py

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from .env without requiring extra packages."""
    backend_dir = Path(__file__).resolve().parents[1]
    for env_path in (backend_dir / ".env", backend_dir.parent / ".env"):
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_dotenv()


LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_TIMEOUT = float(os.getenv("NVIDIA_TIMEOUT", "45"))
