# llm1/llm_config.py

import os


LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
