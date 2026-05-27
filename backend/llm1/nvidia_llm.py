# llm1/nvidia_llm.py

import json
import urllib.error
import urllib.request

from llm1.llm_config import (
    MAX_TOKENS,
    NVIDIA_API_BASE,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    NVIDIA_TIMEOUT,
    TEMPERATURE,
)


class NvidiaChatLLM:
    """Small OpenAI-compatible client for NVIDIA hosted NIM chat completions."""

    def __init__(self, json_mode: bool = False):
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured")

        self.json_mode = json_mode
        self.url = f"{NVIDIA_API_BASE.rstrip('/')}/chat/completions"

    def invoke(self, prompt: str) -> str:
        system_prompt = (
            "You are a precise API response generator. Return only valid JSON."
            if self.json_mode
            else "You are a helpful communication coach. Return polished, actionable feedback."
        )
        payload = {
            "model": NVIDIA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=NVIDIA_TIMEOUT) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NVIDIA API error {e.code}: {error_body}") from e

        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]
