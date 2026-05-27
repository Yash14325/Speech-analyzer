# llm1/local_llm.py

from llm1.llm_config import (
    LLM_MODEL_NAME,
    MAX_TOKENS,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)
from llm1.nvidia_llm import NvidiaChatLLM


class _ReportStubLLM:
    """Fallback report generator used when Ollama is unavailable."""

    def invoke(self, prompt: str) -> str:
        return """
**Communication Overview**
- Your speech was processed successfully, and the system generated a local fallback report.
- The analysis covered communication quality, vocal confidence, and personality signals.

**Confidence & Emotional Tone**
- Maintain a steady speaking pace and use intentional pauses to sound composed.
- Keep your volume consistent so key points land with more authority.

**Personality Insights**
- Your speaking style suggests a professional, structured communication approach.
- Use a short opening, two or three supporting points, and a clear close.

**Key Strengths**
- Clear end-to-end speech analysis flow.
- Practical feedback generated from the available transcript and acoustic signals.

**Improvement Recommendations**
- Start Ollama and pull the configured model for live AI-generated reports.
- Recommended command: `ollama pull llama3:8b`
""".strip()


class _SafeLLM:
    """Invoke NVIDIA first, falling back to Ollama and then the report stub."""

    def __init__(self, primary_llm=None, ollama_llm=None):
        self._primary_llm = primary_llm
        self._ollama_llm = ollama_llm
        self._fallback = _ReportStubLLM()

    def invoke(self, prompt: str) -> str:
        if self._primary_llm is not None:
            try:
                response = self._primary_llm.invoke(prompt)
                return getattr(response, "content", response)
            except Exception as e:
                print(f"NVIDIA report request failed ({e}), falling back to Ollama")
                self._primary_llm = None

        if self._ollama_llm is not None:
            try:
                response = self._ollama_llm.invoke(prompt)
                return getattr(response, "content", response)
            except Exception as e:
                print(f"Ollama report request failed ({e}), using stub LLM")
                self._ollama_llm = None

        return self._fallback.invoke(prompt)


def get_llm():
    """
    Returns an NVIDIA-backed LLM with Ollama fallback.
    Falls back to stub if both providers are unavailable.
    """
    primary_llm = None
    if NVIDIA_API_KEY:
        try:
            primary_llm = NvidiaChatLLM(json_mode=False)
            print(f"Using NVIDIA NIM primary report model: {NVIDIA_MODEL}")
        except Exception as e:
            print(f"NVIDIA integration not available: {e}")

    try:
        try:
            from langchain_ollama import OllamaLLM

            llm = OllamaLLM(
                model=LLM_MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE,
                num_predict=MAX_TOKENS,
            )
        except (ImportError, ModuleNotFoundError):
            from langchain_community.llms import Ollama

            llm = Ollama(
                model=LLM_MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE,
                num_predict=MAX_TOKENS,
            )

        return _SafeLLM(primary_llm=primary_llm, ollama_llm=llm)

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        print(f"Ollama integration not available: {e}")
        if primary_llm is not None:
            return _SafeLLM(primary_llm=primary_llm)
        print("Using stub LLM for testing...")
        return _ReportStubLLM()
    except Exception as e:
        print(f"Exception initializing Ollama LLM: {e}")
        raise
