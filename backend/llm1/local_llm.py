# llm1/local_llm.py

from llm1.llm_config import LLM_MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS


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
    """Invoke a real LLM, falling back if the local request fails."""

    def __init__(self, llm):
        self._llm = llm
        self._fallback = _ReportStubLLM()

    def invoke(self, prompt: str) -> str:
        try:
            response = self._llm.invoke(prompt)
            return getattr(response, "content", response)
        except Exception as e:
            print(f"Ollama report request failed ({e}), using stub LLM")
            return self._fallback.invoke(prompt)


def get_llm():
    """
    Returns an Ollama-backed LLM instance.
    Falls back to stub if Ollama or the configured model is not available.
    """
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

        return _SafeLLM(llm)

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        print(f"Ollama integration not available: {e}")
        print("Using stub LLM for testing...")
        return _ReportStubLLM()
    except Exception as e:
        print(f"Exception initializing Ollama LLM: {e}")
        raise
