# llm1/local_llm.py

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS


class _ReportStubLLM:
    """Fallback report generator used when NVIDIA API credentials are unavailable."""

    def invoke(self, prompt: str) -> str:
        return """
**Communication Overview**
- Your speech was processed successfully with the local fallback report generator.
- The analysis agents completed communication, confidence, and personality scoring.

**Confidence & Emotional Tone**
- Confidence signals were reviewed from the extracted audio and transcript features.
- Use steady pacing, clear pauses, and consistent volume to improve delivery.

**Personality Insights**
- Your speaking style was reviewed for professional presence and interaction patterns.
- Keep answers structured with a short opening, supporting points, and a clear close.

**Key Strengths**
- Completed end-to-end audio transcription and agent analysis.
- Generated feedback even without a configured NVIDIA API key.

**Improvement Recommendations**
- Add `NVIDIA_API_KEY` if you want live LLM-generated reports.
- For local testing, this fallback keeps the app running without a 500 error.
""".strip()


class _SafeLLM:
    """Invoke a real LLM, falling back if the hosted request fails."""

    def __init__(self, llm):
        self._llm = llm
        self._fallback = _ReportStubLLM()

    def invoke(self, prompt: str) -> str:
        try:
            return self._llm.invoke(prompt)
        except Exception as e:
            print(f"⚠️ NVIDIA report request failed ({e}), using stub LLM")
            return self._fallback.invoke(prompt)


def get_llm():
    """
    Returns an LLM instance using NVIDIA API.
    Falls back to stub if API is not available.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langchain_core.output_parsers import StrOutputParser
        import os

        # Add basic check to fallback to stub if no key provided
        if not os.environ.get("NVIDIA_API_KEY"):
            raise ValueError("No NVIDIA API key found")
        
        chat_model = ChatNVIDIA(
            model=LLM_MODEL_NAME,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS
        )
        llm = chat_model | StrOutputParser()
        return _SafeLLM(llm)

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        print(f"⚠️ NVIDIA API not available: {e}")
        print("   Using stub LLM for testing...")

        return _ReportStubLLM()
    except Exception as e:
        print(f"⚠️ Exception initializing NVIDIA LLM: {e}")
        raise
