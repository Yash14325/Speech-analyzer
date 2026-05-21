# llm1/local_llm.py

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS


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
            max_tokens=MAX_TOKENS
        )
        llm = chat_model | StrOutputParser()
        return llm

    except (ImportError, ModuleNotFoundError, ValueError) as e:
        print(f"⚠️ NVIDIA API not available: {e}")
        print("   Using stub LLM for testing...")

        # Import stub from main llm module
        from llm.local_llm import _StubLLM
        return _StubLLM()
    except Exception as e:
        print(f"⚠️ Exception initializing NVIDIA LLM: {e}")
        raise
