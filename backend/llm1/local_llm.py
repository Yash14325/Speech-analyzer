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
        
        chat_model = ChatNVIDIA(
            model=LLM_MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        llm = chat_model | StrOutputParser()
        return llm

    except (ImportError, ModuleNotFoundError) as e:
        print(f"⚠️ NVIDIA API not available: {e}")
        print("   Using stub LLM for testing...")

        # Import stub from main llm module
        from llm.local_llm import _StubLLM
        return _StubLLM()
    except Exception:
        # Re-raise any other exception (API failures, auth, network, config)
        raise
