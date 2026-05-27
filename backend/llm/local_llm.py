"""LLM wrapper used by agents.

This module tries NVIDIA hosted NIM first, then Ollama, then a lightweight
stub that returns deterministic JSON for testing purposes.
"""

import json
import threading

from llm1.llm_config import (
    LLM_MODEL_NAME,
    MAX_TOKENS,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)
from llm1.nvidia_llm import NvidiaChatLLM


class _StubLLM:
    """Fallback LLM that returns deterministic JSON for testing when Ollama is unavailable."""

    def invoke(self, prompt: str) -> str:
        p = prompt.lower() if prompt else ""
        if "communication skills analyst" in p:
            resp = {
                "communication_score": 85,
                "clarity_level": "High",
                "fluency_level": "High",
                "speech_pacing": "Balanced",
                "key_observations": ["Clear delivery"],
                "communication_strengths": ["Strong enunciation"],
                "communication_gaps": ["None"],
                "improvement_suggestions": ["Keep it up"],
            }
        elif "voice confidence analyst" in p:
            resp = {
                "confidence_score": 90,
                "confidence_level": "High",
                "emotional_tone": "Assertive",
                "vocal_energy_assessment": "High",
                "confidence_indicators": ["Steady pitch"],
                "possible_challenges": ["None"],
                "confidence_enhancement_tips": ["Continue current pace"],
            }
        elif "personality insight engine" in p:
            resp = {
                "personality_type": "Ambivert",
                "interaction_style": "Balanced",
                "professional_presence": "Strong",
                "key_personality_traits": ["Adaptable"],
                "strengths_in_interaction": ["Good listener"],
                "growth_opportunities": ["None"],
                "overall_summary": "Professional and balanced.",
            }
        elif (
            "communication coach" in p
            or "final report" in p
            or "improvement recommendations" in p
            or "communication overview" in p
        ):
            return """
**Communication Overview**
- Clarity Score: 85/100 (Good)
- Fluency: Good with structured delivery
- Vocabulary: Advanced level

**Confidence & Emotional Tone**
- Confidence Level: High
- Nervousness: Low
- Emotional State: Calm and composed

**Personality Insights**
- Type: Balanced communicator
- Assertiveness: Moderate
- Expressiveness: Moderate

**Key Strengths**
- Clear and structured communication
- Confident delivery with controlled emotions
- Professional and balanced approach

**Improvement Recommendations**
- Continue practicing for a more natural flow
- Add more vocal variety for engagement
- Maintain your current confident pace

*Note: This is a stub response - Ollama is not running.*
""".strip()
        else:
            resp = {"message": "stub response", "note": "Ollama is not running - using fallback"}
        return json.dumps(resp)


class _LazyPrimaryLLM:
    """Lazy-loading wrapper that tries NVIDIA first, then Ollama, then stub."""

    def __init__(self):
        self._nvidia_llm = None
        self._nvidia_disabled = False
        self._llm = None
        self._initialized = False
        self._init_lock = threading.Lock()

    def _get_nvidia_llm(self):
        if not NVIDIA_API_KEY or self._nvidia_disabled:
            return None

        if self._nvidia_llm is None:
            self._nvidia_llm = NvidiaChatLLM(json_mode=True)
            print(f"Using NVIDIA NIM primary model: {NVIDIA_MODEL}")
        return self._nvidia_llm

    def _get_ollama_llm(self):
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    try:
                        try:
                            from langchain_ollama import OllamaLLM

                            self._llm = OllamaLLM(
                                model=LLM_MODEL_NAME,
                                base_url=OLLAMA_BASE_URL,
                                temperature=TEMPERATURE,
                                num_predict=MAX_TOKENS,
                                format="json",
                            )
                        except (ImportError, ModuleNotFoundError):
                            from langchain_community.llms import Ollama

                            self._llm = Ollama(
                                model=LLM_MODEL_NAME,
                                base_url=OLLAMA_BASE_URL,
                                temperature=TEMPERATURE,
                                num_predict=MAX_TOKENS,
                            )
                        self._initialized = True
                    except (ImportError, ModuleNotFoundError, ValueError) as e:
                        print(f"Ollama not available ({e}), using stub LLM")
                        self._llm = _StubLLM()
                        self._initialized = True
                    except Exception as e:
                        print(f"Exception initializing Ollama LLM: {e}")
                        raise
        return self._llm

    def invoke(self, prompt: str) -> str:
        nvidia_llm = self._get_nvidia_llm()
        if nvidia_llm is not None:
            try:
                return nvidia_llm.invoke(prompt)
            except Exception as e:
                print(f"NVIDIA LLM request failed ({e}), falling back to Ollama")
                self._nvidia_disabled = True

        llm = self._get_ollama_llm()
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", response)
        except Exception as e:
            print(f"Ollama LLM request failed ({e}), using stub LLM")
            with self._init_lock:
                self._llm = _StubLLM()
            self._initialized = True
            return self._llm.invoke(prompt)


llm = _LazyPrimaryLLM()
