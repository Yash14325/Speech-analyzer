"""Confidence Agent: Analyzes vocal confidence and emotional tone."""

import logging
from typing import Dict, Any

from llm_helper import llm
from llm1.prompt_templates import CONFIDENCE_PROMPT
from utils.parser import safe_parse
from utils.feature_scoring import confidence_score

# Configure logging
logger = logging.getLogger(__name__)

# RAG integration (optional dependency)
try:
    from rag.retriever import get_retriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.info("RAG not available - confidence agent will run without context retrieval")

# Guardrails integration (optional dependency)
try:
    from guardrails_config import validate_agent_response
except ImportError:
    logger.info("Guardrails not available - using passthrough validation")
    def validate_agent_response(x: Any, _: str) -> Any:
        return x


class ConfidenceAgentError(Exception):
    """Raised when confidence agent encounters an error."""
    pass


def _get_confidence_context(state: Dict[str, Any]) -> str:
    """
    Retrieve RAG context for confidence analysis.
    
    Args:
        state: Pipeline state containing audio features
        
    Returns:
        Retrieved context string, or empty string if retrieval fails
    """
    if not RAG_AVAILABLE:
        logger.debug("RAG not available, skipping context retrieval")
        return ""
    
    try:
        retriever = get_retriever()
        audio_features = state.get("audio_features", {})
        
        if not audio_features:
            logger.warning("No audio features provided for RAG context retrieval")
            return ""
        
        context = retriever.get_context_for_analysis("confidence", audio_features)
        logger.debug(f"Retrieved RAG context: {len(context)} characters")
        return context
        
    except ConnectionError as e:
        logger.error(f"RAG retrieval connection failed: {e}")
        return ""
    except ValueError as e:
        logger.error(f"Invalid RAG retrieval parameters: {e}")
        return ""
    except Exception as e:
        logger.exception(f"Unexpected error in RAG retrieval: {e}")
        return ""


def confidence_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze confidence levels and emotional tone from audio features.
    
    This agent evaluates:
    - Vocal confidence indicators
    - Emotional tone (neutral, positive, nervous, assertive)
    - Vocal energy assessment
    - Confidence challenges and enhancement tips
    
    Args:
        state: Pipeline state dictionary containing:
            - audio_features (dict): Audio feature measurements including:
                - pitch_variance: Pitch variation measurement
                - energy_level: Vocal energy level
                - pause_ratio: Ratio of pauses to speech
            
    Returns:
        Dictionary with 'confidence_emotion_analysis' key containing:
            - confidence_score: 0-100 score
            - confidence_level: Low/Medium/High
            - emotional_tone: Neutral/Positive/Nervous/Assertive
            - vocal_energy_assessment: Low/Medium/High
            - confidence_indicators: List of confidence markers
            - possible_challenges: List of challenges
            - confidence_enhancement_tips: List of improvement tips
            
    Raises:
        ConfidenceAgentError: If agent execution fails critically
    """
    try:
        # Extract audio features
        audio_features = state.get("audio_features", {})
        if not audio_features:
            logger.warning("No audio features provided to confidence agent")
            raise ConfidenceAgentError("No audio features available")
        
        # Calculate confidence score
        score = confidence_score(audio_features)
        logger.info(f"Confidence score calculated: {score}")
        
        # Get RAG context for enhanced analysis
        rag_context = _get_confidence_context(state)
        
        # Build prompt
        prompt = CONFIDENCE_PROMPT.format(
            rag_context=f"EXPERT KNOWLEDGE:\n{rag_context}\n" if rag_context else "",
            pitch_variance=audio_features.get("pitch_variance", "N/A"),
            energy_level=audio_features.get("energy_level", "N/A"),
            pause_ratio=audio_features.get("pause_ratio", "N/A"),
            confidence_score=score
        )
        
        # Invoke LLM
        logger.debug("Invoking LLM for confidence analysis")
        response = llm.invoke(prompt)
        
        # Parse and validate response
        parsed = safe_parse(response)

        # Check for parse errors (sentinel keys from safe_parse)
        if "error" in parsed or "status" in parsed:
            error_msg = parsed.get("error", "Unknown parse error")
            status = parsed.get("status", "unknown")
            logger.error(
                f"Parse failure in confidence agent - error: {error_msg}, "
                f"status: {status}, raw: {parsed.get('raw', 'N/A')}"
            )
            raise ConfidenceAgentError(f"Failed to parse LLM response: {error_msg}")

        # Validate only genuinely parsed outputs
        validated = validate_agent_response(parsed, "confidence_agent")

        logger.info("Confidence analysis completed successfully")
        return {"confidence_emotion_analysis": validated}

    except Exception as e:
        logger.exception(f"Confidence agent failed: {e}")
        raise ConfidenceAgentError(f"Agent execution failed: {e}") from e
