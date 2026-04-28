"""Communication Agent: Analyzes speech communication patterns and clarity."""

import logging
from typing import Dict, Any, Optional

from llm_helper import llm
from llm1.prompt_templates import COMMUNICATION_PROMPT
from utils.parser import safe_parse
from utils.feature_scoring import communication_score

# Configure logging
logger = logging.getLogger(__name__)

# RAG integration (optional dependency)
try:
    from rag.retriever import get_retriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.info("RAG not available - communication agent will run without context retrieval")

# Guardrails integration (optional dependency)
try:
    from guardrails_config import validate_agent_response
except ImportError:
    logger.info("Guardrails not available - using passthrough validation")
    def validate_agent_response(x: Any, _: str) -> Any:
        return x


class CommunicationAgentError(Exception):
    """Raised when communication agent encounters an error."""
    pass


def _get_communication_context(state: Dict[str, Any]) -> str:
    """
    Retrieve RAG context for communication analysis.
    
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
        
        context = retriever.get_context_for_analysis("communication", audio_features)
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


def communication_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze communication patterns from speech transcript and audio features.
    
    This agent evaluates:
    - Clarity and articulation
    - Fluency and coherence
    - Speech pacing (rate and pauses)
    - Communication strengths and gaps
    
    Args:
        state: Pipeline state dictionary containing:
            - transcript (str): Speech transcript
            - audio_features (dict): Audio feature measurements
            
    Returns:
        Dictionary with 'communication_analysis' key containing:
            - communication_score: 0-100 score
            - clarity_level: Low/Medium/High
            - fluency_level: Low/Medium/High
            - speech_pacing: Too Slow/Balanced/Too Fast
            - key_observations: List of observations
            - communication_strengths: List of strengths
            - communication_gaps: List of areas for improvement
            - improvement_suggestions: List of actionable suggestions
            
    Raises:
        CommunicationAgentError: If agent execution fails critically
    """
    try:
        # Extract and validate transcript
        transcript = state.get("transcript", "").strip()
        if not transcript:
            logger.warning("Empty transcript provided to communication agent")
            raise CommunicationAgentError("No transcript available")
        
        # Extract audio features
        audio_features = state.get("audio_features", {})
        if not audio_features:
            logger.warning("No audio features provided to communication agent")
        
        # Calculate communication score
        score = communication_score(audio_features)
        logger.info(f"Communication score calculated: {score}")
        
        # Get RAG context for enhanced analysis
        rag_context = _get_communication_context(state)
        
        # Build prompt
        prompt = COMMUNICATION_PROMPT.format(
            rag_context=f"EXPERT KNOWLEDGE:\n{rag_context}\n" if rag_context else "",
            transcript=transcript[:500],  # Limit transcript length for prompt
            speech_rate=audio_features.get("speech_rate", "N/A"),
            pause_ratio=audio_features.get("pause_ratio", "N/A"),
            communication_score=score
        )
        
        # Invoke LLM
        logger.debug("Invoking LLM for communication analysis")
        response = llm.invoke(prompt)
        
        # Parse and validate response
        parsed = safe_parse(response)
        validated = validate_agent_response(parsed, "communication_agent")
        
        logger.info("Communication analysis completed successfully")
        return {"communication_analysis": validated}
        
    except KeyError as e:
        logger.error(f"Missing required key in state: {e}")
        raise CommunicationAgentError(f"Invalid state structure: {e}")
    except Exception as e:
        logger.exception(f"Communication agent failed: {e}")
        raise CommunicationAgentError(f"Agent execution failed: {e}")
