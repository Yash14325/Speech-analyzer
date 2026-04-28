"""Personality Agent: Analyzes personality traits from communication patterns."""

import logging
from typing import Dict, Any, Optional

from llm_helper import llm
from llm1.prompt_templates import PERSONALITY_PROMPT
from utils.parser import safe_parse

# Configure logging
logger = logging.getLogger(__name__)

# RAG integration (optional dependency)
try:
    from rag.retriever import get_retriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.info("RAG not available - personality agent will run without context retrieval")

# Guardrails integration (optional dependency)
try:
    from guardrails_config import validate_agent_response
except ImportError:
    logger.info("Guardrails not available - using passthrough validation")
    def validate_agent_response(x: Any, _: str) -> Any:
        return x


class PersonalityAgentError(Exception):
    """Raised when personality agent encounters an error."""
    pass


def _get_personality_context(state: Dict[str, Any]) -> str:
    """
    Retrieve RAG context for personality analysis.
    
    Args:
        state: Pipeline state with previous agent analyses
        
    Returns:
        Retrieved context string, or empty string if retrieval fails
    """
    if not RAG_AVAILABLE:
        logger.debug("RAG not available, skipping context retrieval")
        return ""
    
    try:
        retriever = get_retriever()
        context = retriever.get_context_for_analysis("personality", state)
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


def personality_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze personality traits based on communication and confidence patterns.
    
    This agent synthesizes insights from previous agents to identify:
    - Personality type (Introvert/Ambivert/Extrovert)
    - Interaction style (Reserved/Balanced/Expressive)
    - Professional presence level
    - Key personality traits
    - Strengths and growth opportunities
    
    Args:
        state: Pipeline state dictionary containing:
            - communication_analysis (dict): Results from communication agent
            - confidence_emotion_analysis (dict): Results from confidence agent
            
    Returns:
        Dictionary with 'personality_analysis' key containing:
            - personality_type: Introvert/Ambivert/Extrovert
            - interaction_style: Reserved/Balanced/Expressive
            - professional_presence: Developing/Competent/Strong
            - key_personality_traits: List of traits
            - strengths_in_interaction: List of strengths
            - growth_opportunities: List of opportunities
            - overall_summary: Professional summary
            
    Raises:
        PersonalityAgentError: If agent execution fails critically
    """
    try:
        # Extract previous agent results
        communication_analysis = state.get("communication_analysis", {})
        confidence_analysis = state.get("confidence_emotion_analysis", {})
        
        if not communication_analysis:
            logger.warning("No communication analysis available for personality agent")
        if not confidence_analysis:
            logger.warning("No confidence analysis available for personality agent")
        
        # Build prompt
        prompt = PERSONALITY_PROMPT.format(
            rag_context="",  # RAG context not used in personality analysis
            communication_analysis=communication_analysis,
            confidence_analysis=confidence_analysis,
            communication_score=communication_analysis.get("communication_score", "N/A"),
            confidence_score=confidence_analysis.get("confidence_score", "N/A")
        )
        
        # Invoke LLM
        logger.debug("Invoking LLM for personality analysis")
        response = llm.invoke(prompt)
        
        # Parse and validate response
        parsed = safe_parse(response)
        validated = validate_agent_response(parsed, "personality_agent")
        
        logger.info("Personality analysis completed successfully")
        return {"personality_analysis": validated}
        
    except KeyError as e:
        logger.error(f"Missing required key in state: {e}")
        raise PersonalityAgentError(f"Invalid state structure: {e}")
    except Exception as e:
        logger.exception(f"Personality agent failed: {e}")
        raise PersonalityAgentError(f"Agent execution failed: {e}")
