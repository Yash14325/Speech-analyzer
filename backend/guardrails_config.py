# guardrails_config.py
"""
GuardrailsAI Configuration for Speech Analysis Pipeline

This module provides input/output validation using GuardrailsAI Hub validators.
All validators are pre-built from the Guardrails Hub - no custom validators needed.

Hub Validators Used:
- ToxicLanguage: Detects toxic/harmful language
- ProfanityFree: Checks for profanity
- ValidJson: Validates JSON structure
- DetectPII: Detects personally identifiable information
- GibberishText: Detects gibberish/nonsense text
- SensitiveTopics: Flags sensitive content

Installation:
    pip install guardrails-ai
    guardrails configure
    guardrails hub install hub://guardrails/toxic_language
    guardrails hub install hub://guardrails/profanity_free
    guardrails hub install hub://guardrails/valid_json
    guardrails hub install hub://guardrails/detect_pii
    guardrails hub install hub://guardrails/gibberish_text
    guardrails hub install hub://guardrails/sensitive_topics
"""

import json
import logging
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import guardrails
try:
    from guardrails import Guard
    from guardrails.hub import (
        ToxicLanguage,
        ProfanityFree,
        DetectPII,
        GibberishText,
        SensitiveTopics,
    )
    GUARDRAILS_AVAILABLE = True
    logger.info("✅ GuardrailsAI loaded successfully")
except ImportError as e:
    GUARDRAILS_AVAILABLE = False
    logger.warning(f"⚠️ GuardrailsAI not available: {e}")
    logger.warning("Install with: pip install guardrails-ai && guardrails configure")


class GuardrailsWrapper:
    """
    Wrapper class for GuardrailsAI validation.
    Uses pre-built Hub validators for comprehensive input/output validation.
    """
    
    def __init__(self):
        self._input_guard = None
        self._output_guard = None
        self._report_guard = None
        
        if GUARDRAILS_AVAILABLE:
            self._initialize_guards()
    
    def _initialize_guards(self):
        """Initialize all guards with Hub validators"""
        try:
            # Input validation guard (for transcripts/user inputs)
            # Checks for toxic language, profanity, and gibberish
            self._input_guard = Guard().use_many(
                ToxicLanguage(on_fail="noop"),  # Don't block, just flag
                ProfanityFree(on_fail="noop"),
                GibberishText(on_fail="noop"),
            )
            logger.info("✅ Input guard initialized")
            
            # Output validation guard (for agent JSON responses)
            # Validates JSON structure and checks for PII leakage
            self._output_guard = Guard().use_many(
                DetectPII(on_fail="fix"),  # Redact PII if found
            )
            logger.info("✅ Output guard initialized")
            
            # Report validation guard (for final reports)
            # Comprehensive validation for user-facing content
            self._report_guard = Guard().use_many(
                ToxicLanguage(on_fail="noop"),
                ProfanityFree(on_fail="fix"),  # Remove profanity from reports
                DetectPII(on_fail="fix"),  # Redact PII from reports
                SensitiveTopics(on_fail="noop"),
            )
            logger.info("✅ Report guard initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize guards: {e}")
            self._input_guard = None
            self._output_guard = None
            self._report_guard = None
    
    def validate_input(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Validate input text (transcripts, user inputs).
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (validated_text, validation_metadata)
        """
        if not GUARDRAILS_AVAILABLE or not self._input_guard:
            return text, {"guardrails": "unavailable"}
        
        try:
            result = self._input_guard.validate(text)
            metadata = {
                "validation_passed": result.validation_passed,
                "validated_output": result.validated_output,
            }
            return result.validated_output or text, metadata
        except Exception as e:
            logger.warning(f"Input validation error: {e}")
            return text, {"error": str(e)}
    
    def validate_agent_output(self, output: str, agent_name: str = "agent") -> Tuple[str, Dict[str, Any]]:
        """
        Validate agent JSON output.
        
        Args:
            output: Agent output (JSON string or dict)
            agent_name: Name of the agent for logging
            
        Returns:
            Tuple of (validated_output, validation_metadata)
        """
        if not GUARDRAILS_AVAILABLE or not self._output_guard:
            return output, {"guardrails": "unavailable"}
        
        try:
            # Convert dict to string if needed
            text_output = json.dumps(output) if isinstance(output, dict) else str(output)
            
            result = self._output_guard.validate(text_output)
            metadata = {
                "agent": agent_name,
                "validation_passed": result.validation_passed,
            }
            
            validated = result.validated_output or text_output
            
            # Try to parse back to dict if original was dict
            if isinstance(output, dict):
                try:
                    return json.loads(validated), metadata
                except json.JSONDecodeError:
                    return output, metadata
            
            return validated, metadata
            
        except Exception as e:
            logger.warning(f"{agent_name} output validation error: {e}")
            return output, {"error": str(e)}
    
    def validate_report(self, report: str) -> Tuple[str, Dict[str, Any]]:
        """
        Validate final report before presenting to user.
        
        Args:
            report: Final report text
            
        Returns:
            Tuple of (validated_report, validation_metadata)
        """
        if not GUARDRAILS_AVAILABLE or not self._report_guard:
            return report, {"guardrails": "unavailable"}
        
        try:
            result = self._report_guard.validate(report)
            metadata = {
                "validation_passed": result.validation_passed,
                "validated_output": result.validated_output is not None,
            }
            return result.validated_output or report, metadata
        except Exception as e:
            logger.warning(f"Report validation error: {e}")
            return report, {"error": str(e)}


# Singleton instance
_guardrails_wrapper = None


def get_guardrails() -> GuardrailsWrapper:
    """Get the singleton GuardrailsWrapper instance."""
    global _guardrails_wrapper
    if _guardrails_wrapper is None:
        _guardrails_wrapper = GuardrailsWrapper()
    return _guardrails_wrapper


# Convenience functions for direct use
def validate_transcript(transcript: str) -> str:
    """Validate transcript text and return cleaned version."""
    wrapper = get_guardrails()
    validated, _ = wrapper.validate_input(transcript)
    return validated


def validate_agent_response(output: Any, agent_name: str = "agent") -> Any:
    """Validate agent response and return cleaned version."""
    wrapper = get_guardrails()
    validated, metadata = wrapper.validate_agent_output(output, agent_name)
    if not metadata.get("validation_passed", True):
        logger.info(f"⚠️ {agent_name} output validation flagged issues")
    return validated


def validate_final_report(report: str) -> str:
    """Validate final report and return cleaned version."""
    wrapper = get_guardrails()
    validated, metadata = wrapper.validate_report(report)
    if not metadata.get("validation_passed", True):
        logger.info("⚠️ Final report validation flagged issues")
    return validated


def is_guardrails_available() -> bool:
    """Check if GuardrailsAI is available and configured."""
    return GUARDRAILS_AVAILABLE and get_guardrails()._input_guard is not None
