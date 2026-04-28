"""JSON parsing utilities for LLM outputs with error recovery."""

import json
import re
import logging
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


def _balance_braces(text: str) -> str:
    """
    Attempt to auto-close missing JSON braces/brackets.
    
    Helps when LLM output is truncated mid-JSON.
    
    Args:
        text: Input text possibly containing incomplete JSON
        
    Returns:
        Text with balanced braces and brackets
    """
    open_curly = text.count("{")
    close_curly = text.count("}")
    open_square = text.count("[")
    close_square = text.count("]")

    text += "}" * max(0, open_curly - close_curly)
    text += "]" * max(0, open_square - close_square)

    return text


def safe_parse(s: Optional[Union[str, dict]]) -> Dict[str, Any]:
    """
    Robust JSON parser for LLM output with multiple recovery strategies.

    Handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Single quotes instead of double quotes
    - Trailing commas
    - Partial/truncated JSON (common with Ollama)
    
    Args:
        s: LLM output string or dict to parse
        
    Returns:
        Parsed dictionary. On failure, returns dict with 'error' and 'status' keys.
        
    Examples:
        >>> safe_parse('{"score": 85}')
        {'score': 85}
        
        >>> safe_parse('```json\\n{"score": 85}\\n```')
        {'score': 85}
        
        >>> safe_parse("Here's the analysis: {'score': 85, 'level': 'high',}")
        {'score': 85, 'level': 'high'}
    """
    # Handle None input
    if s is None:
        logger.warning("Received None input for JSON parsing")
        return {"error": "no response", "status": "failed"}

    # Already a dict - return as-is
    if isinstance(s, dict):
        return s

    # Convert to string
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()

    # 1️⃣ Try direct JSON parse
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            logger.debug("Successfully parsed JSON directly")
            return result
        else:
            logger.warning(f"JSON parsed but not a dict (got {type(result).__name__}), falling back")
    except json.JSONDecodeError:
        # Note: TypeError is not expected here because we convert to string above (line 72)
        logger.debug("Direct JSON parse failed, trying recovery strategies")

    # 2️⃣ Strip markdown code blocks
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if code_block:
        try:
            result = json.loads(code_block.group(1).strip())
            if isinstance(result, dict):
                logger.debug("Successfully parsed JSON from markdown code block")
                return result
            else:
                logger.warning(f"Code block parsed but not a dict (got {type(result).__name__})")
                s = code_block.group(1).strip()
        except json.JSONDecodeError:
            s = code_block.group(1).strip()
            logger.debug("Code block found but invalid JSON, continuing recovery")

    # 3️⃣ Extract probable JSON region
    start = s.find("{")
    if start == -1:
        logger.warning("No JSON object found in input")
        return {
            "raw": s[:500],
            "error": "No JSON object found",
            "status": "parse_failed"
        }

    candidate = s[start:]

    # 4️⃣ Auto-fix common issues
    candidate = _balance_braces(candidate)
    logger.debug("Balanced braces/brackets")

    # Replace single quotes carefully (not after backslash)
    candidate = re.sub(r"(?<!\\)'", '"', candidate)

    # Remove trailing commas
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    logger.debug("Fixed common JSON issues (quotes, trailing commas)")

    # 5️⃣ Final parse attempt
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            logger.info("Successfully parsed JSON after recovery")
            return result
        else:
            logger.error(f"JSON parsed but not a dict (got {type(result).__name__})")
            return {
                "raw": candidate[:500],
                "error": f"JSON is not a dict (got {type(result).__name__})",
                "status": "parse_failed"
            }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed after all recovery attempts: {e}")
        return {
            "raw": candidate[:500],
            "error": f"JSON parse failed after recovery: {str(e)}",
            "status": "parse_failed"
        }


__all__ = ["safe_parse"]
