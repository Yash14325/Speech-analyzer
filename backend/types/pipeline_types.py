"""Type definitions for pipeline state and data structures."""

from typing import Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    """Audio feature measurements from speech analysis."""
    speech_rate: float = Field(description="Speech rate in words per minute")
    pitch_variance: float = Field(description="Pitch variance measurement")
    pause_ratio: float = Field(description="Ratio of pauses to speech")
    energy_level: str = Field(description="Energy level assessment")


class PipelineState(TypedDict, total=False):
    """Type definition for pipeline state dictionary.
    
    This TypedDict defines the structure of the state object passed
    through the analysis pipeline.
    """
    transcript: str
    audio_features: Dict[str, Any]
    communication_analysis: Dict[str, Any]
    confidence_emotion_analysis: Dict[str, Any]
    personality_analysis: Dict[str, Any]


class AgentResults(TypedDict):
    """Type definition for combined agent results."""
    communication_analysis: Dict[str, Any]
    confidence_emotion_analysis: Dict[str, Any]
    personality_analysis: Dict[str, Any]
