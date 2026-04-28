"""Pydantic schemas for agent outputs ensuring type safety and validation."""

from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class ClarityLevel(str, Enum):
    """Enumeration for clarity/fluency/confidence levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SpeechPacing(str, Enum):
    """Enumeration for speech pacing assessment."""
    TOO_SLOW = "Too Slow"
    BALANCED = "Balanced"
    TOO_FAST = "Too Fast"


class EmotionalTone(str, Enum):
    """Enumeration for emotional tone."""
    NEUTRAL = "Neutral"
    POSITIVE = "Positive"
    NERVOUS = "Nervous"
    ASSERTIVE = "Assertive"


class PersonalityType(str, Enum):
    """Enumeration for personality type classification."""
    INTROVERT = "Introvert"
    AMBIVERT = "Ambivert"
    EXTROVERT = "Extrovert"


class InteractionStyle(str, Enum):
    """Enumeration for interaction style."""
    RESERVED = "Reserved"
    BALANCED = "Balanced"
    EXPRESSIVE = "Expressive"


class ProfessionalPresence(str, Enum):
    """Enumeration for professional presence level."""
    DEVELOPING = "Developing"
    COMPETENT = "Competent"
    STRONG = "Strong"


class CommunicationAnalysis(BaseModel):
    """Structured output schema for Communication Agent.
    
    Ensures consistent, validated output from the communication analysis agent.
    """
    communication_score: int = Field(
        ge=0, 
        le=100, 
        description="Communication quality score (0-100)"
    )
    clarity_level: ClarityLevel = Field(
        description="Overall clarity of communication"
    )
    fluency_level: ClarityLevel = Field(
        description="Speech fluency level"
    )
    speech_pacing: SpeechPacing = Field(
        description="Assessment of speech pacing"
    )
    key_observations: List[str] = Field(
        min_length=1,
        max_length=5,
        description="List up to 5 key observations (concise, specific insights or patterns noticed in the communication). Limit to 5 to focus on the most important points and avoid overwhelming detail."
    )
    communication_strengths: List[str] = Field(
        description="Identified communication strengths"
    )
    communication_gaps: List[str] = Field(
        description="Areas needing improvement"
    )
    improvement_suggestions: List[str] = Field(
        description="Actionable improvement suggestions"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ConfidenceAnalysis(BaseModel):
    """Structured output schema for Confidence Agent.
    
    Ensures consistent, validated output from the confidence analysis agent.
    """
    confidence_score: int = Field(
        ge=0,
        le=100,
        description="Confidence score (0-100)"
    )
    confidence_level: ClarityLevel = Field(
        description="Overall confidence level"
    )
    emotional_tone: EmotionalTone = Field(
        description="Detected emotional tone"
    )
    vocal_energy_assessment: ClarityLevel = Field(
        description="Vocal energy level assessment"
    )
    confidence_indicators: List[str] = Field(
        description="Indicators of confidence in speech"
    )
    possible_challenges: List[str] = Field(
        description="Potential confidence challenges"
    )
    confidence_enhancement_tips: List[str] = Field(
        description="Tips for enhancing confidence"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class PersonalityAnalysis(BaseModel):
    """Structured output schema for Personality Agent.
    
    Ensures consistent, validated output from the personality analysis agent.
    """
    personality_type: PersonalityType = Field(
        description="Personality type classification"
    )
    interaction_style: InteractionStyle = Field(
        description="Interaction style assessment"
    )
    professional_presence: ProfessionalPresence = Field(
        description="Professional presence level"
    )
    key_personality_traits: List[str] = Field(
        description="Key personality traits identified"
    )
    strengths_in_interaction: List[str] = Field(
        description="Strengths in interpersonal interaction"
    )
    growth_opportunities: List[str] = Field(
        description="Opportunities for personal growth"
    )
    overall_summary: str = Field(
        min_length=10,
        description="Professional summary of personality analysis"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
