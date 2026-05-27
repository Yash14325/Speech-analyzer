# llm1/report_generator.py
"""
Report generation using RAG-enhanced LLM.

This module provides generate_final_report(), which mirrors
rag.rag_pipeline.rag_enhanced_report() for tests and direct use.
"""

from llm1.local_llm import get_llm
from llm1.prompt_templates import REPORT_PROMPT

try:
    from rag.retriever import get_retriever

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG module not available, proceeding without retrieval augmentation")

try:
    from guardrails_config import validate_final_report
except ImportError:

    def validate_final_report(x):
        return x


def _get_rag_context(agent_outputs: dict) -> str:
    """Retrieve relevant improvement recommendations from the knowledge base."""
    if not RAG_AVAILABLE:
        return ""

    try:
        retriever = get_retriever()

        comm = agent_outputs.get("communication_analysis", {})
        conf = agent_outputs.get("confidence_emotion_analysis", {})
        pers = agent_outputs.get("personality_analysis", {})

        weak_areas = []

        if isinstance(comm, dict):
            if comm.get("communication_score", comm.get("clarity_score", 100)) < 70:
                weak_areas.append("clarity")
            fluency = str(comm.get("fluency_level", "")).lower()
            if fluency in ["poor", "average", "low", "medium"]:
                weak_areas.append("fluency")
            pacing = str(comm.get("speech_pacing", "")).lower()
            if pacing in ["too slow", "too fast"]:
                weak_areas.append("speech pacing")

        if isinstance(conf, dict):
            confidence = str(conf.get("confidence_level", "")).lower()
            if confidence in ["low", "medium"]:
                weak_areas.append("confidence")
            energy = str(conf.get("vocal_energy_assessment", "")).lower()
            if energy in ["low", "moderate"]:
                weak_areas.append("vocal energy")

        if isinstance(pers, dict):
            presence = str(pers.get("professional_presence", "")).lower()
            if presence in ["developing", "competent"]:
                weak_areas.append("professional presence")

        improve_metrics = {"weak_areas": weak_areas or ["general speaking skills"]}
        return retriever.get_context_for_analysis("improvement", improve_metrics)

    except Exception as e:
        print(f"RAG context retrieval failed: {e}")
        return ""


def generate_final_report(agent_outputs: dict):
    """
    Convert agent outputs into a user-friendly AI report.
    Uses RAG context when available and falls back gracefully when it is not.
    """
    llm = get_llm()
    rag_context = _get_rag_context(agent_outputs)
    prompt = REPORT_PROMPT.format(
        rag_context=rag_context or "No specific recommendations available.",
        agent_outputs=agent_outputs,
    )

    report = llm.invoke(prompt)
    return validate_final_report(report)
