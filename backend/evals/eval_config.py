# evals/eval_config.py
"""
LangChain Evaluation Configuration for Speech Analysis Pipeline

This module provides evaluation capabilities using LangChain's built-in evaluators.
All evaluators are from langchain.evaluation - no custom evaluators needed.

Built-in Evaluators Used:
- CriteriaEvalChain: Evaluate based on criteria (helpfulness, relevance, coherence, etc.)
- ScoreStringEvalChain: Score outputs on a 1-10 scale
- JsonValidityEvaluator: Validate JSON structure
- StringDistanceEvalChain: Measure string similarity
- EmbeddingDistanceEvalChain: Semantic similarity using embeddings

Criteria Available:
- helpfulness, relevance, coherence, conciseness, depth, creativity, detail
"""

import json
import logging
from typing import Any, Dict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LangChain evaluation components
# Priority: langchain_classic (has full evaluation module)
try:
    # Try langchain_classic first (has the full evaluation module)
    from langchain_classic.evaluation import (
        CriteriaEvalChain,
        ScoreStringEvalChain,
        JsonValidityEvaluator,
    )
    LANGCHAIN_EVAL_AVAILABLE = True
    logger.info("✅ LangChain Classic evaluation module loaded")
except ImportError:
    try:
        # Try the old langchain path
        from langchain.evaluation import (
            CriteriaEvalChain,
            ScoreStringEvalChain,
            JsonValidityEvaluator,
        )
        LANGCHAIN_EVAL_AVAILABLE = True
        logger.info("✅ LangChain evaluation module loaded")
    except ImportError:
        # Define stubs for unavailable imports
        LANGCHAIN_EVAL_AVAILABLE = False
        CriteriaEvalChain = None
        ScoreStringEvalChain = None
        JsonValidityEvaluator = None
        logger.warning("⚠️ LangChain evaluation not available. Install langchain-classic.")


class EvalCriteria(str, Enum):
    """Pre-defined evaluation criteria from LangChain."""
    HELPFULNESS = "helpfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CONCISENESS = "conciseness"
    DEPTH = "depth"
    CREATIVITY = "creativity"
    DETAIL = "detail"


# Custom criteria for speech analysis domain
SPEECH_ANALYSIS_CRITERIA = {
    "actionability": "Does the analysis provide actionable feedback that the user can apply?",
    "specificity": "Is the analysis specific to the speech patterns rather than generic?",
    "accuracy": "Does the analysis accurately reflect the speech metrics provided?",
    "completeness": "Does the analysis cover all relevant aspects of speech (clarity, confidence, etc.)?",
    "professional_tone": "Is the analysis written in a professional, constructive tone?",
}


class SpeechAnalysisEvaluator:
    """
    Evaluator for Speech Analysis Pipeline outputs.
    Uses LangChain's built-in evaluators with speech-specific criteria.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the evaluator.
        
        Args:
            llm: LangChain LLM to use for evaluation. If None, uses the project's LLM.
        """
        self._llm = llm
        self._criteria_evaluators = {}
        self._score_evaluator = None
        self._json_evaluator = None
        
        if LANGCHAIN_EVAL_AVAILABLE:
            self._initialize_evaluators()
    
    def _get_llm(self):
        """Get the LLM for evaluation."""
        if self._llm is not None:
            return self._llm
        
        try:
            from llm1.local_llm import get_llm
            return get_llm()
        except Exception as e:
            logger.warning(f"Could not load LLM: {e}")
            return None
    
    def _initialize_evaluators(self):
        """Initialize LangChain evaluators."""
        llm = self._get_llm()
        if llm is None:
            logger.warning("No LLM available for evaluation")
            return
        
        try:
            # Initialize criteria evaluators for standard criteria
            for criteria in [EvalCriteria.HELPFULNESS, EvalCriteria.RELEVANCE, 
                           EvalCriteria.COHERENCE, EvalCriteria.CONCISENESS]:
                try:
                    evaluator = CriteriaEvalChain.from_llm(
                        llm=llm,
                        criteria=criteria.value
                    )
                    self._criteria_evaluators[criteria.value] = evaluator
                    logger.info(f"✅ Initialized {criteria.value} evaluator")
                except Exception as e:
                    logger.warning(f"Could not initialize {criteria.value} evaluator: {e}")
            
            # Initialize score evaluator (1-10 scale)
            try:
                self._score_evaluator = ScoreStringEvalChain.from_llm(
                    llm=llm,
                    criteria="helpfulness",
                    normalize_by=10.0  # Normalize to 0-1 scale
                )
                logger.info("✅ Initialized score evaluator")
            except Exception as e:
                logger.warning(f"Could not initialize score evaluator: {e}")
            
            # JSON validity evaluator (doesn't need LLM)
            try:
                self._json_evaluator = JsonValidityEvaluator()
                logger.info("✅ Initialized JSON validity evaluator")
            except Exception as e:
                logger.warning(f"Could not initialize JSON evaluator: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}")
    
    def evaluate_criteria(
        self,
        prediction: str,
        input_text: str,
        criteria: str | Dict[str, str] = "helpfulness"
    ) -> Dict[str, Any]:
        """
        Evaluate output against a specific criterion.
        
        Args:
            prediction: The model output to evaluate
            input_text: The input that generated the prediction
            criteria: The criterion to evaluate against (string for built-in, dict for custom)
            
        Returns:
            Dict with 'score', 'value', and 'reasoning'
        """
        if not LANGCHAIN_EVAL_AVAILABLE:
            return {"error": "LangChain evaluation not available", "score": None}
        
        # Handle custom criteria dict
        criteria_key = criteria if isinstance(criteria, str) else list(criteria.keys())[0]
        
        evaluator = self._criteria_evaluators.get(criteria_key)
        if evaluator is None:
            # Create on-the-fly for custom or new criteria
            llm = self._get_llm()
            if llm:
                try:
                    evaluator = CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)
                    # Cache it for reuse
                    self._criteria_evaluators[criteria_key] = evaluator
                except Exception as e:
                    return {"error": f"Could not create evaluator: {e}", "score": None}
            else:
                return {"error": "No evaluator available", "score": None}
        
        try:
            result = evaluator.evaluate_strings(
                prediction=prediction,
                input=input_text
            )
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e), "score": None}
    
    def evaluate_score(
        self,
        prediction: str,
        input_text: str
    ) -> Dict[str, Any]:
        """
        Score the output on a 1-10 scale.
        
        Args:
            prediction: The model output to score
            input_text: The input that generated the prediction
            
        Returns:
            Dict with 'score' (normalized 0-1) and 'reasoning'
        """
        if not LANGCHAIN_EVAL_AVAILABLE or self._score_evaluator is None:
            return {"error": "Score evaluator not available", "score": None}
        
        try:
            result = self._score_evaluator.evaluate_strings(
                prediction=prediction,
                input=input_text
            )
            return result
        except Exception as e:
            logger.error(f"Score evaluation failed: {e}")
            return {"error": str(e), "score": None}
    
    def evaluate_json_validity(self, json_string: str) -> Dict[str, Any]:
        """
        Check if output is valid JSON.
        
        Args:
            json_string: String to validate as JSON
            
        Returns:
            Dict with 'score' (1 for valid, 0 for invalid)
        """
        if self._json_evaluator is None:
            # Fallback to simple JSON parsing
            try:
                json.loads(json_string)
                return {"score": 1, "reasoning": "Valid JSON"}
            except json.JSONDecodeError as e:
                return {"score": 0, "reasoning": f"Invalid JSON: {e}"}
        
        try:
            result = self._json_evaluator.evaluate_strings(prediction=json_string)
            return result
        except Exception as e:
            return {"error": str(e), "score": None}
    
    def evaluate_agent_output(
        self,
        agent_output: Dict[str, Any],
        agent_name: str,
        input_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of an agent's output.
        
        Args:
            agent_output: The agent's output dictionary
            agent_name: Name of the agent (communication, confidence, personality)
            input_metrics: The input metrics provided to the agent
            
        Returns:
            Dict with evaluation scores for multiple criteria
        """
        results = {
            "agent": agent_name,
            "evaluations": {},
            "custom_criteria": {},
            "weak_areas": [],
            "passed_areas": []
        }
        
        # Convert output to string for evaluation
        if isinstance(agent_output, dict):
            prediction = json.dumps(agent_output, indent=2)
        else:
            prediction = str(agent_output)
        
        input_text = f"Analyze speech for {agent_name}. Metrics: {json.dumps(input_metrics)}"
        
        # Evaluate JSON validity
        json_result = self.evaluate_json_validity(prediction)
        results["evaluations"]["json_validity"] = json_result
        
        # Evaluate with LangChain built-in criteria
        for criterion in ["helpfulness", "relevance", "coherence"]:
            criterion_result = self.evaluate_criteria(
                prediction=prediction,
                input_text=input_text,
                criteria=criterion
            )
            results["evaluations"][criterion] = criterion_result
        
        # Evaluate with custom speech analysis criteria
        for criterion_name, criterion_desc in SPEECH_ANALYSIS_CRITERIA.items():
            custom_result = self.evaluate_criteria(
                prediction=prediction,
                input_text=input_text,
                criteria={criterion_name: criterion_desc}
            )
            results["custom_criteria"][criterion_name] = custom_result
            
            # Track weak/strong areas
            score = custom_result.get("score")
            if isinstance(score, (int, float)):
                if score < 0.6:
                    results["weak_areas"].append({
                        "criterion": criterion_name,
                        "description": criterion_desc,
                        "score": score,
                        "reasoning": custom_result.get("reasoning", "")
                    })
                else:
                    results["passed_areas"].append(criterion_name)
        
        # Calculate overall score from all evaluations
        all_scores = []
        for eval_result in results["evaluations"].values():
            if isinstance(eval_result.get("score"), (int, float)):
                all_scores.append(eval_result["score"])
        for eval_result in results["custom_criteria"].values():
            if isinstance(eval_result.get("score"), (int, float)):
                all_scores.append(eval_result["score"])
        
        results["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else None
        results["needs_refinement"] = len(results["weak_areas"]) > 0
        
        return results
    
    def evaluate_final_report(
        self,
        report: str,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the final generated report.
        
        Args:
            report: The final report text
            agent_outputs: The agent outputs used to generate the report
            
        Returns:
            Dict with comprehensive evaluation results
        """
        results = {
            "report_evaluation": {},
            "criteria_scores": {},
            "custom_criteria": {},
            "weak_areas": [],
            "passed_areas": []
        }
        
        input_text = f"Generate a speech analysis report based on: {json.dumps(agent_outputs)}"
        
        # Evaluate against LangChain built-in criteria
        criteria_to_evaluate = ["helpfulness", "relevance", "coherence", "conciseness"]
        
        for criterion in criteria_to_evaluate:
            result = self.evaluate_criteria(
                prediction=report,
                input_text=input_text,
                criteria=criterion
            )
            results["criteria_scores"][criterion] = result
        
        # Evaluate against custom speech analysis criteria
        for criterion_name, criterion_desc in SPEECH_ANALYSIS_CRITERIA.items():
            custom_result = self.evaluate_criteria(
                prediction=report,
                input_text=input_text,
                criteria={criterion_name: criterion_desc}
            )
            results["custom_criteria"][criterion_name] = custom_result
            
            # Track weak/strong areas
            score = custom_result.get("score")
            if isinstance(score, (int, float)):
                if score < 0.6:
                    results["weak_areas"].append({
                        "criterion": criterion_name,
                        "description": criterion_desc,
                        "score": score
                    })
                else:
                    results["passed_areas"].append(criterion_name)
        
        # Get overall score
        score_result = self.evaluate_score(
            prediction=report,
            input_text=input_text
        )
        results["report_evaluation"]["overall_score"] = score_result
        
        # Calculate average criteria score
        all_scores = []
        for result in results["criteria_scores"].values():
            if isinstance(result.get("score"), (int, float)):
                all_scores.append(result["score"])
        for result in results["custom_criteria"].values():
            if isinstance(result.get("score"), (int, float)):
                all_scores.append(result["score"])
        
        results["average_criteria_score"] = sum(all_scores) / len(all_scores) if all_scores else None
        results["needs_refinement"] = len(results["weak_areas"]) > 0
        
        return results
    
    def get_refinement_feedback(self, eval_results: Dict[str, Any]) -> str:
        """
        Generate refinement feedback based on evaluation results.
        
        Args:
            eval_results: Evaluation results from evaluate_agent_output or evaluate_final_report
            
        Returns:
            String feedback for refinement prompt
        """
        weak_areas = eval_results.get("weak_areas", [])
        if not weak_areas:
            return ""
        
        feedback_lines = ["The output needs improvement in the following areas:"]
        for weak in weak_areas:
            feedback_lines.append(
                f"- {weak['criterion']}: {weak['description']} (current score: {weak.get('score', 'N/A')})"
            )
        
        feedback_lines.append("\nPlease address these issues while maintaining accuracy.")
        return "\n".join(feedback_lines)


# Singleton instance
_evaluator = None


def get_evaluator(llm=None) -> SpeechAnalysisEvaluator:
    """Get the singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = SpeechAnalysisEvaluator(llm=llm)
    return _evaluator


def evaluate_agent(
    agent_output: Dict[str, Any],
    agent_name: str,
    input_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to evaluate an agent's output.
    
    Args:
        agent_output: The agent's output dictionary
        agent_name: Name of the agent
        input_metrics: Input metrics provided to the agent
        
    Returns:
        Evaluation results
    """
    evaluator = get_evaluator()
    return evaluator.evaluate_agent_output(agent_output, agent_name, input_metrics)


def evaluate_report(report: str, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate the final report.
    
    Args:
        report: The final report text
        agent_outputs: Agent outputs used to generate report
        
    Returns:
        Evaluation results
    """
    evaluator = get_evaluator()
    return evaluator.evaluate_final_report(report, agent_outputs)


def is_eval_available() -> bool:
    """Check if LangChain evaluation is available."""
    return LANGCHAIN_EVAL_AVAILABLE
