# evals/eval_runner.py
"""
Evaluation Runner for Speech Analysis Pipeline

Provides batch evaluation and pipeline-level evaluation using LangChain evaluators.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import evaluation components
try:
    from evals.eval_config import (
        get_evaluator,
        is_eval_available,
    )
except ImportError:
    from eval_config import (
        get_evaluator,
        is_eval_available,
    )


class EvaluationRunner:
    """
    Runner for batch and pipeline evaluations.
    Uses LangChain's built-in evaluators.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the evaluation runner.
        
        Args:
            llm: Optional LLM for evaluation. Uses project LLM if not provided.
        """
        self.evaluator = get_evaluator(llm)
        self.results_history = []
    
    def evaluate_pipeline_output(
        self,
        pipeline_output: Dict[str, Any],
        include_agents: bool = True,
        include_report: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate complete pipeline output.
        
        Args:
            pipeline_output: Complete output from the speech analysis pipeline
            include_agents: Whether to evaluate individual agent outputs
            include_report: Whether to evaluate the final report
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_available": is_eval_available(),
            "agent_evaluations": {},
            "report_evaluation": None,
            "summary": {}
        }
        
        if not is_eval_available():
            results["error"] = "LangChain evaluation not available"
            return results
        
        # Extract components from pipeline output
        audio_features = pipeline_output.get("audio_features", {})
        transcript = pipeline_output.get("transcript", "")
        
        # Evaluate each agent if requested
        if include_agents:
            # Communication Agent
            comm_output = pipeline_output.get("communication_analysis", {})
            if comm_output:
                comm_metrics = {
                    "transcript": transcript[:200],
                    "speech_rate": audio_features.get("speech_rate"),
                    "pause_ratio": audio_features.get("pause_ratio")
                }
                results["agent_evaluations"]["communication"] = self.evaluator.evaluate_agent_output(
                    comm_output, "communication", comm_metrics
                )
            
            # Confidence Agent
            conf_output = pipeline_output.get("confidence_emotion_analysis", {})
            if conf_output:
                conf_metrics = {
                    "pitch_variance": audio_features.get("pitch_variance"),
                    "energy_level": audio_features.get("energy_level"),
                    "pause_ratio": audio_features.get("pause_ratio")
                }
                results["agent_evaluations"]["confidence"] = self.evaluator.evaluate_agent_output(
                    conf_output, "confidence", conf_metrics
                )
            
            # Personality Agent
            pers_output = pipeline_output.get("personality_analysis", {})
            if pers_output:
                pers_metrics = {
                    "communication_analysis": comm_output,
                    "confidence_analysis": conf_output
                }
                results["agent_evaluations"]["personality"] = self.evaluator.evaluate_agent_output(
                    pers_output, "personality", pers_metrics
                )
        
        # Evaluate final report if requested
        if include_report:
            report = pipeline_output.get("final_report", "")
            if report:
                agent_outputs = {
                    "communication_analysis": pipeline_output.get("communication_analysis", {}),
                    "confidence_emotion_analysis": pipeline_output.get("confidence_emotion_analysis", {}),
                    "personality_analysis": pipeline_output.get("personality_analysis", {})
                }
                results["report_evaluation"] = self.evaluator.evaluate_final_report(
                    report, agent_outputs
                )
        
        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results)
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        summary = {
            "agents_evaluated": len(results.get("agent_evaluations", {})),
            "report_evaluated": results.get("report_evaluation") is not None,
            "average_scores": {}
        }
        
        # Calculate average agent scores
        agent_scores = []
        for agent_name, agent_eval in results.get("agent_evaluations", {}).items():
            if agent_eval and agent_eval.get("overall_score") is not None:
                agent_scores.append(agent_eval["overall_score"])
                summary["average_scores"][agent_name] = agent_eval["overall_score"]
        
        if agent_scores:
            summary["average_agent_score"] = sum(agent_scores) / len(agent_scores)
        
        # Add report score
        report_eval = results.get("report_evaluation", {})
        if report_eval and report_eval.get("average_criteria_score") is not None:
            summary["report_score"] = report_eval["average_criteria_score"]
        
        return summary
    
    def run_batch(
        self,
        pipeline_outputs: List[Dict[str, Any]],
        include_agents: bool = True,
        include_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple pipeline outputs.
        
        Args:
            pipeline_outputs: List of pipeline outputs to evaluate
            include_agents: Whether to evaluate individual agents
            include_report: Whether to evaluate reports
            
        Returns:
            Batch evaluation results with aggregate statistics
        """
        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(pipeline_outputs),
            "individual_results": [],
            "aggregate_statistics": {}
        }
        
        all_agent_scores = []
        all_report_scores = []
        
        for i, output in enumerate(pipeline_outputs):
            logger.info(f"Evaluating sample {i+1}/{len(pipeline_outputs)}")
            result = self.evaluate_pipeline_output(
                output,
                include_agents=include_agents,
                include_report=include_report
            )
            batch_results["individual_results"].append(result)
            
            # Collect scores for aggregation
            summary = result.get("summary", {})
            if summary.get("average_agent_score") is not None:
                all_agent_scores.append(summary["average_agent_score"])
            if summary.get("report_score") is not None:
                all_report_scores.append(summary["report_score"])
        
        # Calculate aggregate statistics
        if all_agent_scores:
            batch_results["aggregate_statistics"]["mean_agent_score"] = sum(all_agent_scores) / len(all_agent_scores)
            batch_results["aggregate_statistics"]["min_agent_score"] = min(all_agent_scores)
            batch_results["aggregate_statistics"]["max_agent_score"] = max(all_agent_scores)
        
        if all_report_scores:
            batch_results["aggregate_statistics"]["mean_report_score"] = sum(all_report_scores) / len(all_report_scores)
            batch_results["aggregate_statistics"]["min_report_score"] = min(all_report_scores)
            batch_results["aggregate_statistics"]["max_report_score"] = max(all_report_scores)
        
        return batch_results
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.results_history
    
    def export_results(self, filepath: str) -> None:
        """Export evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        logger.info(f"Results exported to {filepath}")


def run_full_evaluation(pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run full evaluation on pipeline output.
    
    Args:
        pipeline_output: Complete pipeline output dictionary
        
    Returns:
        Evaluation results
    """
    runner = EvaluationRunner()
    return runner.evaluate_pipeline_output(pipeline_output)


def run_batch_evaluation(
    pipeline_outputs: List[Dict[str, Any]],
    export_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run batch evaluation.
    
    Args:
        pipeline_outputs: List of pipeline outputs
        export_path: Optional path to export results
        
    Returns:
        Batch evaluation results
    """
    runner = EvaluationRunner()
    results = runner.run_batch(pipeline_outputs)
    
    if export_path:
        runner.export_results(export_path)
    
    return results
