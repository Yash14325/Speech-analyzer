# evals/test_evals.py
"""
Test script for LangChain-based evaluations.

Run: python -m evals.test_evals
"""

from datetime import datetime


def test_evaluation_imports():
    """Test that all evaluation imports work."""
    print("\n" + "="*60)
    print("TEST 1: Evaluation Imports")
    print("="*60)
    
    try:
        from evals.eval_config import (
            is_eval_available,
            EvalCriteria,
        )
        print("✅ eval_config imports successful")
        print(f"   LangChain Eval Available: {is_eval_available()}")
        print(f"   Criteria: {[c.value for c in EvalCriteria]}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_evaluator_initialization():
    """Test evaluator initialization."""
    print("\n" + "="*60)
    print("TEST 2: Evaluator Initialization")
    print("="*60)
    
    try:
        from evals.eval_config import get_evaluator, is_eval_available
        
        evaluator = get_evaluator()
        print("✅ Evaluator initialized")
        print(f"   Eval available: {is_eval_available()}")
        print(f"   Criteria evaluators: {list(evaluator._criteria_evaluators.keys())}")
        print(f"   Score evaluator: {'Yes' if evaluator._score_evaluator else 'No'}")
        print(f"   JSON evaluator: {'Yes' if evaluator._json_evaluator else 'No'}")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_json_evaluation():
    """Test JSON validity evaluation."""
    print("\n" + "="*60)
    print("TEST 3: JSON Validity Evaluation")
    print("="*60)
    
    try:
        from evals.eval_config import get_evaluator
        
        evaluator = get_evaluator()
        
        # Test valid JSON
        valid_json = '{"fluency_level": "good", "clarity_score": 85}'
        result = evaluator.evaluate_json_validity(valid_json)
        print("✅ Valid JSON test:")
        print(f"   Input: {valid_json}")
        print(f"   Result: {result}")
        
        # Test invalid JSON
        invalid_json = '{fluency_level: good, clarity_score: 85'
        result = evaluator.evaluate_json_validity(invalid_json)
        print("\n✅ Invalid JSON test:")
        print(f"   Input: {invalid_json}")
        print(f"   Result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ JSON evaluation failed: {e}")
        return False


def test_agent_evaluation():
    """Test agent output evaluation."""
    print("\n" + "="*60)
    print("TEST 4: Agent Output Evaluation")
    print("="*60)
    
    try:
        from evals.eval_config import get_evaluator, is_eval_available
        
        if not is_eval_available():
            print("⚠️ LangChain eval not available, skipping LLM-based tests")
            return True
        
        evaluator = get_evaluator()
        
        # Sample agent output
        agent_output = {
            "fluency_level": "good",
            "clarity_score": 85,
            "speech_structure": "well-organized",
            "vocabulary_usage": "appropriate",
            "areas_for_improvement": ["reduce filler words", "vary pace"]
        }
        
        input_metrics = {
            "transcript": "Hello, I wanted to talk about the importance of communication skills...",
            "speech_rate": 145,
            "pause_ratio": 0.15
        }
        
        result = evaluator.evaluate_agent_output(
            agent_output,
            "communication",
            input_metrics
        )
        
        print("✅ Agent evaluation completed:")
        print(f"   Agent: {result.get('agent')}")
        print(f"   Overall Score: {result.get('overall_score')}")
        print(f"   Evaluations: {list(result.get('evaluations', {}).keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Agent evaluation failed: {e}")
        return False


def test_criteria_evaluation():
    """Test criteria-based evaluation."""
    print("\n" + "="*60)
    print("TEST 5: Criteria Evaluation")
    print("="*60)
    
    try:
        from evals.eval_config import get_evaluator, is_eval_available
        
        if not is_eval_available():
            print("⚠️ LangChain eval not available, skipping")
            return True
        
        evaluator = get_evaluator()
        
        prediction = """
        Based on the speech analysis, the speaker demonstrates good fluency 
        with a speech rate of 145 words per minute. The pause ratio of 0.15 
        indicates natural pauses. Recommendations: Practice reducing filler 
        words and varying pace for more engaging delivery.
        """
        
        input_text = "Analyze speech with rate=145wpm, pause_ratio=0.15"
        
        for criterion in ["helpfulness", "relevance", "coherence"]:
            result = evaluator.evaluate_criteria(
                prediction=prediction,
                input_text=input_text,
                criteria=criterion
            )
            print(f"\n✅ {criterion.upper()} evaluation:")
            print(f"   Score: {result.get('score')}")
            print(f"   Value: {result.get('value')}")
        
        return True
    except Exception as e:
        print(f"❌ Criteria evaluation failed: {e}")
        return False


def test_runner():
    """Test the evaluation runner."""
    print("\n" + "="*60)
    print("TEST 6: Evaluation Runner")
    print("="*60)
    
    try:
        from evals.eval_runner import EvaluationRunner
        
        # Sample pipeline output
        pipeline_output = {
            "transcript": "Hello, I wanted to discuss the importance of public speaking...",
            "audio_features": {
                "speech_rate": 145,
                "pause_ratio": 0.15,
                "pitch_variance": 0.25,
                "energy_level": 0.7
            },
            "communication_analysis": {
                "fluency_level": "good",
                "clarity_score": 85,
                "speech_structure": "well-organized"
            },
            "confidence_emotion_analysis": {
                "confidence_level": "high",
                "emotion": "enthusiastic",
                "nervousness": "low"
            },
            "personality_analysis": {
                "assertiveness": "moderate",
                "sociability": "high",
                "speaking_style": "engaging"
            },
            "final_report": """
            Speech Analysis Report
            =====================
            Overall, the speaker demonstrates strong communication skills with 
            good fluency and high confidence. The speech was well-organized 
            with an engaging delivery style.
            
            Recommendations:
            1. Continue practicing varied pacing
            2. Maintain the enthusiastic tone
            3. Consider adding more structured pauses for emphasis
            """
        }
        
        runner = EvaluationRunner()
        result = runner.evaluate_pipeline_output(pipeline_output)
        
        print("✅ Runner evaluation completed:")
        print(f"   Timestamp: {result.get('timestamp')}")
        print(f"   Eval Available: {result.get('evaluation_available')}")
        print(f"   Agents Evaluated: {result.get('summary', {}).get('agents_evaluated')}")
        print(f"   Report Evaluated: {result.get('summary', {}).get('report_evaluated')}")
        
        if result.get('summary', {}).get('average_agent_score'):
            print(f"   Avg Agent Score: {result['summary']['average_agent_score']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all evaluation tests."""
    print("\n" + "="*60)
    print("LANGCHAIN EVALUATION TESTS")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*60)
    
    tests = [
        ("Imports", test_evaluation_imports),
        ("Initialization", test_evaluator_initialization),
        ("JSON Evaluation", test_json_evaluation),
        ("Agent Evaluation", test_agent_evaluation),
        ("Criteria Evaluation", test_criteria_evaluation),
        ("Runner", test_runner),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
