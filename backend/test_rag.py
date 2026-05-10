# test_rag.py
"""
Test script for RAG system implementation
Tests the retrieval and integration with LLM
"""

import json


def test_knowledge_base():
    """Test the knowledge base loading"""
    print("\n" + "="*50)
    print("🧪 TEST 1: Knowledge Base")
    print("="*50)
    
    from rag.knowledge_base import KnowledgeBase
    
    kb = KnowledgeBase()
    docs = kb.get_all_documents()
    
    print(f"✅ Loaded {len(docs)} knowledge documents")
    
    # Test category filtering
    comm_docs = kb.get_documents_by_category("communication")
    conf_docs = kb.get_documents_by_category("confidence")
    pers_docs = kb.get_documents_by_category("personality")
    improve_docs = kb.get_documents_by_category("improvement")
    
    print(f"   - Communication docs: {len(comm_docs)}")
    print(f"   - Confidence docs: {len(conf_docs)}")
    print(f"   - Personality docs: {len(pers_docs)}")
    print(f"   - Improvement docs: {len(improve_docs)}")
    
    return True


def test_retriever():
    """Test the RAG retriever"""
    print("\n" + "="*50)
    print("🧪 TEST 2: RAG Retriever")
    print("="*50)
    
    from rag.retriever import get_retriever
    
    retriever = get_retriever()
    
    # Test basic retrieval
    query = "speech rate confidence pitch"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"✅ Retrieved {len(results)} documents for query: '{query}'")
    
    for i, doc in enumerate(results, 1):
        print(f"\n   📄 Result {i}:")
        print(f"      Category: {doc.get('category')}")
        print(f"      Content preview: {doc.get('content', '')[:100]}...")
    
    return len(results) > 0


def test_context_retrieval():
    """Test context retrieval for analysis types"""
    print("\n" + "="*50)
    print("🧪 TEST 3: Context Retrieval for Analysis")
    print("="*50)
    
    from rag.retriever import get_retriever
    
    retriever = get_retriever()
    
    # Test communication context
    comm_context = retriever.get_context_for_analysis(
        "communication", 
        {"speech_rate": 145}
    )
    print(f"✅ Communication context length: {len(comm_context)} chars")
    
    # Test confidence context
    conf_context = retriever.get_context_for_analysis(
        "confidence",
        {"energy_level": "medium"}
    )
    print(f"✅ Confidence context length: {len(conf_context)} chars")
    
    # Test improvement context
    improve_context = retriever.get_context_for_analysis(
        "improvement",
        {}
    )
    print(f"✅ Improvement context length: {len(improve_context)} chars")
    
    return True


def test_agent_integration():
    """Test that agents work with RAG integration"""
    print("\n" + "="*50)
    print("🧪 TEST 4: Agent Integration with RAG")
    print("="*50)
    
    from agent import run_agents
    
    sample_state = {
        "transcript": "I believe we can achieve our goals through focused effort and collaboration.",
        "audio_features": {
            "speech_rate": 135,
            "pitch_variance": 22.5,
            "pause_ratio": 0.18,
            "energy_level": "medium-high"
        }
    }
    
    try:
        result = run_agents(sample_state)
        
        print("✅ Agents executed successfully with RAG")
        print("\n📌 Communication Analysis:")
        print(json.dumps(result.get("communication_analysis"), indent=2))
        
        print("\n📌 Confidence Analysis:")
        print(json.dumps(result.get("confidence_emotion_analysis"), indent=2))
        
        print("\n📌 Personality Analysis:")
        print(json.dumps(result.get("personality_analysis"), indent=2))
        
        return "error" not in str(result).lower()
    
    except Exception as e:
        print(f"❌ Agent integration failed: {e}")
        return False


def test_report_generation():
    """Test final report generation with RAG"""
    print("\n" + "="*50)
    print("🧪 TEST 5: Report Generation with RAG")
    print("="*50)
    
    from llm1.report_generator import generate_final_report
    
    sample_agent_outputs = {
        "communication_analysis": {
            "clarity_score": 82,
            "fluency_level": "Good",
            "speech_structure": "Structured",
            "vocabulary_level": "Intermediate"
        },
        "confidence_emotion_analysis": {
            "confidence_level": "Moderate",
            "nervousness": "Low",
            "emotion": "Engaged"
        },
        "personality_analysis": {
            "personality_type": "Balanced",
            "assertiveness": "Moderate",
            "expressiveness": "Moderate"
        }
    }
    
    try:
        print("Generating report with RAG context...")
        report = generate_final_report(sample_agent_outputs)
        
        print("\n✅ Report generated successfully!")
        print(f"   Report length: {len(report)} characters")
        print("\n📝 Report Preview (first 500 chars):")
        print("-" * 40)
        print(report[:500] if len(report) > 500 else report)
        print("-" * 40)
        
        return True
    
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False


def main():
    """Run all RAG tests"""
    print("\n" + "="*60)
    print("🚀 RAG SYSTEM TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Knowledge Base
    try:
        results["Knowledge Base"] = test_knowledge_base()
    except Exception as e:
        print(f"❌ Knowledge Base test failed: {e}")
        results["Knowledge Base"] = False
    
    # Test 2: Retriever
    try:
        results["Retriever"] = test_retriever()
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        results["Retriever"] = False
    
    # Test 3: Context Retrieval
    try:
        results["Context Retrieval"] = test_context_retrieval()
    except Exception as e:
        print(f"❌ Context Retrieval test failed: {e}")
        results["Context Retrieval"] = False
    
    # Test 4: Agent Integration
    try:
        results["Agent Integration"] = test_agent_integration()
    except Exception as e:
        print(f"❌ Agent Integration test failed: {e}")
        results["Agent Integration"] = False
    
    # Test 5: Report Generation (optional - requires LLM)
    try:
        results["Report Generation"] = test_report_generation()
    except Exception as e:
        print(f"⚠️ Report Generation test skipped: {e}")
        results["Report Generation"] = None
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        if passed is True:
            print(f"   ✅ {test_name}: PASSED")
        elif passed is False:
            print(f"   ❌ {test_name}: FAILED")
        else:
            print(f"   ⚠️ {test_name}: SKIPPED")
    
    passed_count = sum(1 for v in results.values() if v is True)
    total_count = len([v for v in results.values() if v is not None])
    
    print(f"\n   Total: {passed_count}/{total_count} tests passed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
