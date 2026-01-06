"""
Agent evaluation tests using DeepEval built-in metrics.

Run with:
    pytest eval/test_agent_evaluation.py -v
    
Or with deepeval CLI:
    deepeval test run eval/test_agent_evaluation.py
"""
import json
import pytest
from pathlib import Path

from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)

from src.application.agent.graph import ProductResearchAgent


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def agent():
    """Initialize agent once for all tests."""
    return ProductResearchAgent()


@pytest.fixture(scope="module")
def golden_dataset():
    """Load golden dataset from JSON."""
    dataset_path = Path("eval/datasets/golden_dataset.json")
    
    if not dataset_path.exists():
        pytest.skip("Golden dataset not found. Run: python -m eval.generate_test_data")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Metrics Configuration
# ============================================================================

def get_answer_relevancy_metric():
    """Answer Relevancy: Does the answer address the user's query?"""
    return AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4o-mini",
        include_reason=True,
        async_mode=False
    )


def get_faithfulness_metric():
    """Faithfulness: Is the answer grounded in retrieved context?"""
    return FaithfulnessMetric(
        threshold=0.5,
        model="gpt-4o-mini",
        include_reason=True,
        async_mode=False
    )


def get_correctness_metric():
    """Correctness: Custom G-Eval for answer correctness."""
    return GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct and provides accurate information based on the context of the question.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model="gpt-4o-mini",
        async_mode=False
    )


def get_hallucination_metric():
    """Hallucination: Detect fabricated information."""
    return HallucinationMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
        async_mode=False
    )


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.asyncio
async def test_product_qa_relevancy(agent, golden_dataset):
    """Test Product QA queries for answer relevancy."""
    product_qa_cases = [c for c in golden_dataset["test_cases"] if c["category"] == "product_qa"]
    
    if not product_qa_cases:
        pytest.skip("No product_qa test cases found")
    
    # Test first 2 cases for quick feedback
    for case in product_qa_cases[:2]:
        result = await agent.query(case["input"])
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
        )
        
        metric = get_answer_relevancy_metric()
        metric.measure(test_case)
        
        assert metric.score >= 0.5, f"Answer relevancy failed for: {case['input'][:50]}... Score: {metric.score}"


@pytest.mark.asyncio
async def test_product_qa_faithfulness(agent, golden_dataset):
    """Test Product QA queries for faithfulness to retrieved context."""
    product_qa_cases = [c for c in golden_dataset["test_cases"] if c["category"] == "product_qa"]
    
    if not product_qa_cases:
        pytest.skip("No product_qa test cases found")
    
    for case in product_qa_cases[:2]:
        result = await agent.query(case["input"])
        
        # Build retrieval context from products
        retrieval_context = []
        for product in result.get("products", []):
            if isinstance(product, dict):
                retrieval_context.append(json.dumps(product))
            else:
                retrieval_context.append(str(product))
        
        # Skip if no context retrieved
        if not retrieval_context:
            continue
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
            retrieval_context=retrieval_context,
        )
        
        metric = get_faithfulness_metric()
        metric.measure(test_case)
        
        assert metric.score >= 0.5, f"Faithfulness failed for: {case['input'][:50]}... Score: {metric.score}"


@pytest.mark.asyncio
async def test_market_analysis_relevancy(agent, golden_dataset):
    """Test Market Analysis queries for answer relevancy."""
    market_cases = [c for c in golden_dataset["test_cases"] if c["category"] == "market_analysis"]
    
    if not market_cases:
        pytest.skip("No market_analysis test cases found")
    
    for case in market_cases[:2]:
        result = await agent.query(case["input"])
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
        )
        
        metric = get_answer_relevancy_metric()
        metric.measure(test_case)
        
        assert metric.score >= 0.5, f"Answer relevancy failed for: {case['input'][:50]}... Score: {metric.score}"


@pytest.mark.asyncio
async def test_market_analysis_hallucination(agent, golden_dataset):
    """Test Market Analysis queries for hallucination detection."""
    market_cases = [c for c in golden_dataset["test_cases"] if c["category"] == "market_analysis"]
    
    if not market_cases:
        pytest.skip("No market_analysis test cases found")
    
    for case in market_cases[:2]:
        result = await agent.query(case["input"])
        
        # Build context from sources
        context = []
        for source in result.get("sources", []):
            if isinstance(source, dict):
                context.append(source.get("snippet", str(source)))
            else:
                context.append(str(source))
        
        if not context:
            continue
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
            context=context,
        )
        
        metric = get_hallucination_metric()
        metric.measure(test_case)
        
        # Higher threshold = more tolerant (0.7 allows some hallucination)
        assert metric.score <= 0.7, f"Hallucination detected for: {case['input'][:50]}... Score: {metric.score}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # 5 minute timeout
async def test_multi_tool_correctness(agent, golden_dataset):
    """Test multi-tool queries for overall correctness."""
    multi_tool_cases = [c for c in golden_dataset["test_cases"] if c["category"] == "multi_tool"]
    
    if not multi_tool_cases:
        pytest.skip("No multi_tool test cases found")
    
    for case in multi_tool_cases[:2]:
        result = await agent.query(case["input"])
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
        )
        
        metric = get_correctness_metric()
        metric.measure(test_case)
        
        assert metric.score >= 0.5, f"Correctness failed for: {case['input'][:50]}... Score: {metric.score}"


# ============================================================================
# Batch Evaluation (for comprehensive testing)
# ============================================================================

@pytest.mark.asyncio
async def test_full_evaluation(agent, golden_dataset):
    """
    Run full evaluation on all test cases.
    
    This test is slower but provides comprehensive metrics.
    Run with: pytest eval/test_agent_evaluation.py::test_full_evaluation -v
    """
    test_cases = []
    
    # Limit to 5 cases for faster testing
    all_cases = golden_dataset["test_cases"][:5]
    
    for case in all_cases:
        result = await agent.query(case["input"])
        
        # Build context
        context = []
        for product in result.get("products", []):
            context.append(json.dumps(product) if isinstance(product, dict) else str(product))
        for source in result.get("sources", []):
            if isinstance(source, dict):
                context.append(source.get("snippet", str(source)))
        
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=result["answer"],
            retrieval_context=context if context else ["No context retrieved"],
        )
        test_cases.append(test_case)
    
    # Evaluate with multiple metrics
    metrics = [
        get_answer_relevancy_metric(),
        get_correctness_metric(),
    ]
    
    results = evaluate(test_cases=test_cases, metrics=metrics)
    
    # Calculate pass rate from test_results
    passed = sum(1 for r in results.test_results if r.success)
    total = len(results.test_results)
    pass_rate = passed / total if total > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases: {total}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {pass_rate:.1%}")
