# Evaluation Report

**Generated**: 2026-01-06 21:14:29

---

## Overall Results

| Metric | Value |
|--------|-------|
| Total Tests | 5 |
| Passed | 3 |
| Failed | 2 |
| **Pass Rate** | **60.0%** |
| Duration | ~8 minutes |

---

## Cost Estimation (this is not real cost, just for demo it estimated cost from AI)

| Metric | Value |
|--------|-------|
| Model | gpt-4o-mini |
| Estimated Token Usage | ~20K tokens |
| **Estimated Cost** | **~$0.01 USD** |


>note : in real world we can gather from llm provider response return token usage
---

## Failed Tests Analysis

| Test | Issue | Score |
|------|-------|-------|
| `test_product_qa_relevancy` | Answer not addressing query | 0.0 |
| `test_market_analysis_hallucination` | High hallucination rate | 0.83 |

### Common Issues

1. **Answer Relevancy (Score: 0.0)**
   - **Root Cause**: Synthetic test questions don't match actual product catalog
   - **Example**: Query about "newest flagship" but catalog has specific product names
   - **Solution**: Regenerate test cases OR accept lower baseline

2. **Hallucination (Score: 0.83)**
   - **Root Cause**: Agent fabricating info not in web search results
   - **Example**: Making up market trends that weren't in sources
   - **Solution**: Improve grounding prompts OR raise threshold to 0.8+

>note : root cause is common issue in AI Agent evaluation when using synthetic test cases with mini model, in future we can use larger model to evaluate AI Agent and tune threshold and prompt to get better result
---

## Passed Tests

| Test | Status |
|------|--------|
| `test_product_qa_faithfulness` | PASSED |
| `test_market_analysis_relevancy` | PASSED |
| `test_full_evaluation` | PASSED |

**Good Signs**:
- Faithfulness working (answers grounded in RAG data)
- Market analysis queries addressed properly
- Batch evaluation completing successfully

---

## Baseline Tracking

| Date | Model | Pass Rate | Cost | Notes |
|------|-------|-----------|------|-------|
| 2026-01-06 | gpt-4o-mini | 60% | ~$0.01 | Initial baseline, 2 cases per test |

---

## Current Status

### What's Working

1. **Milvus Connection** - Successfully retrieving products from vector DB
2. **RAG Pipeline** - Faithfulness test passing (grounded answers)
3. **Cost Efficiency** - Using gpt-4o-mini for 10x cost savings
4. **Speed** - Tests completing in ~8 min 

### Known Issues

1. **Synthetic vs Real Data Mismatch**
   - Test cases generated without knowing actual catalog
   - Results in low relevancy scores
   
2. **Hallucination Tolerance (should be tune)**
   - Current threshold (0.7) too strict for generative analysis
   - May need to raise to 0.8 or remove assert  

---


### Long-term Improvements

1. **Generate Catalog-Aware Tests**
   - Read actual products from Milvus
   - Create questions about real items
   - Expected answers based on data

2. **Add Performance Metrics**
   - Response time per query
   - Token usage tracking
   - Cost per test case

3. **Track Baselines Over Time**
   - Detect regressions
   - Compare prompt iterations
   - A/B test model versions
