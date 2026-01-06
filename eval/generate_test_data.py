"""
Generate synthetic test cases for agent evaluation using DeepEval.

This script generates test cases for:
1. Product QA queries (product catalog RAG)
2. Market Analysis queries (web search)
3. Multi-tool queries (combining both)

Usage:
    python -m eval.generate_test_data

Output:
    eval/datasets/golden_dataset.json
"""
import json
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig


def generate_product_qa_cases(num_cases: int = 20) -> list:
    """
    Generate test cases for Product QA scenarios.
    
    Based on examples:
    - "What wireless headphones do we have in stock?"
    - "Show me high-rated electronics under $100"
    - "Which products from AudioMax brand are bestsellers?"
    """
    styling_config = StylingConfig(
        input_format="Questions about e-commerce product catalog in natural language",
        expected_output_format="Natural language answer with specific product details (name, price, category, stock status)",
        task="Answer questions about product catalog using RAG retrieval from product database",
        scenario="Product managers asking about inventory, pricing, stock availability, and product specifications",
    )
    
    synthesizer = Synthesizer(
        model="gpt-4o-mini",
        styling_config=styling_config
    )
    
    goldens = synthesizer.generate_goldens_from_scratch(
        num_goldens=num_cases,
    )
    
    return [{"input": g.input, "expected_output": g.expected_output, "category": "product_qa"} for g in goldens]


def generate_market_analysis_cases(num_cases: int = 15) -> list:
    """
    Generate test cases for Market Analysis scenarios.
    
    Based on examples:
    - "What is the current market price for noise-cancelling headphones?"
    - "Latest reviews for Sony WH-1000XM5"
    - "Trending products in home fitness equipment"
    """
    styling_config = StylingConfig(
        input_format="Questions about market trends, competitor pricing, and product reviews",
        expected_output_format="Natural language answer with market insights, price comparisons, and source references",
        task="Search the web for current market information and provide analysis",
        scenario="Product managers researching market trends, competitor pricing, and customer reviews",
    )
    
    synthesizer = Synthesizer(
        model="gpt-4o-mini",
        styling_config=styling_config
    )
    
    goldens = synthesizer.generate_goldens_from_scratch(
        num_goldens=num_cases,
    )
    
    return [{"input": g.input, "expected_output": g.expected_output, "category": "market_analysis"} for g in goldens]


def generate_multi_tool_cases(num_cases: int = 15) -> list:
    """
    Generate test cases requiring multiple tools.
    
    Based on examples:
    - "Should we adjust AudioMax headphones pricing vs competitors?"
    - "Compare our Sony headphone prices with market prices"
    """
    styling_config = StylingConfig(
        input_format="Complex questions requiring both internal product data and external market research",
        expected_output_format="Comprehensive analysis with internal pricing, market comparison, and strategic recommendations",
        task="Analyze products from internal database and compare with market data to provide recommendations",
        scenario="Product managers making strategic decisions about pricing, inventory, and market positioning",
    )
    
    synthesizer = Synthesizer(
        model="gpt-4o-mini",
        styling_config=styling_config
    )
    
    goldens = synthesizer.generate_goldens_from_scratch(
        num_goldens=num_cases,
    )
    
    return [{"input": g.input, "expected_output": g.expected_output, "category": "multi_tool"} for g in goldens]


def main():
    """Generate all test cases and save to JSON."""
    print("[INFO] Generating synthetic test cases with DeepEval...")
    print("       This may take a few minutes...\n")
    
    # Generate test cases
    print("[1/3] Generating Product QA cases...")
    product_qa = generate_product_qa_cases(5)
    print(f"      [OK] Generated {len(product_qa)} cases\n")
    
    print("[2/3] Generating Market Analysis cases...")
    market_analysis = generate_market_analysis_cases(5)
    print(f"      [OK] Generated {len(market_analysis)} cases\n")
    
    print("[3/3] Generating Multi-tool cases...")
    multi_tool = generate_multi_tool_cases(5)
    print(f"      [OK] Generated {len(multi_tool)} cases\n")
    
    # Combine all cases
    all_cases = product_qa + market_analysis + multi_tool
    
    # Save to JSON
    output_path = Path("eval/datasets/golden_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "metadata": {
            "total_cases": len(all_cases),
            "categories": {
                "product_qa": len(product_qa),
                "market_analysis": len(market_analysis),
                "multi_tool": len(multi_tool)
            }
        },
        "test_cases": all_cases
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("=" * 50)
    print(f"[DONE] Generated {len(all_cases)} total test cases")
    print(f"       - Product QA: {len(product_qa)}")
    print(f"       - Market Analysis: {len(market_analysis)}")
    print(f"       - Multi-tool: {len(multi_tool)}")
    print(f"\n[OUTPUT] Saved to: {output_path}")


if __name__ == "__main__":
    main()
