import argparse
import os
from typing import Literal

# Fix for DSPy readonly database error MUST BE BEFORE IMPORTING DSPY
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), ".dspy_cache")

import dspy
from dotenv import load_dotenv

load_dotenv()

# Setup DSPy Model (using text-only model since this is purely text classification)
lm = dspy.LM(
    f"openai/{os.getenv('MODEL', 'qwen3.5-flash')}", 
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("BASE_URL")
)
dspy.configure(lm=lm)

class QueryClassificationSignature(dspy.Signature):
    """
    Classify a CAD query into one of two categories:
    - 'general drawing'
    - 'sub-region'

    You MUST follow this strict decision process:

    ─────────────────────────────
    STEP 1 — Identify the PRIMARY TARGET of the query
    ─────────────────────────────
    Determine whether the query refers to:
    A) An entire building / floor / elevation / section
    B) A specific component / room / system / detail

    ─────────────────────────────
    STEP 2 — Apply PRIORITY RULE (CRITICAL)
    ─────────────────────────────
    If the query contains ANY specific component, it MUST be classified as 'sub-region',
    even if it also contains general drawing terms like "平面图".

    Example:
    - "电梯平面图" → sub-region (because 电梯 is a component)
    - "楼梯平面图" → sub-region
    - "卫生间详图" → sub-region

    COMPONENT KEYWORDS (non-exhaustive):
    电梯 (elevator), 楼梯 (stairs), 卫生间 (toilet), 厨房 (kitchen),
    幕墙 (curtain wall), 门 (door), 窗 (window), 管道 (pipe), 设备 (equipment)

    ─────────────────────────────
    STEP 3 — Identify GENERAL DRAWINGS
    ─────────────────────────────
    Classify as 'general drawing' ONLY if the query refers to:
    - Entire floor plans: "一层平面图", "二层平面图"
    - Elevations: "立面图", "南立面图"
    - Sections: "剖面图"
    - Whole-building or whole-system views

    GENERAL KEYWORDS:
    层 (floor), 总体 (overall), 立面 (elevation), 剖面 (section)

    ─────────────────────────────
    STEP 4 — Resolve Ambiguity
    ─────────────────────────────
    If the query is ambiguous or short:
    - If it implies a specific object → 'sub-region'
    - Otherwise → 'general drawing'

    When uncertain, default to 'sub-region' (more specific).

    ─────────────────────────────
    STEP 5 — STRICT OUTPUT REQUIREMENTS
    ─────────────────────────────
    - classification MUST be exactly:
        'general drawing' OR 'sub-region'
    - reasoning MUST reference:
        (1) detected keyword(s)
        (2) why they imply general vs specific

    BAD reasoning example:
    "It looks like a plan drawing."

    GOOD reasoning example:
    "Contains '电梯', which is a specific component, so it is classified as sub-region despite '平面图'."
    """
    
    query: str = dspy.InputField(desc="The CAD drawing query to classify.")
    classification: Literal["general drawing", "sub-region"] = dspy.OutputField(desc="Must be exactly 'general drawing' or 'sub-region'.")
    reasoning: str = dspy.OutputField(desc="Brief but specific explanation referencing keywords.")

def classify_query(query: str) -> tuple[str, str]:
    """
    Uses an LLM via DSPy to classify the query.
    Returns a tuple of (classification, reasoning).
    """
    classifier = dspy.Predict(QueryClassificationSignature)
    
    try:
        result = classifier(query=query)
        # Clean up output just in case the model adds extra quotes or spaces
        classification = result.classification.strip().lower().strip("'\"")
        if classification not in ["general drawing", "sub-region"]:
            # Fallback if model disobeys instructions
            classification = "sub-region" if "sub-region" in classification else "general drawing"
        return classification, result.reasoning
    except Exception as e:
        print(f"Error during classification: {e}")
        return "unknown", str(e)

def test_samples():
    print("Testing sample queries from run_comparisons.sh:\n")
    
    samples = [
        "地下一层平面图",
        "六~八层平面图",
        "屋顶平面图",
        "电梯平面图",
        "楼梯",
        "二层平面图",
        "轴立面图",
        "双层玻璃幕墙",
        "剖面图",
        "2#楼梯、4#电梯平面图",
        "防雨百叶窗",
        "水平铝合金遮阳",
        "呼叫坐席",
        "kitchen",
        "walk-in-closet",
        "great room"
    ]
    
    print(f"{'Query':<20} | {'Classification':<20} | {'Reasoning'}")
    print("-" * 100)
    for q in samples:
        classification, reasoning = classify_query(q)
        print(f"{q:<20} | {classification:<20} | {reasoning}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify CAD queries using DSPy.")
    parser.add_argument("--query", type=str, help="The query string to classify")
    parser.add_argument("--test", action="store_true", help="Run against predefined sample queries")
    
    args = parser.parse_args()
    
    if args.test or not args.query:
        test_samples()
    elif args.query:
        print(f"Query: {args.query}")
        classification, reasoning = classify_query(args.query)
        print(f"Classification: {classification}")
        print(f"Reasoning: {reasoning}")