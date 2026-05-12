import json
import os
import sys
import argparse
import subprocess

# Fix for DSPy readonly database error
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), ".dspy_cache")

import dspy
from pydantic import BaseModel, Field, ConfigDict

# Load tools from the current project
from src.vector import find_vector_matches
from src.svg2image import svg_to_png_cli
from dotenv import load_dotenv

load_dotenv()

# Setup DSPy Model
lm = dspy.LM(
    f"openai/{os.getenv('VISION_MODEL', 'gpt-4o')}", 
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("BASE_URL")
)
dspy.configure(lm=lm)

class CroppingMultipliers(BaseModel):
    left: float = Field(..., description="Fraction of width to crop from the left (e.g., 0.0 for none, 0.3 to remove left 30%)")
    right: float = Field(..., description="Fraction of width to crop from the right")
    top: float = Field(..., description="Fraction of height to crop from the top")
    bottom: float = Field(..., description="Fraction of height to crop from the bottom")
    
class CroppingDecisionSignature(dspy.Signature):
    """
    Analyze the provided CAD region image. The user is looking for a specific target drawing described by the query.
    Often, bounding boxes inadvertently capture adjacent, unrelated drawings, title blocks, schedules, or details.
    Determine if there are distinct, unrelated elements at the edges of the image that should be cropped out.
    Provide the fraction of the total width/height to remove from each edge.
    If an edge does NOT contain unrelated drawings, its crop value MUST be 0.0.
    Keep crops precise to avoid cutting into the target drawing.
    """
    query: str = dspy.InputField(desc="The target drawing the user wants to isolate")
    image_region: dspy.Image = dspy.InputField(desc="Image view of the current bounding box")
    
    cropping_decision: CroppingMultipliers = dspy.OutputField()

def visualize(file_path, bbox_str, output_path):
    cmd = [
        sys.executable,
        "-m",
        "src.visualize_cad",
        file_path,
        "--bbox", bbox_str,
        "--output", output_path
    ]
    subprocess.run(cmd, check=True)

def is_valid_bbox(bb):
    return bb and bb['max_x'] > bb['min_x'] and bb['max_y'] > bb['min_y']

def estimate_and_apply_cropping(file_path, keyword, output_dir="output_cropping_test"):
    print(f"\n{'='*80}")
    print(f"Cropping Estimation for '{keyword}' in {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading JSON data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("Running initial extractions to get a bounding box...")
    res_region = find_vector_matches(keyword, data=data, bbox_direct=True, topk=1, threshold=100, min_ar=0.35, max_ar=5.0)
    
    bb_reg = res_region['matches'].get('0', {}).get('candidates_mean_bbox_xyxy') or res_region['matches'].get(0, {}).get('candidates_mean_bbox_xyxy')
    if not bb_reg:
        for k in sorted(res_region['matches'].keys(), key=int):
            bb_reg = res_region['matches'][k]['candidates_mean_bbox_xyxy']
            break
            
    if not is_valid_bbox(bb_reg):
        print("Could not find any valid matches. Exiting.")
        return
        
    initial_bbox_str = f"{bb_reg['min_x']:.2f},{bb_reg['min_y']:.2f},{bb_reg['max_x']:.2f},{bb_reg['max_y']:.2f}"
    print(f"Initial BBox: {initial_bbox_str}")
    
    safe_kw = "".join([c if c.isalnum() else "_" for c in keyword])
    base_out = os.path.join(output_dir, f"cropping_test_{safe_kw}")
    initial_svg_path = f"{base_out}_initial.svg"
    initial_png_path = f"{base_out}_initial.png"
    
    print("\nRasterizing initial bounding box...")
    visualize(file_path, initial_bbox_str, initial_svg_path)
    svg_to_png_cli(initial_svg_path, initial_png_path, scale=4.0)
    
    print(f"\nAnalyzing cropping requirements with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
    cropping_module = dspy.ChainOfThought(CroppingDecisionSignature)
    
    try:
        output = cropping_module(
            query=keyword,
            image_region=dspy.Image.from_file(initial_png_path)
        )
        decision = output.cropping_decision
        
        print("\n=== Cropping Multipliers ===")
        print(f"Left: {decision.left}")
        print(f"Right: {decision.right}")
        print(f"Top: {decision.top}")
        print(f"Bottom: {decision.bottom}")
        print(f"Reasoning: {output.reasoning}")
        print("============================")
        
        # Apply cropping
        w = bb_reg['max_x'] - bb_reg['min_x']
        h = bb_reg['max_y'] - bb_reg['min_y']
        
        new_min_x = bb_reg['min_x'] + (w * decision.left)
        new_max_x = bb_reg['max_x'] - (w * decision.right)
        new_min_y = bb_reg['min_y'] + (h * decision.bottom)
        new_max_y = bb_reg['max_y'] - (h * decision.top)
        
        if new_min_x >= new_max_x or new_min_y >= new_max_y:
            print("Error: Cropping values are too large, resulting in invalid bounding box. Aborting crop.")
            return
            
        new_bbox_str = f"{new_min_x:.2f},{new_min_y:.2f},{new_max_x:.2f},{new_max_y:.2f}"
        print(f"\nNew Cropped BBox: {new_bbox_str}")
        
        cropped_svg_path = f"{base_out}_cropped.svg"
        cropped_png_path = f"{base_out}_cropped.png"
        
        print("\nRasterizing cropped bounding box...")
        visualize(file_path, new_bbox_str, cropped_svg_path)
        svg_to_png_cli(cropped_svg_path, cropped_png_path, scale=4.0)
        print(f"Saved cropped visualization to: {cropped_png_path}")
        
    except Exception as e:
        print(f"Error during VLM execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="storage/json/2生产综合楼-平面图及楼电梯详图(FJ-MC-10~19 25~32)_t3(1).json")
    parser.add_argument("--keyword", default="卫生间详图")
    args = parser.parse_args()
    
    estimate_and_apply_cropping(args.file, args.keyword)
