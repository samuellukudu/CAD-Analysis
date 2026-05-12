import json
import os
import sys
import argparse
import subprocess
import cv2
import numpy as np

# Fix for DSPy readonly database error
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), ".dspy_cache")

import dspy
from pydantic import BaseModel, Field, ConfigDict

# Load tools from the current project
from src.occupancy_grids import get_occupancy_grid
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

class BoundaryCheck(BaseModel):
    is_complete: bool = Field(..., description="True if the entire floor plan/region is fully visible and not cut off at the edges.")
    cut_off_edges: list[str] = Field(..., description="List of edges that are cut off (e.g., ['left', 'top', 'right', 'bottom']). Empty list if none.")

    model_config = ConfigDict(extra="forbid")

class EdgeAnalysis(BaseModel):
    left: str
    right: str
    top: str
    bottom: str


class BoundaryCheckSignature(dspy.Signature):
    """
    Analyze the provided CAD region image and determine whether the MAIN DRAWING CONTENT 
    (floor plan, rooms, walls, grids) is fully contained within the image.

    IMPORTANT: The presence of a sheet border DOES NOT imply completeness.
    You must evaluate whether the *actual architectural content* is truncated.
    
    You are also provided with `opencv_margins`, which are exact physical distances from 
    the drawing content to the image edges. 
    If a margin is 0 (or very close to 0), it is a MATHEMATICAL CERTAINTY that the image 
    was cut off at that edge.

    You MUST follow this decision process:

    STEP 1 — Identify primary drawing content:
    Focus on walls, rooms, columns, grid lines, dimensions, and annotations.
    Ignore title blocks, legends, and outer sheet borders.

    STEP 2 — Check each edge independently (top, bottom, left, right):
    For EACH edge, determine if there is evidence of truncation.

    An edge is CUT OFF if ANY of the following are observed:
    - The `opencv_margins` for that edge is 0 (this is an automatic CUT OFF)
    - Walls or rooms are sliced by the image boundary
    - Grid lines or axes start/stop abruptly at the edge
    - Dimension lines extend toward the edge but do not terminate
    - Repeated spacing (e.g., grids) suggests continuation beyond the frame
    - Text or annotations are partially visible or clipped

    STEP 3 — Grid continuity check (CRITICAL):
    - If grid numbering does not start from an expected origin (e.g., starts at 11),
      assume missing content → mark that edge as CUT OFF
    - If spacing between grids suggests continuation beyond visible area → CUT OFF

    STEP 4 — Reject false completeness:
    Even if a border is visible on all sides, the drawing is NOT complete if:
    - The main floor plan is not centered or extends toward edges
    - Structural elements are cut
    - Only a portion of the building is shown

    STEP 5 — Final classification:
    - is_complete = False if ANY edge is cut off
    - cut_off_edges must explicitly list affected edges

    BAD reasoning example (DO NOT DO):
    "The border is visible so the drawing is complete."

    GOOD reasoning example:
    "The left edge is cut off because the OpenCV left margin is 0px, and grid numbering starts at 11, implying missing grids 1–10."

    Your output must be conservative:
    When uncertain, prefer marking edges as CUT OFF rather than assuming completeness.
    """
    query: str = dspy.InputField(desc="The user query describing the target region to focus on")
    opencv_margins: str = dspy.InputField(desc="Programmatically detected pixel margins from the content to the image edges (Left, Top, Right, Bottom). If 0, content hits the very edge.")
    image_region: dspy.Image = dspy.InputField(desc="Image view of the current bounding box")
    
    edge_analysis: EdgeAnalysis = dspy.OutputField(desc="Detailed analysis of each edge (left, right, top, bottom) checking for cut-offs")
    boundary_assessment: BoundaryCheck = dspy.OutputField(desc="The final assessment of the image boundaries based on the edge analysis")


class PaddingMultipliers(BaseModel):
    left: float = Field(..., description="Padding multiplier for the left edge. Must be between 0.0 and 1.0 (e.g., 0.0 for no padding, 0.2 for 20% width extension)")
    right: float = Field(..., description="Padding multiplier for the right edge. Must be between 0.0 and 1.0")
    top: float = Field(..., description="Padding multiplier for the top edge. Must be between 0.0 and 1.0")
    bottom: float = Field(..., description="Padding multiplier for the bottom edge. Must be between 0.0 and 1.0")
    
class PaddingDecisionSignature(dspy.Signature):
    """
    Analyze the CAD region image and the known cut-off edges.
    Estimate how much additional area is needed on each cut-off edge 
    to recover the missing drawing content.

    IMPORTANT: You must ADD padding to the edges that ARE cut off to recover missing content.
    Do NOT confuse padding with margin. Padding is the extra space we want to add.

    ─────────────────────────────
    STEP 1 — Focus only on cut-off edges
    ─────────────────────────────
    - The `cut_off_edges` input tells you exactly which edges need to be expanded.
    - If an edge is listed as CUT OFF, it MUST receive a positive padding value (e.g., > 0.0).
    - If an edge is NOT listed as cut off, it is complete and MUST receive EXACTLY 0.0.

    ─────────────────────────────
    STEP 2 — Identify truncated structures
    ─────────────────────────────
    For each cut-off edge, examine:
    - Walls or rooms cut by the boundary
    - Grid spacing patterns (distance between columns/axes)
    - Dimension lines indicating continuation
    - Repeating structural patterns

    ─────────────────────────────
    STEP 3 — Estimate missing extent (RELATIVE)
    ─────────────────────────────
    You must estimate how far the drawing likely continues BEYOND the edge,
    relative to the current image size.

    You are provided with `bbox_context`, which includes both the 'Initial BBox' 
    (the original matched region) and the 'Current BBox' (the region shown in the 
    current image) in CAD units (min_x, min_y, max_x, max_y). 
    This helps you understand the physical scale of the drawing 
    (width = max_x - min_x, height = max_y - min_y).

    Use visual cues such as:
    - Distance between adjacent grid lines
    - Size of visible rooms or bays

    ─────────────────────────────
    STEP 4 — Normalize to multipliers (CRITICAL)
    ─────────────────────────────
    Convert the estimated missing distance into a percentage multiplier between 0.0 and 1.0.
    - Horizontal edges → percentage of image width
    - Vertical edges → percentage of image height
    - 0.1 = 10% extension, 0.5 = 50% extension, 1.0 = 100% extension.
    - NEVER exceed 1.0. NEVER output raw pixel values or CAD unit values.

    ─────────────────────────────
    OUTPUT REQUIREMENTS
    ─────────────────────────────
    - Non-cut edges → EXACTLY 0.0
    - Cut edges → positive float between 0.01 and 1.0 based on visual estimation
    - All values must be valid fractions representing percentages.

    BAD behavior:
    - Giving a cut-off edge 0.0 padding (this defeats the purpose).
    - Giving a non-cut edge > 0.0 padding.
    - Using raw numbers like 20.0 or 466.0.

    GOOD behavior:
    - "Left edge is cut off. One full grid spacing is missing. This spacing is roughly 15% of the image width → padding = 0.15"
    """
    
    cut_off_edges: str = dspy.InputField(desc="Details about which edges are cut off, including the boundary analysis reasoning to provide context")
    image_region: dspy.Image = dspy.InputField(desc="Image view of the current bounding box")
    bbox_context: str = dspy.InputField(desc="Context containing the initial and current bounding box coordinates in CAD units (min_x, min_y, max_x, max_y)")
    
    padding_decision: PaddingMultipliers = dspy.OutputField()

def analyze_margins_opencv(image_path):
    """
    Uses OpenCV to calculate the exact pixel margin from the CAD content to the image edges.
    Returns a dictionary of margins and a boolean indicating if it's a hard cut-off.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"left": -1, "top": -1, "right": -1, "bottom": -1}, False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    
    if coords is None:
        return {"left": -1, "top": -1, "right": -1, "bottom": -1}, False
        
    x, y, w, h = cv2.boundingRect(coords)
    
    margins = {
        "left": x,
        "top": y,
        "right": img.shape[1] - (x + w),
        "bottom": img.shape[0] - (y + h)
    }
    
    # If any margin is exactly 0, it's a hard cut-off.
    is_hard_cutoff = any(v == 0 for v in margins.values())
    
    return margins, is_hard_cutoff


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

def estimate_and_apply_padding(file_path, keyword, output_dir="output_padding_test"):
    print(f"\n{'='*80}")
    print(f"Boundary Check & Padding Estimation for '{keyword}' in {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading JSON data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("Running initial extractions to get a bounding box...")
    res_region = find_region_matches(keyword, data=data, bbox_direct=True, topk=1, threshold=100, min_ar=0.35, max_ar=5.0)
    
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
    base_out = os.path.join(output_dir, f"padding_test_{safe_kw}")
    initial_svg_path = f"{base_out}_initial.svg"
    initial_png_path = f"{base_out}_initial.png"
    
    print("\nRasterizing initial bounding box...")
    visualize(file_path, initial_bbox_str, initial_svg_path)
    svg_to_png_cli(initial_svg_path, initial_png_path, scale=4.0)
    
    print("\nRunning OpenCV Pre-Check for Boundaries...")
    margins, is_hard_cutoff = analyze_margins_opencv(initial_png_path)
    margin_str = f"Left: {margins['left']}px, Top: {margins['top']}px, Right: {margins['right']}px, Bottom: {margins['bottom']}px"
    print(f"Detected Margins: {margin_str}")
    
    if is_hard_cutoff:
        print(">> OpenCV detected a margin of 0! This is a mathematically guaranteed cut-off.")
        
    print(f"\nAnalyzing boundaries with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
    boundary_module = dspy.ChainOfThought(BoundaryCheckSignature)
    
    try:
        boundary_output = boundary_module(
            query=keyword,
            opencv_margins=margin_str,
            image_region=dspy.Image.from_file(initial_png_path)
        )
        assessment = boundary_output.boundary_assessment
        print("\n=== Boundary Assessment Results ===")
        print(f"Is Complete: {assessment.is_complete}")
        print(f"Cut Off Edges: {assessment.cut_off_edges}")
        print(f"Reasoning: {boundary_output.reasoning}")
        print("===================================")
        
        if assessment.is_complete or not assessment.cut_off_edges:
            print("\nThe drawing appears to be complete. No padding needed.")
            return
            
        cut_off_edges_str = f"Cut Off Edges: {', '.join(assessment.cut_off_edges)}\nBoundary Analysis Reasoning: {boundary_output.reasoning}"
        
    except Exception as e:
        print(f"Error during Boundary Check execution: {e}")
        return
    
    print(f"\nAnalyzing padding requirements with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
    padding_module = dspy.ChainOfThought(PaddingDecisionSignature)
    # padding_module = dspy.ProgramOfThought(PaddingDecisionSignature)
    
    bbox_context = f"Initial BBox: {initial_bbox_str}\nCurrent BBox: {initial_bbox_str}"
    
    try:
        output = padding_module(
            cut_off_edges=cut_off_edges_str,
            image_region=dspy.Image.from_file(initial_png_path),
            bbox_context=bbox_context
        )
        decision = output.padding_decision
        
        print("\n=== Padding Multipliers ===")
        print(f"Left: {decision.left}")
        print(f"Right: {decision.right}")
        print(f"Top: {decision.top}")
        print(f"Bottom: {decision.bottom}")
        print(f"Reasoning: {output.reasoning}")
        print("===========================")
        
        # Apply padding
        w = bb_reg['max_x'] - bb_reg['min_x']
        h = bb_reg['max_y'] - bb_reg['min_y']
        
        new_min_x = bb_reg['min_x'] - (w * decision.left)
        new_max_x = bb_reg['max_x'] + (w * decision.right)
        new_min_y = bb_reg['min_y'] - (h * decision.bottom)
        new_max_y = bb_reg['max_y'] + (h * decision.top)
        
        new_bbox_str = f"{new_min_x:.2f},{new_min_y:.2f},{new_max_x:.2f},{new_max_y:.2f}"
        print(f"\nNew Expanded BBox: {new_bbox_str}")
        
        padded_svg_path = f"{base_out}_padded.svg"
        padded_png_path = f"{base_out}_padded.png"
        
        print("\nRasterizing padded bounding box...")
        visualize(file_path, new_bbox_str, padded_svg_path)
        svg_to_png_cli(padded_svg_path, padded_png_path, scale=4.0)
        print(f"Saved padded visualization to: {padded_png_path}")
        
    except Exception as e:
        print(f"Error during VLM execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="storage/json/2生产综合楼-平面图及楼电梯详图(FJ-MC-10~19 25~32)_t3(1).json")
    parser.add_argument("--keyword", default="六~八层平")
    args = parser.parse_args()
    
    estimate_and_apply_padding(args.file, args.keyword)
