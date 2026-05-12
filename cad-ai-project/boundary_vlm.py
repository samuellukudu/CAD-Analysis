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
from find_region import find_region_matches
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


def analyze_margins_opencv(image_path, output_annotated_path=None):
    """
    Uses OpenCV to calculate the exact pixel margin from the CAD content to the image edges.
    Optionally saves an annotated version of the image highlighting the bounding box, lines, and circles.
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
    
    # Optional Visualization
    if output_annotated_path:
        # 1. Draw Content Bounding Box (Cyan)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
        
        # 2. Line Detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=20)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if angle < 5 or angle > 175:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green for horizontal
                elif 85 < angle < 95:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red for vertical
                    
        # 3. Circle Detection
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=60)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                cv2.circle(img, (cx, cy), r, (255, 0, 0), 2) # Blue for circles
                
        cv2.imwrite(output_annotated_path, img)
    
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

def check_boundaries(file_path, keyword, output_dir="output_boundary_test"):
    print(f"\n{'='*80}")
    print(f"Boundary Check for '{keyword}' in {os.path.basename(file_path)}")
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
        
    bbox_str = f"{bb_reg['min_x']:.2f},{bb_reg['min_y']:.2f},{bb_reg['max_x']:.2f},{bb_reg['max_y']:.2f}"
    print(f"Initial BBox: {bbox_str}")
    
    safe_kw = "".join([c if c.isalnum() else "_" for c in keyword])
    base_out = os.path.join(output_dir, f"boundary_test_{safe_kw}")
    svg_path = f"{base_out}.svg"
    png_path = f"{base_out}.png"
    annotated_png_path = f"{base_out}_opencv_annotated.png"
    
    print("\nRasterizing bounding box...")
    visualize(file_path, bbox_str, svg_path)
    svg_to_png_cli(svg_path, png_path, scale=4.0)
    
    print("\nRunning OpenCV Pre-Check...")
    margins, is_hard_cutoff = analyze_margins_opencv(png_path, output_annotated_path=annotated_png_path)
    margin_str = f"Left: {margins['left']}px, Top: {margins['top']}px, Right: {margins['right']}px, Bottom: {margins['bottom']}px"
    print(f"Detected Margins: {margin_str}")
    print(f"Saved OpenCV annotated visualization to: {annotated_png_path}")
    
    if is_hard_cutoff:
        print(">> OpenCV detected a margin of 0! This is a mathematically guaranteed cut-off.")
    
    print(f"\nAnalyzing boundaries with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
    # Increase temperature to encourage better reasoning or try a different approach if it keeps hallucinating.
    boundary_module = dspy.ChainOfThought(BoundaryCheckSignature)
    
    try:
        output = boundary_module(
            query=keyword,
            opencv_margins=margin_str,
            image_region=dspy.Image.from_file(png_path)
        )
        edge_analysis = output.edge_analysis
        assessment = output.boundary_assessment
        
        print("\n=== Edge Analysis ===")
        print(f"Left: {edge_analysis.left}")
        print(f"Right: {edge_analysis.right}")
        print(f"Top: {edge_analysis.top}")
        print(f"Bottom: {edge_analysis.bottom}")
        
        print("\n=== Boundary Assessment Results ===")
        print(f"Is Complete: {assessment.is_complete}")
        print(f"Cut Off Edges: {assessment.cut_off_edges}")
        print(f"Reasoning: {output.reasoning}")
        print("===================================")
        # print(output)
        
    except Exception as e:
        print(f"Error during VLM execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="storage/json/2生产综合楼-平面图及楼电梯详图(FJ-MC-10~19 25~32)_t3(1).json")
    parser.add_argument("--keyword", default="六~八层平")
    args = parser.parse_args()
    
    check_boundaries(args.file, args.keyword)
