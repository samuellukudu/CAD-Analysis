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
from src.svg2image import svg_to_png_cli
from src.padding_vlm import analyze_margins_opencv, BoundaryCheckSignature, PaddingDecisionSignature
from src.cropping_vlm import CroppingDecisionSignature
from dotenv import load_dotenv

load_dotenv()

# Setup DSPy Model
lm = dspy.LM(
    f"openai/{os.getenv('VISION_MODEL', 'gpt-4o')}", 
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("BASE_URL")
)
dspy.configure(lm=lm)

class SubLayout(BaseModel):
    description: str = Field(..., description="Description of the sub-layout (e.g. 'left layout', 'top right layout')")
    x_min_frac: float = Field(..., description="Left edge fraction (0.0 is left edge of image, 1.0 is right edge)")
    x_max_frac: float = Field(..., description="Right edge fraction (0.0 is left edge, 1.0 is right edge)")
    y_min_frac: float = Field(..., description="Top edge fraction (0.0 is top edge of image, 1.0 is bottom edge)")
    y_max_frac: float = Field(..., description="Bottom edge fraction (0.0 is top edge, 1.0 is bottom edge)")

class LayoutSplitDecision(BaseModel):
    has_multiple_layouts: bool = Field(..., description="True if there are multiple independent floor plans or layouts side-by-side or stacked")
    layouts: list[SubLayout] = Field(..., description="List of the distinct layouts found in the image. Empty if has_multiple_layouts is False.")

class LayoutSplitDecisionSignature(dspy.Signature):
    """
    Analyze the provided CAD region image.
    The bounding box might contain MULTIPLE independent floor plans, layouts, or drawings 
    placed side-by-side or stacked.
    
    Determine if there are multiple distinct layouts.
    If so, provide the relative bounding boxes for each individual layout as fractions 
    of the total image width and height.
    
    For example:
    - If there are two identical layouts side-by-side, split them around x=0.5.
    - Keep crops precise to only include each specific layout and its immediate annotations.
    """
    query: str = dspy.InputField(desc="The target drawing description")
    image_region: dspy.Image = dspy.InputField(desc="Image view of the current bounding box")
    
    split_decision: LayoutSplitDecision = dspy.OutputField()

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

def process_bbox_for_split(file_path, keyword, initial_bbox_str, output_dir="output_multi_layout"):
    print(f"\n{'='*80}")
    print(f"Multi-Layout Split & Refine Estimation for '{keyword}' in {os.path.basename(file_path)}")
    print(f"Initial BBox: {initial_bbox_str}")
    print(f"{'='*80}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    safe_kw = "".join([c if c.isalnum() else "_" for c in keyword])
    base_out = os.path.join(output_dir, f"multi_layout_{safe_kw}")
    
    # Parse initial bbox
    parts = initial_bbox_str.split(',')
    if len(parts) != 4:
        print("Invalid bounding box format. Expected min_x,min_y,max_x,max_y")
        return
        
    min_x, min_y, max_x, max_y = map(float, parts)
    w_initial = max_x - min_x
    h_initial = max_y - min_y
    
    # -------------------------------------------------------------------------
    # STEP 1: LAYOUT SPLITTING
    # -------------------------------------------------------------------------
    initial_svg_path = f"{base_out}_initial.svg"
    initial_png_path = f"{base_out}_initial.png"
    
    print("\nRasterizing initial bounding box...")
    visualize(file_path, initial_bbox_str, initial_svg_path)
    svg_to_png_cli(initial_svg_path, initial_png_path, scale=4.0)
    
    print(f"\nAnalyzing multiple layouts with VLM on {os.path.basename(initial_png_path)}...")
    split_module = dspy.ChainOfThought(LayoutSplitDecisionSignature)
    
    layouts_to_process = []
    
    try:
        output = split_module(
            query=keyword,
            image_region=dspy.Image.from_file(initial_png_path)
        )
        decision = output.split_decision
        
        print("\n=== Split Decision ===")
        print(f"Has Multiple Layouts: {decision.has_multiple_layouts}")
        print(f"Reasoning: {output.reasoning}")
        print("======================")
        
        if decision.has_multiple_layouts and decision.layouts:
            for idx, layout in enumerate(decision.layouts):
                print(f"\n--- Layout {idx+1}: {layout.description} ---")
                
                sub_min_x = min_x + (w_initial * layout.x_min_frac)
                sub_max_x = min_x + (w_initial * layout.x_max_frac)
                sub_max_y = max_y - (h_initial * layout.y_min_frac)
                sub_min_y = max_y - (h_initial * layout.y_max_frac)
                
                sub_bbox_str = f"{sub_min_x:.2f},{sub_min_y:.2f},{sub_max_x:.2f},{sub_max_y:.2f}"
                layouts_to_process.append({
                    "id": f"sub_{idx+1}",
                    "bbox": sub_bbox_str,
                    "min_x": sub_min_x, "min_y": sub_min_y, "max_x": sub_max_x, "max_y": sub_max_y
                })
        else:
            print("No multiple layouts detected. Proceeding with the initial layout.")
            layouts_to_process.append({
                "id": "main",
                "bbox": initial_bbox_str,
                "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y
            })
            
    except Exception as e:
        print(f"Error during Layout Splitting execution: {e}")
        print("Fallback to processing the initial layout.")
        layouts_to_process.append({
            "id": "main",
            "bbox": initial_bbox_str,
            "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y
        })
        
    # -------------------------------------------------------------------------
    # STEP 2: PER-LAYOUT PADDING OR CROPPING
    # -------------------------------------------------------------------------
    boundary_module = dspy.ChainOfThought(BoundaryCheckSignature)
    padding_module = dspy.ChainOfThought(PaddingDecisionSignature)
    cropping_module = dspy.ChainOfThought(CroppingDecisionSignature)
    
    for layout_data in layouts_to_process:
        l_id = layout_data["id"]
        l_bbox = layout_data["bbox"]
        l_min_x, l_min_y, l_max_x, l_max_y = layout_data["min_x"], layout_data["min_y"], layout_data["max_x"], layout_data["max_y"]
        l_w = l_max_x - l_min_x
        l_h = l_max_y - l_min_y
        
        print(f"\n{'='*60}")
        print(f"Refining Layout: {l_id}")
        print(f"BBox: {l_bbox}")
        print(f"{'='*60}")
        
        l_svg_path = f"{base_out}_{l_id}_raw.svg"
        l_png_path = f"{base_out}_{l_id}_raw.png"
        
        print(f"Rasterizing raw layout {l_id}...")
        visualize(file_path, l_bbox, l_svg_path)
        svg_to_png_cli(l_svg_path, l_png_path, scale=4.0)
        
        print("Running OpenCV Pre-Check for Boundaries...")
        margins, is_hard_cutoff = analyze_margins_opencv(l_png_path)
        margin_str = f"Left: {margins['left']}px, Top: {margins['top']}px, Right: {margins['right']}px, Bottom: {margins['bottom']}px"
        print(f"Detected Margins: {margin_str}")
        
        print(f"Analyzing boundaries with VLM...")
        try:
            boundary_output = boundary_module(
                query=keyword,
                opencv_margins=margin_str,
                image_region=dspy.Image.from_file(l_png_path)
            )
            assessment = boundary_output.boundary_assessment
            print("\n=== Boundary Assessment Results ===")
            print(f"Is Complete: {assessment.is_complete}")
            print(f"Cut Off Edges: {assessment.cut_off_edges}")
            print(f"Reasoning: {boundary_output.reasoning}")
            print("===================================")
            
            if not assessment.is_complete and assessment.cut_off_edges:
                # Needs Padding
                cut_off_edges_str = f"Cut Off Edges: {', '.join(assessment.cut_off_edges)}\nBoundary Analysis Reasoning: {boundary_output.reasoning}"
                
                print(f"\nLayout {l_id} is cut off. Analyzing padding requirements...")
                bbox_context = f"Initial BBox: {l_bbox}\nCurrent BBox: {l_bbox}"
                
                pad_output = padding_module(
                    cut_off_edges=cut_off_edges_str,
                    image_region=dspy.Image.from_file(l_png_path),
                    bbox_context=bbox_context
                )
                pad_decision = pad_output.padding_decision
                
                print("\n=== Padding Multipliers ===")
                print(f"Left: {pad_decision.left}, Right: {pad_decision.right}, Top: {pad_decision.top}, Bottom: {pad_decision.bottom}")
                print(f"Reasoning: {pad_output.reasoning}")
                print("===========================")
                
                new_min_x = l_min_x - (l_w * pad_decision.left)
                new_max_x = l_max_x + (l_w * pad_decision.right)
                new_min_y = l_min_y - (l_h * pad_decision.bottom)
                new_max_y = l_max_y + (l_h * pad_decision.top)
                
                refined_bbox_str = f"{new_min_x:.2f},{new_min_y:.2f},{new_max_x:.2f},{new_max_y:.2f}"
                print(f"\nNew Padded BBox for {l_id}: {refined_bbox_str}")
                
                refined_svg_path = f"{base_out}_{l_id}_padded.svg"
                refined_png_path = f"{base_out}_{l_id}_padded.png"
                
                print(f"Rasterizing padded layout {l_id}...")
                visualize(file_path, refined_bbox_str, refined_svg_path)
                svg_to_png_cli(refined_svg_path, refined_png_path, scale=4.0)
                print(f"Saved padded layout to: {refined_png_path}")
                
            else:
                # Complete, check if it needs cropping
                print(f"\nLayout {l_id} appears complete. Analyzing cropping requirements...")
                crop_output = cropping_module(
                    query=keyword,
                    image_region=dspy.Image.from_file(l_png_path)
                )
                crop_decision = crop_output.cropping_decision
                
                print("\n=== Cropping Multipliers ===")
                print(f"Left: {crop_decision.left}, Right: {crop_decision.right}, Top: {crop_decision.top}, Bottom: {crop_decision.bottom}")
                print(f"Reasoning: {crop_output.reasoning}")
                print("============================")
                
                if any(v > 0 for v in [crop_decision.left, crop_decision.right, crop_decision.top, crop_decision.bottom]):
                    new_min_x = l_min_x + (l_w * crop_decision.left)
                    new_max_x = l_max_x - (l_w * crop_decision.right)
                    new_min_y = l_min_y + (l_h * crop_decision.bottom)
                    new_max_y = l_max_y - (l_h * crop_decision.top)
                    
                    if new_min_x >= new_max_x or new_min_y >= new_max_y:
                        print(f"Error: Cropping values for {l_id} result in invalid bounding box. Keeping raw layout.")
                    else:
                        refined_bbox_str = f"{new_min_x:.2f},{new_min_y:.2f},{new_max_x:.2f},{new_max_y:.2f}"
                        print(f"\nNew Cropped BBox for {l_id}: {refined_bbox_str}")
                        
                        refined_svg_path = f"{base_out}_{l_id}_cropped.svg"
                        refined_png_path = f"{base_out}_{l_id}_cropped.png"
                        
                        print(f"Rasterizing cropped layout {l_id}...")
                        visualize(file_path, refined_bbox_str, refined_svg_path)
                        svg_to_png_cli(refined_svg_path, refined_png_path, scale=4.0)
                        print(f"Saved cropped layout to: {refined_png_path}")
                else:
                    print(f"No cropping needed for {l_id}. The layout is final.")
                    
        except Exception as e:
            print(f"Error during Refinement execution for {l_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="storage/json/2生产综合楼-平面图及楼电梯详图(FJ-MC-10~19 25~32)_t3(1).json")
    parser.add_argument("--keyword", default="一层平面图")
    parser.add_argument("--bbox", required=True, help="Bounding box string min_x,min_y,max_x,max_y")
    args = parser.parse_args()
    
    process_bbox_for_split(args.file, args.keyword, args.bbox)
