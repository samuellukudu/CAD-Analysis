import json
import os
import sys
import argparse
import subprocess
import math
from functools import lru_cache
from typing import Literal

from src.occupancy_grids import find_region_matches
from src.vector import find_vector_matches
from src.spatial_memory import MemoryEntry, save_memory, get_memories, get_all_parents, clear_memories


# Fix for DSPy readonly database error MUST BE BEFORE IMPORTING DSPY
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), ".dspy_cache")

import dspy
from dotenv import load_dotenv
load_dotenv()

# Setup DSPy text model (lightweight, needed for classify_query)
text_lm = dspy.LM(
    f"openai/{os.getenv('MODEL', 'qwen3.5-flash')}", 
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("BASE_URL")
)

dspy.configure(lm=text_lm)

# VLM-heavy dependencies are deferred until actually needed (--vlm flag).
# This avoids initialising the vision model connection and importing
# rendering / VLM-signature modules on every invocation.
_vlm_ready = False
vision_lm = None
BoundaryCheckSignature = None
PaddingDecisionSignature = None
CroppingDecisionSignature = None
analyze_margins_opencv = None
svg_to_png_cli = None

def _init_vlm():
    """Lazily import VLM signatures, rendering helpers, and the vision LM."""
    global _vlm_ready, vision_lm
    global BoundaryCheckSignature, PaddingDecisionSignature, CroppingDecisionSignature
    global analyze_margins_opencv, svg_to_png_cli
    if _vlm_ready:
        return
    from src.padding_vlm import (
        BoundaryCheckSignature as _BCS,
        PaddingDecisionSignature as _PDS,
        analyze_margins_opencv as _amo,
    )
    from src.cropping_vlm import CroppingDecisionSignature as _CDS
    from src.svg2image import svg_to_png_cli as _s2p
    BoundaryCheckSignature = _BCS
    PaddingDecisionSignature = _PDS
    CroppingDecisionSignature = _CDS
    analyze_margins_opencv = _amo
    svg_to_png_cli = _s2p
    vision_lm = dspy.LM(
        f"openai/{os.getenv('VISION_MODEL', 'gpt-4o')}",
        api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
    _vlm_ready = True

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
    电梯 (elevator), 楼梯 (stairs), 卫生间|男卫|女卫|WC|厕所 (toilet), 厨房 (kitchen),
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
    target_keyword: str = dspy.OutputField(desc="The actual object or region to find. MUST be translated to the language of the CAD drawing (e.g. Chinese). For synonyms, use regex pipe WITHOUT spaces e.g. '卫生间|男卫|女卫|WC|厕所'.")
    reasoning: str = dspy.OutputField(desc="Reasoning for the classification.")
    layout: str = dspy.OutputField(desc="The parent layout scope, if any (e.g. '一层平面图' or '1st floor'). Leave empty or 'none' if not specified.")

@lru_cache(maxsize=1)
def _get_query_classifier():
    return dspy.ChainOfThought(QueryClassificationSignature)

@lru_cache(maxsize=256)
def _classify_query_cached(normalized_query: str) -> dict:
    classifier = _get_query_classifier()

    try:
        result = classifier(query=normalized_query)
        classification = result.classification.strip().lower().strip("'\"")
        if classification not in ["general drawing", "sub-region"]:
            classification = "sub-region" if "sub-region" in classification else "general drawing"

        layout = result.layout.strip().lower()
        if layout in ["none", "empty", "", "n/a", "null"]:
            layout = None

        return {
            "classification": classification,
            "target_keyword": result.target_keyword.strip(),
            "layout": layout,
            "reasoning": result.reasoning
        }
    except Exception as e:
        print(f"Error during classification: {e}")
        return {"classification": "unknown", "target_keyword": normalized_query, "layout": None, "reasoning": str(e)}

def classify_query(query: str) -> dict:
    """
    Uses an LLM via DSPy to classify the query and extract intent.
    Returns a dictionary with classification, target_keyword, layout, and reasoning.
    """
    normalized_query = (query or "").strip()
    return dict(_classify_query_cached(normalized_query))

@lru_cache(maxsize=1)
def _get_boundary_module():
    if BoundaryCheckSignature is None:
        raise RuntimeError("VLM boundary signature not initialized.")
    return dspy.ChainOfThought(BoundaryCheckSignature)

@lru_cache(maxsize=1)
def _get_padding_module():
    if PaddingDecisionSignature is None:
        raise RuntimeError("VLM padding signature not initialized.")
    return dspy.ChainOfThought(PaddingDecisionSignature)

@lru_cache(maxsize=1)
def _get_cropping_module():
    if CroppingDecisionSignature is None:
        raise RuntimeError("VLM cropping signature not initialized.")
    return dspy.ChainOfThought(CroppingDecisionSignature)

def format_bbox_arg(bb):
    return f"{bb['min_x']:.2f},{bb['min_y']:.2f},{bb['max_x']:.2f},{bb['max_y']:.2f}"

def visualize(file_path, bbox_str, output_path):
    cmd = [
        sys.executable,
        "-m",
        "src.visualize_cad",
        file_path,
        "--bbox", bbox_str,
        "--output", output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def is_valid_bbox(bb):
    return bb and bb['max_x'] > bb['min_x'] and bb['max_y'] > bb['min_y']

def get_numeric(entry, key, default=0.0):
    if not entry:
        return default
    try:
        return float(entry.get(key, default))
    except (TypeError, ValueError):
        return default

def bboxes_overlap(b1, b2):
    return (max(b1['min_x'], b2['min_x']) <= min(b1['max_x'], b2['max_x']) and
            max(b1['min_y'], b2['min_y']) <= min(b1['max_y'], b2['max_y']))

def merge_two_bboxes(b1, b2):
    return {
        'min_x': min(b1['min_x'], b2['min_x']),
        'min_y': min(b1['min_y'], b2['min_y']),
        'max_x': max(b1['max_x'], b2['max_x']),
        'max_y': max(b1['max_y'], b2['max_y'])
    }

def bbox_area(bb):
    if not is_valid_bbox(bb):
        return 0.0
    return float((bb['max_x'] - bb['min_x']) * (bb['max_y'] - bb['min_y']))

def bbox_contains(outer, inner):
    if not is_valid_bbox(outer) or not is_valid_bbox(inner):
        return False
    return (
        outer['min_x'] <= inner['min_x']
        and outer['min_y'] <= inner['min_y']
        and outer['max_x'] >= inner['max_x']
        and outer['max_y'] >= inner['max_y']
    )

def bbox_intersection_metrics(b1, b2):
    if not is_valid_bbox(b1) or not is_valid_bbox(b2):
        return {
            "intersection_area": 0.0,
            "overlap_width": 0.0,
            "overlap_height": 0.0,
            "smaller_area_ratio": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
        }

    overlap_width = max(0.0, min(b1['max_x'], b2['max_x']) - max(b1['min_x'], b2['min_x']))
    overlap_height = max(0.0, min(b1['max_y'], b2['max_y']) - max(b1['min_y'], b2['min_y']))
    intersection_area = float(overlap_width * overlap_height)

    w1 = float(b1['max_x'] - b1['min_x'])
    h1 = float(b1['max_y'] - b1['min_y'])
    w2 = float(b2['max_x'] - b2['min_x'])
    h2 = float(b2['max_y'] - b2['min_y'])
    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    smaller_area = min(area1, area2) if area1 > 0.0 and area2 > 0.0 else 0.0
    min_width = min(w1, w2) if w1 > 0.0 and w2 > 0.0 else 0.0
    min_height = min(h1, h2) if h1 > 0.0 and h2 > 0.0 else 0.0

    return {
        "intersection_area": intersection_area,
        "overlap_width": overlap_width,
        "overlap_height": overlap_height,
        "smaller_area_ratio": 0.0 if smaller_area == 0.0 else intersection_area / smaller_area,
        "width_ratio": 0.0 if min_width == 0.0 else overlap_width / min_width,
        "height_ratio": 0.0 if min_height == 0.0 else overlap_height / min_height,
    }

def bboxes_overlap_significantly(b1, b2, min_area_ratio=0.05, min_axis_ratio=0.15):
    metrics = bbox_intersection_metrics(b1, b2)
    return (
        metrics["intersection_area"] > 0.0
        and metrics["smaller_area_ratio"] >= float(min_area_ratio)
        and metrics["width_ratio"] >= float(min_axis_ratio)
        and metrics["height_ratio"] >= float(min_axis_ratio)
    )

def merge_overlapping_bboxes(bboxes):
    if not bboxes:
        return []
    
    merged = []
    for bbox in bboxes:
        new_bbox = bbox
        merged_any = True
        
        # Keep trying to merge new_bbox with existing ones until no more overlaps are found
        while merged_any:
            merged_any = False
            overlapping_indices = []
            for i, m_bbox in enumerate(merged):
                if bboxes_overlap_significantly(new_bbox, m_bbox):
                    overlapping_indices.append(i)
            
            if overlapping_indices:
                merged_any = True
                for i in sorted(overlapping_indices, reverse=True):
                    new_bbox = merge_two_bboxes(new_bbox, merged.pop(i))
                    
        merged.append(new_bbox)
            
    return merged

def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    p = max(0.0, min(100.0, float(p)))
    sorted_vals = sorted(float(v) for v in values)
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight

def detect_high_outlier_indices(values):
    if len(values) < 3:
        return set()
    positive_values = [float(v) for v in values if float(v) > 0.0]
    if len(positive_values) < 3:
        return set()
    log_values = [math.log10(max(float(v), 1e-12)) for v in values]
    median_log = percentile(log_values, 50.0)
    deviations = [abs(v - median_log) for v in log_values]
    mad = percentile(deviations, 50.0)
    median_value = percentile(values, 50.0)
    outlier_indices = set()
    for i, value in enumerate(values):
        ratio = float("inf") if median_value == 0 else float(value) / float(median_value)
        robust_z = 0.0
        if mad and mad > 1e-9:
            robust_z = 0.6745 * abs(log_values[i] - median_log) / mad
        if robust_z >= 3.5 or ratio >= 8.0:
            outlier_indices.add(i)
    return outlier_indices

def filter_dominant_outlier_matches(valid_matches):
    ordered_keys = sorted(valid_matches.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    match_infos = []
    for key in ordered_keys:
        entry = valid_matches[key]
        bbox = entry.get('candidates_mean_bbox_xyxy')
        area = get_numeric(entry, "best_area", 0.0)
        if area <= 0.0:
            area = bbox_area(bbox)
        area_pct = get_numeric(entry, "best_area_pct_of_drawing", 0.0)
        if area_pct <= 0.0:
            drawing_area = get_numeric(entry, "drawing_area", 0.0)
            if drawing_area > 0.0:
                area_pct = (area / drawing_area) * 100.0
        match_infos.append({
            "key": key,
            "entry": entry,
            "bbox": bbox,
            "area": area,
            "area_pct": area_pct,
        })

    outlier_indices = detect_high_outlier_indices([info["area_pct"] for info in match_infos])
    if not outlier_indices:
        return valid_matches

    filtered_matches = {}
    removed_keys = []
    for i, info in enumerate(match_infos):
        contains_count = 0
        for j, other in enumerate(match_infos):
            if i == j:
                continue
            if bbox_contains(info["bbox"], other["bbox"]):
                contains_count += 1
        if i in outlier_indices and contains_count > 0:
            removed_keys.append(info["key"])
            print(
                f"  -> Ignoring Match {info['key']} as dominant outlier "
                f"(bbox_area_pct={info['area_pct']:.4f}%, contains={contains_count} other match(es))"
            )
            continue
        filtered_matches[info["key"]] = info["entry"]

    return filtered_matches if filtered_matches else valid_matches

def is_valid_match(entry):
    if not entry:
        return False
    if not is_valid_bbox(entry.get('candidates_mean_bbox_xyxy')):
        return False
        
    return True

def get_overlapping_parents(sub_bbox, parent_bboxes):
    """Return all parent bounding boxes that intersect with the sub_bbox."""
    if not parent_bboxes:
        return []
    
    overlapping = []
    for p_bbox in parent_bboxes:
        if bboxes_overlap(sub_bbox, p_bbox):
            overlapping.append(p_bbox)
            
    return overlapping

def find_entity_bbox(file_path, keyword, classification, data, parent_bboxes=None, threshold=100, min_ar=0.25, max_ar=5.0):
    if classification == 'general drawing':
        print(f"Using find_region.py for '{keyword}'...")
        res_region = find_region_matches(keyword, data=data, bbox_direct=True, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
    else:
        print(f"Using vector.py for '{keyword}'...")
        res_region = find_vector_matches(keyword, data=data, bbox_direct=True, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
        
    matches = res_region.get('matches', {})
    valid_matches = {}
    
    for k, entry in matches.items():
        if is_valid_match(entry):
            valid_matches[k] = entry
            
    # Spatial Filtering if parent_bboxes is provided
    if parent_bboxes and isinstance(parent_bboxes, list) and len(parent_bboxes) > 0:
        print(f"\nApplying spatial filtering: Sub-regions must intersect with ANY of {len(parent_bboxes)} parent region(s).")
        filtered_by_spatial = {}
        for k, entry in valid_matches.items():
            bb = entry.get('candidates_mean_bbox_xyxy')
            if bb and get_overlapping_parents(bb, parent_bboxes):
                filtered_by_spatial[k] = entry
            else:
                print(f"  -> Ignoring Match {k} as it is outside ALL parent bounding boxes.")
        valid_matches = filtered_by_spatial

    bb_regs = []
    matched_text = keyword
    if valid_matches:
        print(f"\nFound {len(valid_matches)} valid match(es) for '{keyword}':")
        for k in sorted(valid_matches.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            entry = valid_matches[k]
            union_pct = get_numeric(entry, "union_area_pct_of_drawing", 0.0)
            bb = entry.get('candidates_mean_bbox_xyxy', {})
            print(f"  - Match {k}: union_area_pct={union_pct:.4f}%, bbox=[{bb.get('min_x', 0):.2f}, {bb.get('min_y', 0):.2f}, {bb.get('max_x', 0):.2f}, {bb.get('max_y', 0):.2f}]")
            
        if len(valid_matches) >= 3:
            valid_matches = filter_dominant_outlier_matches(valid_matches)
            
        if valid_matches:
            print(f"\nMerging overlapping bounding boxes among valid matches for '{keyword}'...")
            valid_bboxes = [entry.get('candidates_mean_bbox_xyxy') for entry in valid_matches.values() if is_valid_bbox(entry.get('candidates_mean_bbox_xyxy'))]
            merged_bboxes = merge_overlapping_bboxes(valid_bboxes)
            
            print(f"Reduced {len(valid_bboxes)} valid match(es) into {len(merged_bboxes)} merged region(s):")
            for i, m_bbox in enumerate(merged_bboxes):
                print(f"  - Merged Region {i+1}: [{m_bbox['min_x']:.2f}, {m_bbox['min_y']:.2f}, {m_bbox['max_x']:.2f}, {m_bbox['max_y']:.2f}]")
            
            bb_regs = merged_bboxes
            if bb_regs:
                first_key = sorted(valid_matches.keys(), key=lambda x: int(x) if str(x).isdigit() else x)[0]
                matched_text = valid_matches[first_key].get('text', keyword)
                
    if bb_regs:
        try:
            # Clear previous entries for this exact query to avoid duplication on repeated runs
            clear_memories(os.path.abspath(file_path), keyword)
        except Exception as e:
            print(f"Warning: Failed to clear previous spatial memories: {e}")
            
        saved_count = 0
        for bb_reg in bb_regs:
            if is_valid_bbox(bb_reg):
                try:
                    # Determine parent association
                    parent_kw = None
                    p_min_x, p_min_y, p_max_x, p_max_y = None, None, None, None
                    if parent_bboxes:
                        overlaps = get_overlapping_parents(bb_reg, parent_bboxes)
                        # Find the first valid parent keyword from the overlaps
                        for p in overlaps:
                            if p.get('keyword'):
                                parent_kw = p.get('keyword')
                                p_min_x = p.get('min_x')
                                p_min_y = p.get('min_y')
                                p_max_x = p.get('max_x')
                                p_max_y = p.get('max_y')
                                break
                    
                    mem_entry = MemoryEntry(
                        file_path=os.path.abspath(file_path),
                        query_keyword=keyword,
                        matched_text=matched_text,
                        classification=classification,
                        min_x=bb_reg['min_x'],
                        min_y=bb_reg['min_y'],
                        max_x=bb_reg['max_x'],
                        max_y=bb_reg['max_y'],
                        parent_keyword=parent_kw,
                        parent_min_x=p_min_x,
                        parent_min_y=p_min_y,
                        parent_max_x=p_max_x,
                        parent_max_y=p_max_y
                    )
                    save_memory(mem_entry)
                    saved_count += 1
                except Exception as e:
                    print(f"Warning: Failed to save to spatial memory: {e}")
        if saved_count > 0:
            print(f"Saved {saved_count} merged region(s) for '{keyword}' to spatial memory.")
            
    return bb_regs

def process_and_refine(
    file_path,
    target_keyword,
    layout=None,
    output_dir="output_refined",
    threshold=100,
    min_ar=0.25,
    max_ar=5.0,
    keep_images=False,
    vlm=False,
):
    if vlm:
        _init_vlm()

    intent = classify_query(target_keyword)
    classification = intent["classification"]

    use_layout = bool(layout) and classification == 'sub-region'
    display_kw = f"{layout or ''} {target_keyword}".strip() if use_layout else target_keyword

    print(f"\n{'='*80}")
    print(f"Extracting BBox for '{display_kw}' in {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    if vlm and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Using target keyword: '{target_keyword}'")
    print(f"Classification for target: {classification}")
    if layout:
        if use_layout:
            print(f"Explicit Layout: {layout}")
        else:
            print(f"Ignoring layout '{layout}' because it only applies to sub-region targets.")

    print("Loading JSON data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parent_bboxes = []
    abs_file_path = os.path.abspath(file_path)
    
    if use_layout:
        print(f"\n--- Checking spatial memory for layout: '{layout}' ---")
        parent_mems = get_memories(abs_file_path, layout, "general drawing")
        if parent_mems:
            print(f"Memory Hit! Found {len(parent_mems)} parent region(s) for '{layout}' in DB.")
            for mem in parent_mems:
                parent_bboxes.append({
                    'min_x': mem.min_x,
                    'min_y': mem.min_y,
                    'max_x': mem.max_x,
                    'max_y': mem.max_y,
                    'keyword': layout
                })
        else:
            print(f"Memory Miss. Resolving layout '{layout}' first...")
            parent_bboxes_raw = find_entity_bbox(file_path, layout, "general drawing", data, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
            if parent_bboxes_raw:
                for pb in parent_bboxes_raw:
                    pb['keyword'] = layout
                    parent_bboxes.append(pb)
            else:
                print(f"Warning: Could not resolve layout '{layout}'. Proceeding without spatial filtering.")
    else:
        if classification == 'sub-region':
            print("\n--- No explicit layout provided. Fetching all known parents to validate sub-region... ---")
            all_parents = get_all_parents(abs_file_path)
            if all_parents:
                print(f"Found {len(all_parents)} known parent region(s) in memory.")
                for mem in all_parents:
                    parent_bboxes.append({
                        'min_x': mem.min_x,
                        'min_y': mem.min_y,
                        'max_x': mem.max_x,
                        'max_y': mem.max_y,
                        'keyword': mem.query_keyword
                    })
            else:
                print("Warning: No general drawings found in memory for this file. Sub-regions might not be strictly filtered.")

    print(f"\n--- Resolving target entity: '{target_keyword}' ---")
    target_bboxes = find_entity_bbox(file_path, target_keyword, classification, data, parent_bboxes, threshold=threshold, min_ar=min_ar, max_ar=max_ar)
    
    if not target_bboxes:
        print(f"Error: Could not extract bounding box(es) for '{target_keyword}'.")
        return None
        
    print(f"\nFound {len(target_bboxes)} final region(s) for processing.")
    
    final_bboxes = []

    if not vlm:
        for i, target_bbox in enumerate(target_bboxes):
            print(f"  Region {i+1}: [{target_bbox['min_x']:.2f}, {target_bbox['min_y']:.2f}, {target_bbox['max_x']:.2f}, {target_bbox['max_y']:.2f}]")
            final_bboxes.append([
                round(target_bbox['min_x'], 2),
                round(target_bbox['min_y'], 2),
                round(target_bbox['max_x'], 2),
                round(target_bbox['max_y'], 2),
            ])
        return final_bboxes

    for i, target_bbox in enumerate(target_bboxes):
        print(f"\n--- Processing Region {i+1}/{len(target_bboxes)} (VLM refinement) ---")
        print(f"BBox: [{target_bbox['min_x']:.2f}, {target_bbox['min_y']:.2f}, {target_bbox['max_x']:.2f}, {target_bbox['max_y']:.2f}]")
        
        safe_kw = "".join([c if c.isalnum() else "_" for c in display_kw])
        region_suffix = f"_region{i+1}" if len(target_bboxes) > 1 else ""
        base_out = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_{safe_kw}{region_suffix}")
        
        svg_initial = f"{base_out}_initial.svg"
        png_initial = f"{base_out}_initial.png"

        print(f"Rasterizing initial bounding box for region {i+1}...")
        visualize(file_path, format_bbox_arg(target_bbox), svg_initial)
        svg_to_png_cli(svg_initial, png_initial, scale=4.0)
        
        current_bbox = target_bbox
        current_png = png_initial
        
        with dspy.context(lm=vision_lm):
            if classification == 'general drawing':
                print("\nRunning OpenCV Pre-Check for Boundaries...")
                margins, is_hard_cutoff = analyze_margins_opencv(current_png)
                margin_str = f"Left: {margins['left']}px, Top: {margins['top']}px, Right: {margins['right']}px, Bottom: {margins['bottom']}px"
                print(f"Detected Margins: {margin_str}")
                
                if is_hard_cutoff:
                    print(">> OpenCV detected a margin of 0! This is a mathematically guaranteed cut-off.")
                    
                print(f"\nAnalyzing boundaries with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
                boundary_module = _get_boundary_module()
                
                try:
                    boundary_output = boundary_module(
                        query=target_keyword,
                        opencv_margins=margin_str,
                        image_region=dspy.Image.from_file(current_png)
                    )
                    assessment = boundary_output.boundary_assessment
                    print("\n=== Boundary Assessment Results ===")
                    print(f"Is Complete: {assessment.is_complete}")
                    print(f"Cut Off Edges: {assessment.cut_off_edges}")
                    print(f"Reasoning: {boundary_output.reasoning}")
                    print("===================================")
                    
                    if not assessment.is_complete and assessment.cut_off_edges:
                        cut_off_edges_str = f"Cut Off Edges: {', '.join(assessment.cut_off_edges)}\nBoundary Analysis Reasoning: {boundary_output.reasoning}"
                        
                        print(f"\nAnalyzing padding requirements with VLM...")
                        padding_module = _get_padding_module()
                        
                        bbox_context = f"Initial BBox: {format_bbox_arg(target_bbox)}\nCurrent BBox: {format_bbox_arg(current_bbox)}"
                        
                        output = padding_module(
                            cut_off_edges=cut_off_edges_str,
                            image_region=dspy.Image.from_file(current_png),
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
                        
                        w = current_bbox['max_x'] - current_bbox['min_x']
                        h = current_bbox['max_y'] - current_bbox['min_y']
                        
                        new_min_x = current_bbox['min_x'] - (w * decision.left)
                        new_max_x = current_bbox['max_x'] + (w * decision.right)
                        new_min_y = current_bbox['min_y'] - (h * decision.bottom)
                        new_max_y = current_bbox['max_y'] + (h * decision.top)
                        
                        current_bbox = {
                            'min_x': new_min_x, 'min_y': new_min_y,
                            'max_x': new_max_x, 'max_y': new_max_y
                        }
                        
                        print(f"\nNew Expanded BBox: {format_bbox_arg(current_bbox)}")
                        
                        svg_padded = f"{base_out}_padded.svg"
                        png_padded = f"{base_out}_padded.png"
                        
                        print("\nRasterizing padded bounding box...")
                        visualize(file_path, format_bbox_arg(current_bbox), svg_padded)
                        svg_to_png_cli(svg_padded, png_padded, scale=4.0)
                        current_png = png_padded
                        
                except Exception as e:
                    print(f"Error during Boundary Check or Padding VLM execution: {e}")
                    
            elif classification == 'sub-region':
                print(f"\nAnalyzing cropping requirements with VLM (Model: {os.getenv('VISION_MODEL', 'gpt-4o')})...")
                cropping_module = _get_cropping_module()
                
                try:
                    output = cropping_module(
                        query=target_keyword,
                        image_region=dspy.Image.from_file(current_png)
                    )
                    decision = output.cropping_decision
                    
                    print("\n=== Cropping Multipliers ===")
                    print(f"Left: {decision.left}")
                    print(f"Right: {decision.right}")
                    print(f"Top: {decision.top}")
                    print(f"Bottom: {decision.bottom}")
                    print(f"Reasoning: {output.reasoning}")
                    print("============================")
                    
                    w = current_bbox['max_x'] - current_bbox['min_x']
                    h = current_bbox['max_y'] - current_bbox['min_y']
                    
                    new_min_x = current_bbox['min_x'] + (w * decision.left)
                    new_max_x = current_bbox['max_x'] - (w * decision.right)
                    new_min_y = current_bbox['min_y'] + (h * decision.bottom)
                    new_max_y = current_bbox['max_y'] - (h * decision.top)
                    
                    if new_min_x < new_max_x and new_min_y < new_max_y:
                        current_bbox = {
                            'min_x': new_min_x, 'min_y': new_min_y,
                            'max_x': new_max_x, 'max_y': new_max_y
                        }
                        print(f"\nNew Cropped BBox: {format_bbox_arg(current_bbox)}")
                        
                        svg_cropped = f"{base_out}_cropped.svg"
                        png_cropped = f"{base_out}_cropped.png"
                        
                        print("\nRasterizing cropped bounding box...")
                        visualize(file_path, format_bbox_arg(current_bbox), svg_cropped)
                        svg_to_png_cli(svg_cropped, png_cropped, scale=4.0)
                        current_png = png_cropped
                    else:
                        print("Error: Cropping values are too large, resulting in invalid bounding box. Aborting crop.")
                        
                except Exception as e:
                    print(f"Error during Cropping VLM execution: {e}")

        print(f"Done! Final refined bbox: {format_bbox_arg(current_bbox)}\n")
        final_bboxes.append([
            round(current_bbox['min_x'], 2), 
            round(current_bbox['min_y'], 2), 
            round(current_bbox['max_x'], 2), 
            round(current_bbox['max_y'], 2)
        ])
        
    if not keep_images and os.path.exists(output_dir):
        import shutil
        print(f"Cleaning up image folder: {output_dir}")
        shutil.rmtree(output_dir)
        
    return final_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine CAD Bounding Boxes with VLM and Spatial Memory")
    parser.add_argument("--file", required=True, help="Path to the JSON CAD data file")
    
    parser.add_argument("--target_keyword", required=True, help="Explicit target entity to find (e.g., '卫生间')")
    parser.add_argument("--layout", help="Explicit parent layout/drawing scope (e.g., '一层平面图')")
    
    parser.add_argument("--outdir", default="output_refined", help="Directory to save the rasterized output")
    parser.add_argument("--threshold", type=float, default=100, help="Minimum bbox dimension threshold")
    parser.add_argument("--min_ar", type=float, default=0.25, help="Minimum aspect ratio for bbox filtering")
    parser.add_argument("--max_ar", type=float, default=5.0, help="Maximum aspect ratio for bbox filtering")
    parser.add_argument("--keep_images", action="store_true", help="Keep generated images after processing (for testing)")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM refinement (boundary/padding/cropping). Without this flag, returns geometric bboxes only.")
    args = parser.parse_args()
    
    bboxes = process_and_refine(
        args.file, 
        target_keyword=args.target_keyword, 
        layout=args.layout,
        output_dir=args.outdir,
        threshold=args.threshold,
        min_ar=args.min_ar,
        max_ar=args.max_ar,
        keep_images=args.keep_images,
        vlm=args.vlm,
    )
    
    print("\n--- Final Refined Bounding Boxes ---")
    print(json.dumps(bboxes, indent=2))
