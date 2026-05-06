import os
# Set Numba threading layer to 'omp' to avoid concurrency crashes
# Must be set before importing libraries that use Numba
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import io
import base64
import logging
import sys
from time import perf_counter
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import dspy
from tqdm import tqdm
from src.utils import create_layout_dataframe
from src.image2embeddings import create_embedding_index
from src.embeddings2clusters import cluster_embeddings
from src.utils import get_segmentation_crops
from src.utils import pdf_pages_to_base64_pngs, base64_to_pillow
from src.utils import coverage_ratio
from src.retrieve_pages import retrieve_pages_with_metadata
import pandas as pd
from enum import Enum
from typing import List, Optional, Dict, Any, Generator, Union
from collections import defaultdict
import json
from pydantic import BaseModel, Field, ConfigDict

# Lazy load building codes
_BUILDING_CODES_B64S = None

def search_building_codes(query: str, top_k: int = 5, top_p: float = 0.95) -> List[dspy.Image]:
    """
    Retrieve the most relevant building code pages as images based on the query.
    Uses the global index to find pages from uploaded PDFs and the standard building code.
    """
    results = retrieve_pages_with_metadata(query=query, top_k=top_k, top_p=top_p)
    
    images = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_pdfs_dir = os.path.join(project_root, "storage", "pdfs")
    
    for res in results:
        pdf_id = res.get("pdf_id")
        full_path = res.get("full_path")
        # 0-indexed page number for extraction (metadata is usually 1-indexed)
        page_idx = res.get("page_number", 1) - 1 
        
        pdf_path = None
        
        # 1. Check storage by ID
        if pdf_id:
            candidate = os.path.join(storage_pdfs_dir, f"{pdf_id}.pdf")
            if os.path.exists(candidate):
                pdf_path = candidate
        
        # 2. Check full path from metadata
        if not pdf_path and full_path and os.path.exists(full_path):
            pdf_path = full_path
            
        if pdf_path:
            try:
                # Fetch just this single page
                b64s = pdf_pages_to_base64_pngs(pdf_path, [page_idx])
                if b64s:
                    images.append(dspy.Image.from_PIL(base64_to_pillow(b64s[0])))
            except Exception as e:
                print(f"Error fetching page {page_idx} from {pdf_path}: {e}")
                continue
                
    return images

class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("src.dspy_agent")
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = _TqdmLoggingHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)

        log_file = os.getenv("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(file_handler)

    return logger


def _summarize_defects(defects: List["Defect"]) -> Dict[str, Any]:
    by_severity: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    for d in defects:
        by_severity[d.severity.value if hasattr(d.severity, "value") else str(d.severity)] += 1
        by_type[d.type] += 1
    return {
        "count": len(defects),
        "by_severity": dict(sorted(by_severity.items(), key=lambda x: x[0])),
        "by_type": dict(sorted(by_type.items(), key=lambda x: (-x[1], x[0]))),
    }


def _format_duration_s(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m{rem:04.1f}s"


def _env_or_raise(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def _load_resized_image(image_path: str, max_dimension: int = 4096) -> Image.Image:
    """
    Load an image from disk and resize it if it exceeds max_dimension.
    Helps prevent base64 encoding issues with massive images.
    """
    # Increase PIL limit to handle large CAD drawings
    Image.MAX_IMAGE_PIXELS = None
    
    img = Image.open(image_path)
    
    # Check if resize is needed
    if max(img.width, img.height) > max_dimension:
        ratio = max_dimension / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
    return img

load_dotenv()

class CADAnalyzer(dspy.Signature):
    """
    Analyze a CAD drawing image and produce a detailed, technically accurate textual description 
    based strictly on visible content. Use exact labels, terms, dimensions, and annotations as they 
    appear in the drawing. Do not translate any text, do not paraphrase labels, and do not add or 
    assume details that are not clearly visible. Use only what is observable in the image. Do not 
    infer or estimate coordinates, measurements, or unseen context. If describing locations, use 
    descriptive position based on the image (e.g., "upper left near label X") rather than numeric coordinates.

    If user_query is provided, focus the description on aspects relevant to the query (e.g., a specific
    system, room, equipment, or compliance topic). The narrative language of the response must match
    the language of user_query, while still keeping all drawing labels exactly as shown in the image.

    Cover the following in a logical order:
    - Drawing type and apparent purpose
    - Title block information (title, drawing number, project, dates, approvals, etc.)
    - Scale and units
    - Main disciplines or systems shown
    - Key areas, zones, rooms, or equipment groups with their exact labels
    - Major elements (walls, columns, beams, equipment, piping/ducts, symbols, etc.)
    - Notable dimensions, annotations, callouts, legends, and references
    - Revision history, stamps, or approval marks
    - Language(s) used in text and labels

    Write in clear, professional engineering language. Be specific, exhaustive, and concise.
    """

    user_query: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional user query to guide/focus the analysis. If provided, the narrative language of the "
             "response must match the language of this query, while keeping all drawing labels exactly as shown."
    )

    image: dspy.Image = dspy.InputField(
        desc="Full CAD drawing image or high-resolution crop containing technical symbols, "
             "dimensions, annotations, legends, and text in any language."
    )

    response: str = dspy.OutputField(
        desc="Structured textual description of all visible content in the CAD drawing. "
             "Use exact labels and terms as shown. No translations. No invented details."
    )

class LayoutAnalyzer(dspy.Signature):
    """
    Analyze a specific layout extracted from a CAD drawing and produce a detailed, technically accurate 
    textual description based strictly on visible content in the layout_image. Use the provided full 
    drawing description and full_image only for global context (e.g., overall project terminology, 
    orientation, or cross-referencing edge elements). Do not translate any text, do not paraphrase labels, 
    and do not add or assume details that are not clearly visible. Use only what is observable in the image. 
    Do not infer or estimate coordinates, measurements, or unseen context. If describing locations, use 
    descriptive position based on the image (e.g., "lower right adjacent to label Y") rather than numeric coordinates.

    If user_query is provided, focus the description on aspects relevant to the query (e.g., a specific
    system, room, equipment, or compliance topic). The narrative language of the response must match
    the language of user_query, while still keeping all layout labels exactly as shown in the image.

    Cover the following in a logical order:
    - Layout type and apparent purpose (e.g., floor plan, foundation plan, piping layout, section view)
    - Title, label, scale, and orientation specific to this layout
    - Primary discipline or focus
    - Key zones, rooms, areas, or equipment groups with their exact labels
    - Major visible elements and systems (walls, columns, foundations, piping/ducts, electrical routes, etc.)
    - Notable symbols, hatches, line types, and any visible legend explanations
    - Critical dimensions, elevations, annotations, and callouts unique to this layout
    - Relationships to adjacent areas or references to other drawings/layouts
    - Revision marks, notes, or special instructions specific to this layout
    - Language(s) used in text and labels

    Write in clear, professional engineering language. Be specific, exhaustive, and concise. 
    Prioritize content visible in the layout_image.
    """

    user_query: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional user query to guide/focus the layout analysis. If provided, the narrative language of the "
             "response must match the language of this query, while keeping all labels exactly as shown."
    )

    description: str = dspy.InputField(
        desc="Comprehensive textual description of the entire original CAD drawing for overall "
             "project context and terminology."
    )

    full_image: dspy.Image = dspy.InputField(
        desc="Complete original CAD drawing image for global spatial context and cross-referencing."
    )

    layout_image: dspy.Image = dspy.InputField(
        desc="Cropped or extracted image containing only the target layout. "
             "Primary focus: analyze all visible labels, dimensions, symbols, lines, hatches, "
             "and annotations in detail."
    )

    response: str = dspy.OutputField(
        desc="Structured textual description of the specific layout in layout_image. "
             "Use exact labels and terms as shown. No translations. No invented details."
    )

class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Defect(BaseModel):
    id: int = Field(..., description="Unique identifier for the defect (sequential integer across the entire report)")
    type: str = Field(..., description="Category of the defect, e.g., 'Unit Mismatch', 'Improper Layer Usage'")
    description: str = Field(..., description="Detailed explanation of the issue")
    location: Optional[str] = Field(
        None,
        description="Where in the drawing the issue occurs, using only visible context (e.g., 'Layer: A-WALL', 'View: Section A-A', 'upper left near label A-101')"
    )
    severity: Severity = Field(..., description="Impact level: Low, Medium, or High")
    how_to_fix: str = Field(..., description="Recommended steps to resolve the defect")

    # Provenance fields for traceability
    layout_id: str = Field(..., description="Identifier of the layout this defect belongs to")
    chunk_id: Optional[str] = Field(None, description="Identifier of the specific chunk/ROI where the defect was detected")
    cluster_id: Optional[str] = Field(None, description="Optional cluster/group ID if defects span multiple chunks")

    model_config = ConfigDict(
        extra="forbid"  # Prevent extra fields
    )

class CADDefectReport(BaseModel):
    drawing_summary: str = Field(..., description="Brief overview of the analyzed CAD drawing")
    defects: List[Defect] = Field(
        default_factory=list,
        description="List of detected design defects. Empty list if no defects found."
    )
    overall_recommendations: Optional[str] = Field(
        None,
        description="General advice for improving the drawing or preventing future issues"
    )
    sources: Optional[List[str]] = Field(
        default_factory=list,
        description="List of references or sources used for detection logic"
    )

    model_config = ConfigDict(
        json_encoders={Severity: lambda v: v.value},  # Custom encoder for Severity enum
        use_enum_values=True,                         # Serialize enums as their values
        extra="allow"  # or "forbid" if you want strictness here too
    )

class DesignDefectAnalysis(dspy.Signature):
    """
    Detect *significant* and *clearly visible* design defects in the specific chunk of a CAD drawing layout.
    Avoid reporting minor drafting inconsistencies or ambiguous issues unless they pose a real risk.

    **False Positive Prevention (CRITICAL):**
    - **High Confidence Only**: Only report defects where the visual evidence is unambiguous.
    - **Assume Competence**: Do not flag standard schematic representations as defects.
    - **Conservative Reporting**: It is better to return an empty list than to report false positives.

    Base all findings strictly on visible content in the chunk image, using the cluster image, layout image, layout description, 
    and overall description only for supporting context (e.g., terminology, adjacent elements, or 
    project standards). Do not hallucinate, invent, or assume defects, annotations, dimensions, labels, 
    or elements that are not clearly visible. Use exact visible labels, annotations, dimensions, and 
    symbols when evidencing defects. Use only what is observable in the images. Do not infer or estimate 
    coordinates or measurements. If describing locations, use descriptive position based on the image 
    (e.g., "center area between labels X and Y") rather than numeric coordinates.

    If user_query is provided, focus defect detection on aspects related to the query 
    (e.g., specific code compliance, equipment type, or issue category). If None, perform general 
    defect detection focusing on MAJOR violations.

    **Critical Language Requirement**:
    - Detect the language of the user_query automatically.
    - If user_query is provided and not None, generate **ALL textual content in the output** 
      (including defect titles/categories, descriptions, evidence references, and any other text fields 
      in Defect objects) **entirely in the same language as the user_query**.
    - This includes translating or generating defect category names (e.g., "Space Constraint", 
      "Clearance Issue", "Routing Conflict") into the equivalent terms in the user_query's language.
    - Do NOT mix languages: no English category titles with non-English descriptions.
    - If user_query is None, empty, or in English, output everything in English (no Chinese text).

    Consider common defect types such as (use equivalent terms in the output language):
    - Dimensional inconsistencies or errors
    - Missing required elements, labels, or annotations
    - Overlapping, conflicting, or unclear elements
    - Incorrect or missing symbols/hatch patterns (relative to any visible legend)
    - Apparent non-compliance with visible notes, standards, or references
    - Illegible or incomplete annotations
    - Layout or clearance issues visible in the chunk

    Describe each defect professionally, tying it directly to specific visible features. 
    If no defects are found, return an empty list.
    """

    user_query: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional user query to guide/focus defect detection. If None, perform general analysis. "
             "The language of this field **strictly determines the entire output language** (all defect text, "
             "including titles/categories)."
    )

    description: str = dspy.InputField(
        desc="Comprehensive description of the full original CAD drawing for project context."
    )

    layout_description: str = dspy.InputField(
        desc="Detailed description of the current layout for immediate context."
    )

    layout_image: dspy.Image = dspy.InputField(
        desc="Full layout image for broader spatial and systemic context."
    )

    cluster: dspy.Image = dspy.InputField(
        desc="Expanded cluster/region image containing the chunk and surrounding elements."
    )

    chunk: dspy.Image = dspy.InputField(
        desc="Specific chunk image to analyze in detail. Primary focus: detect defects from "
             "visible content here."
    )

    defects: List[Defect] = dspy.OutputField(
        desc="List of detected defects in the chunk. Each defect based solely on visible evidence. "
             "Use exact labels and annotations. No invented defects or details. Empty list if none. "
             "**ALL text in defects (titles, categories, descriptions, etc.) must match the language "
             "of user_query exactly, with no English leftovers.**"
    )

class LayoutDefectReport(dspy.Signature):
    """
    Aggregate and summarize all defects found across chunks in a layout.
    Produce a clean, complete defect report.
    """

    drawing_summary: str = dspy.InputField(desc="High-level summary of the entire CAD drawing")
    layout_description: str = dspy.InputField(desc="Description of this specific layout")
    all_defects_json: str = dspy.InputField(
        desc="JSON string of all Defect objects found across all chunks in this layout"
    )

    report: CADDefectReport = dspy.OutputField(desc="Final structured defect analysis report")


def clean_for_serialization(obj: Any) -> Any:
    """
    Recursively clean objects for JSON serialization.
    Handles dspy.Image and other non-serializable types.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: clean_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_serialization(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_serialization(v) for v in obj]
    
    # Handle dspy.Image
    if isinstance(obj, dspy.Image):
        # Return a simplified string representation to avoid serialization errors
        # and massive base64 strings in logs
        return f"<dspy.Image>"
        
    if hasattr(obj, "model_dump"):
        return clean_for_serialization(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return clean_for_serialization(obj.to_dict())
        
    # Fallback
    return str(obj)

def format_trajectory(trajectory: Union[List[Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format the DSPy trajectory into a flattened dictionary structure 
    as requested by the user.
    """
    if isinstance(trajectory, dict):
        return clean_for_serialization(trajectory)

    formatted = {}
    for i, step in enumerate(trajectory):
        if isinstance(step, str):
             # Fallback for string steps
             formatted[f"step_{i}"] = step
             continue
        
        # Try to extract standard ReAct attributes
        if hasattr(step, "thought"):
            formatted[f"thought_{i}"] = clean_for_serialization(step.thought)
        elif isinstance(step, dict) and "thought" in step:
            formatted[f"thought_{i}"] = clean_for_serialization(step["thought"])
            
        # Handle action (Tool call)
        if hasattr(step, "action"):
             # step.action might be an object or string
             action = step.action
             if hasattr(action, "name"):
                 formatted[f"tool_name_{i}"] = clean_for_serialization(action.name)
                 formatted[f"tool_args_{i}"] = clean_for_serialization(action.args if hasattr(action, "args") else {})
             elif isinstance(action, dict):
                 formatted[f"tool_name_{i}"] = clean_for_serialization(action.get("name"))
                 formatted[f"tool_args_{i}"] = clean_for_serialization(action.get("args"))
             else:
                 # Raw string or unknown object
                 formatted[f"action_{i}"] = clean_for_serialization(str(action))
                 
        elif isinstance(step, dict) and "action" in step:
             formatted[f"action_{i}"] = clean_for_serialization(step["action"])
             
        # Handle observation
        if hasattr(step, "observation"):
            formatted[f"observation_{i}"] = clean_for_serialization(step.observation)
        elif isinstance(step, dict) and "observation" in step:
            formatted[f"observation_{i}"] = clean_for_serialization(step["observation"])
            
    return formatted

def run_agent_pipeline(
    file_id: Optional[str] = None,
    user_query: Optional[str] = None,
    # Overrides for testing or manual runs
    geojson_file: Optional[str] = None,
    transform_file: Optional[str] = None,
    image_file: Optional[str] = None,
    index_file: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run the agent analysis pipeline as a generator, yielding results at each step.
    
    Yields events with format:
    {"type": "event_type", "data": {...}}
    
    Event types:
    - status: Progress updates
    - cad_analysis: CADAnalyzer output
    - layout_analysis: LayoutAnalyzer output
    - defect_found: DesignDefectAnalysis output (chunk level)
    - report: Final aggregated report
    """
    logger = _setup_logging()

    # Configure DSPy
    vision_model = os.getenv("VISION_MODEL")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("BASE_URL")
    
    if not (vision_model and api_key and base_url):
         # If not set, try to use defaults or raise
         vision_model = vision_model or "gpt-4o"
         # We can't proceed without API key usually, but assuming env is set
         if not api_key:
             raise RuntimeError("DASHSCOPE_API_KEY not found in environment")
             
    lm = dspy.LM(f"openai/{vision_model}", api_key=api_key, base_url=base_url)
    
    with dspy.context(lm=lm):
        logger.info("Configured DSPy LM: %s", f"openai/{vision_model}")

        # Resolve paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if file_id:
            storage_geojson = os.path.join(project_root, "storage", "geojson")
            storage_images = os.path.join(project_root, "storage", "images")
            storage_embeddings = os.path.join(project_root, "storage", "embeddings")
            
            geojson_file = geojson_file or os.path.join(storage_geojson, f"{file_id}.geojson")
            transform_file = transform_file or os.path.join(storage_images, f"{file_id}_transform.json")
            image_file = image_file or os.path.join(storage_images, f"{file_id}.png")
            index_file = index_file or os.path.join(storage_embeddings, f"{file_id}_chunks.parquet")
        else:
            # Fallback to env or defaults (legacy behavior)
            geojson_file = geojson_file or os.getenv("GEOJSON_FILE") or "chunked_data/冷冻机房0327_t3.geojson"
            transform_file = transform_file or os.getenv("TRANSFORM_FILE") or "images/冷冻机房0327_t3_transform.json"
            image_file = image_file or os.getenv("IMAGE_FILE") or "images/冷冻机房0327_t3_OVERLAY.png"
            index_file = index_file or os.getenv("INDEX_FILE") or "chunk_embeddings.parquet"

        # Validate inputs
        if not os.path.exists(geojson_file):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_file}")
        if not os.path.exists(transform_file):
            raise FileNotFoundError(f"Transform file not found: {transform_file}")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        yield {"type": "status", "data": {"message": "Loading data..."}}

        logger.info("Loading layout dataframe from %s", geojson_file)
        t0 = perf_counter()
        df = create_layout_dataframe(geojson_path=geojson_file, transform_path=transform_file)
        logger.info("Loaded dataframe: %d rows", len(df))

        if df.empty:
            raise RuntimeError("Layout dataframe is empty.")

        yield {"type": "status", "data": {"message": "Building embedding index..."}}
        logger.info("Building embedding index: %s", index_file)
        
        # Ensure embedding directory exists
        os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
        
        create_embedding_index(df, image_file, index_file)

        yield {"type": "status", "data": {"message": "Clustering embeddings..."}}
        logger.info("Clustering embeddings from: %s", index_file)
        clusters = cluster_embeddings(index_file, visualize=False, method="agglomerative")
        
        yield {"type": "status", "data": {"message": "Analyzing full CAD drawing..."}}
        logger.info("Analyzing full CAD drawing: %s", image_file)
        
        analyzer = dspy.ReAct(CADAnalyzer, tools=[search_building_codes], max_iters=3)
        drawing_output = analyzer(
            user_query=user_query,
            image=dspy.Image.from_PIL(_load_resized_image(image_file)),
        )
        
        # Yield CAD Analysis Result
        yield {
            "type": "cad_analysis",
            "data": {
                "trajectory": format_trajectory(drawing_output.trajectory) if hasattr(drawing_output, "trajectory") else {},
                "reasoning": drawing_output.reasoning if hasattr(drawing_output, "reasoning") else "",
                "response": drawing_output.response
            }
        }

        layout_ids = list(df["layout_id"].unique())
        layout_descriptions: Dict[int, str] = {}
        
        for l in layout_ids:
            yield {"type": "status", "data": {"message": f"Analyzing Layout {l}..."}}
            raw_img, _ = get_segmentation_crops(df, image_file, target_layouts=[l])
            
            layout_analyzer = dspy.ReAct(LayoutAnalyzer, tools=[search_building_codes], max_iters=3)
            layout_output = layout_analyzer(
                user_query=user_query,
                description=drawing_output.response,
                full_image=dspy.Image.from_PIL(_load_resized_image(image_file)),
                layout_image=dspy.Image.from_PIL(raw_img),
            )
            
            layout_descriptions[int(l)] = layout_output.response
            
            # Yield Layout Analysis Result
            yield {
                "type": "layout_analysis",
                "data": {
                    "layout_id": str(l),
                    "trajectory": format_trajectory(layout_output.trajectory) if hasattr(layout_output, "trajectory") else {},
                    "reasoning": layout_output.reasoning if hasattr(layout_output, "reasoning") else "",
                    "response": layout_output.response
                }
            }

        coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD") or "0.85")
        chunk_analyzer = dspy.ChainOfThought(DesignDefectAnalysis)
        aggregator = dspy.ChainOfThought(LayoutDefectReport)
        
        all_layout_reports: Dict[int, CADDefectReport] = {}

        for l in layout_ids:
            layout_img, _ = get_segmentation_crops(df, image_file, target_layouts=[l])
            layout_desc = layout_descriptions[int(l)]
            layout_area = layout_img.width * layout_img.height
            
            layout_defects: List[Defect] = []
            defect_id_counter = 1
            
            yield {"type": "status", "data": {"message": f"Detecting defects in Layout {l}..."}}

            try:
                layout_key = int(l)
            except Exception:
                layout_key = l

            analyzed_chunk_ids: Set[int] = set()

            def _cluster_sort_key(item):
                try:
                    return int(item[0])
                except Exception:
                    return 0

            for cluster_id, data in sorted(clusters.items(), key=_cluster_sort_key):
                layout_cluster_data = data.get(layout_key)
                if not layout_cluster_data:
                    continue

                cluster_chunk_ids = layout_cluster_data.get("chunk_ids") or []
                cluster_chunk_ids = sorted({int(x) for x in cluster_chunk_ids})
                if not cluster_chunk_ids:
                    continue

                try:
                    cluster_id_int = int(cluster_id)
                except Exception:
                    cluster_id_int = None

                multi_chunk = len(cluster_chunk_ids) > 1
                ext_roi, _ = get_segmentation_crops(
                    df,
                    image_file,
                    target_layouts=[l],
                    target_chunks=cluster_chunk_ids,
                )
                if ext_roi is None:
                    continue

                for ch_id in cluster_chunk_ids:
                    if ch_id in analyzed_chunk_ids:
                        continue

                    roi, _ = get_segmentation_crops(
                        df,
                        image_file,
                        target_layouts=[l],
                        target_chunks=[ch_id],
                    )
                    if roi is None:
                        continue

                    if multi_chunk:
                        ratio = (roi.width * roi.height) / layout_area
                        if ratio >= coverage_threshold:
                            analyzed_chunk_ids.add(ch_id)
                            continue

                    try:
                        normalized_user_query = (user_query or "").strip() or None
                        effective_user_query = normalized_user_query or "detect design defects in this chunk"

                        def _is_ascii(text: str) -> bool:
                            return all(ord(ch) < 128 for ch in text)

                        def _contains_cjk(text: str) -> bool:
                            for ch in text:
                                code = ord(ch)
                                if (
                                    0x4E00 <= code <= 0x9FFF
                                    or 0x3400 <= code <= 0x4DBF
                                    or 0x20000 <= code <= 0x2A6DF
                                    or 0x2A700 <= code <= 0x2B73F
                                    or 0x2B740 <= code <= 0x2B81F
                                    or 0x2B820 <= code <= 0x2CEAF
                                    or 0xF900 <= code <= 0xFAFF
                                    or 0x2F800 <= code <= 0x2FA1F
                                ):
                                    return True
                            return False

                        def _defects_have_cjk(defects_list: List[Defect]) -> bool:
                            for d in defects_list:
                                for field_name in ("type", "description", "location", "how_to_fix"):
                                    value = getattr(d, field_name, None)
                                    if isinstance(value, str) and _contains_cjk(value):
                                        return True
                            return False

                        if _is_ascii(effective_user_query):
                            effective_user_query = (
                                effective_user_query
                                + "\n\nOutput language requirement: English only. Do not output Chinese. Do not mix languages. "
                                  "Keep any drawing labels exactly as shown in the images; do not translate labels."
                            )

                        prediction = chunk_analyzer(
                            user_query=effective_user_query,
                            description=drawing_output.response,
                            layout_description=layout_desc,
                            layout_image=dspy.Image.from_PIL(layout_img),
                            cluster=dspy.Image.from_PIL(ext_roi),
                            chunk=dspy.Image.from_PIL(roi),
                        )

                        if _is_ascii(normalized_user_query or "detect design defects in this chunk") and _defects_have_cjk(prediction.defects):
                            retry_user_query = (
                                (normalized_user_query or "detect design defects in this chunk")
                                + "\n\nOutput language requirement: English only. Do not output Chinese. Do not mix languages. "
                                  "Every text field in every defect must be English. "
                                  "Keep any drawing labels exactly as shown in the images; do not translate labels."
                            )
                            prediction = chunk_analyzer(
                                user_query=retry_user_query,
                                description=drawing_output.response,
                                layout_description=layout_desc,
                                layout_image=dspy.Image.from_PIL(layout_img),
                                cluster=dspy.Image.from_PIL(ext_roi),
                                chunk=dspy.Image.from_PIL(roi),
                            )
                        
                        chunk_enriched_defects: List[Defect] = []
                        for defect in prediction.defects:
                            enriched_defect = defect.model_copy(
                                update={
                                    "id": defect_id_counter,
                                    "layout_id": str(l),
                                    "chunk_id": str(ch_id),
                                    "cluster_id": str(cluster_id_int) if multi_chunk else None,
                                }
                            )
                            layout_defects.append(enriched_defect)
                            chunk_enriched_defects.append(enriched_defect)
                            defect_id_counter += 1
                        
                        if chunk_enriched_defects:
                            yield {
                                "type": "defect_found",
                                "data": {
                                    "layout_id": str(l),
                                    "chunk_id": str(ch_id),
                                    "cluster_id": str(cluster_id_int) if multi_chunk else None,
                                    "reasoning": prediction.reasoning if hasattr(prediction, "reasoning") else "",
                                    "defects": [d.model_dump() for d in chunk_enriched_defects]
                                }
                            }
                        else:
                            yield {
                                 "type": "defect_analysis_progress",
                                 "data": {
                                    "layout_id": str(l),
                                    "chunk_id": str(ch_id),
                                    "reasoning": prediction.reasoning if hasattr(prediction, "reasoning") else "",
                                    "defects": []
                                 }
                            }
                        analyzed_chunk_ids.add(ch_id)
                    except Exception:
                        logger.exception("Error analyzing chunk | layout=%s chunk=%s", l, ch_id)
                        analyzed_chunk_ids.add(ch_id)
                        continue
            
            # Aggregate Report
            if layout_defects:
                all_defects_json = json.dumps([d.model_dump() for d in layout_defects], ensure_ascii=False)
                try:
                    final_pred = aggregator(
                        drawing_summary=drawing_output.response,
                        layout_description=layout_desc,
                        all_defects_json=all_defects_json,
                    )
                    report_raw = final_pred.report
                    report = (
                        report_raw
                        if isinstance(report_raw, CADDefectReport)
                        else CADDefectReport.model_validate(report_raw)
                    )
                except Exception:
                    report = CADDefectReport(
                        drawing_summary=drawing_output.response,
                        defects=layout_defects,
                        overall_recommendations="Defects detected but aggregation failed. Review individual findings.",
                        sources=["DSPy Chunk Analysis (fallback mode)"],
                    )
            else:
                report = CADDefectReport(
                    drawing_summary=f"Layout {l}: {layout_desc[:200]}...",
                    defects=[],
                    overall_recommendations="No design defects detected in this layout after detailed chunk analysis.",
                    sources=["DSPy Vision Analysis", "Segmented ROI Inspection"],
                )
            
            all_layout_reports[int(l)] = report
            
            yield {
                "type": "report",
                "data": {
                    "layout_id": str(l),
                    "report": report.model_dump()
                }
            }

def main():
    # Example usage script
    defects_by_layout = defaultdict(list)
    print("--- Starting Pipeline ---")

    for event in run_agent_pipeline():
        event_type = event["type"]
        data = event["data"]

        if event_type == "cad_analysis":
            print("\n[CAD Analysis Complete]")
            print(data["response"][:200] + "...")
        elif event_type == "layout_analysis":
            print(f"\n[Layout {data['layout_id']} Analysis Complete]")
            print(data["response"][:200] + "...")
        elif event_type == "defect_found":
            l_id = data['layout_id']
            c_id = data['chunk_id']
            defects = data['defects']
            defects_by_layout[l_id].extend(defects)
            print(f"  -> Layout {l_id} | Chunk {c_id}: Found {len(defects)} defects")
        elif event_type == "report":
             print(f"[Report Generated for Layout {data['layout_id']}]")

    print("\n=== Final Defect Summary by Layout ===")
    for l_id, defects in sorted(defects_by_layout.items()):
        print(f"\nLayout {l_id} (Total Defects: {len(defects)}):")
        for d in defects:
            severity = d.get('severity', 'UNK')
            dtype = d.get('type', 'Unknown')
            desc = d.get('description', 'No description')
            print(f"  - [{severity}] {dtype}: {desc}")

if __name__ == "__main__":
    main()
