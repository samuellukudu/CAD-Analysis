import os
import subprocess
from pathlib import Path

# Fix for DSPy readonly database error MUST BE BEFORE IMPORTING DSPY
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), ".dspy_cache")

import dspy
from enum import Enum
from typing import List, Optional, Dict, Any, Generator, Union
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from src.svg2image import svg_to_png_cli

load_dotenv()

lm = dspy.LM(f"openai/{os.getenv('VISION_MODEL')}", api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("BASE_URL"), max_tokens=8192)
dspy.configure(lm=lm)

def prepare_image_path(image_path: Union[str, Path]) -> str:
    """
    Ensures the image is in PNG format for the vision model.
    Converts SVG to PNG automatically if needed.
    """
    path = Path(image_path)
    if path.suffix.lower() == '.svg':
        png_path = path.with_suffix(".png")
        svg_to_png_cli(str(path), str(png_path))
        return str(png_path)
    return str(path)

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
        description="Where in the drawing the issue occurs (e.g., 'Layer: A-WALL', 'View: Section A-A')"
    )
    severity: Severity = Field(..., description="Impact level: Low, Medium, or High")
    how_to_fix: str = Field(..., description="Recommended steps to resolve the defect")

    model_config = ConfigDict(
        extra="forbid"
    )

class ComplianceIssue(BaseModel):
    id: int = Field(..., description="Unique identifier for the compliance issue")
    code_reference: str = Field(..., description="The building code or standard that is violated (e.g., 'ADA', 'IBC', 'Local Fire Code')")
    description: str = Field(..., description="Detailed explanation of the compliance violation")
    location: Optional[str] = Field(None, description="Where in the drawing the violation occurs")
    severity: Severity = Field(..., description="Impact level: Low, Medium, or High")
    how_to_fix: str = Field(..., description="Recommended steps to resolve the compliance issue")

    model_config = ConfigDict(
        extra="forbid"
    )

class ImageAnalysisSignature(dspy.Signature):
    """
    Analyze a rasterized CAD drawing image with exhaustive detail and structure.
    
    This signature is designed to produce a text description that serves as a complete visual ground truth for downstream LLM tasks. 
    When processing the image, you must:
    
    1. Identify and list all visible textual annotations:
       - Room names, labels, functional zones
       - Dimensions (lengths, widths, heights)
       - Grid labels and axes
       - Elevations, section marks, notes
       
    2. Detect and describe all structural and architectural elements:
       - Columns, beams, walls (with types and thicknesses)
       - Doors, windows, staircases, elevators
       - Openings, partitions, ramps
       - Furniture, equipment, or mechanical elements if visible
       
    3. Capture spatial relationships:
       - Relative positions of elements (e.g., which rooms are adjacent)
       - Alignment along grids
       - Orientation (north arrow if present)
       - Layered or overlapping elements
       
    4. Include auxiliary elements:
       - Section cuts, legends, symbols
       - Hatching patterns and line types
       - Annotations like callouts or references
       
    5. Provide structured output:
       - Use clear headings for each element category
       - Group elements hierarchically (e.g., building → floor → room → components)
       - Include measurements and spatial references wherever possible
       
    The output must be exhaustive; no visible detail should be omitted.
    """
    
    query: str = dspy.InputField(
        desc="The user query for the image analysis; can specify focus (e.g., 'structural elements' or 'room dimensions')"
    )
    context: str = dspy.InputField(
        desc="Supplementary context provided by the LLM to help ground the analysis (e.g., prior metadata, CAD type, or drawing conventions)"
    )
    image: dspy.Image = dspy.InputField(
        desc="The rasterized CAD drawing image to analyze"
    )
    response: str = dspy.OutputField(
        desc="An exhaustively detailed, highly structured textual description covering all visual elements, spatial relationships, and annotations in the image"
    )

class DesignDefectSignature(dspy.Signature):
    """
    Analyze a rasterized CAD drawing or design image to identify, classify, 
    and describe potential design defects or inconsistencies. 

    The analysis should be thorough, capturing all visible elements and checking 
    for common design issues. This includes, but is not limited to:

    1. Structural and Architectural Defects:
       - Misaligned or overlapping walls, columns, beams
       - Missing structural elements or supports
       - Inconsistent door/window placement
       - Staircase, elevator, or ramp inconsistencies

    2. Dimension and Annotation Errors:
       - Conflicting or missing dimensions
       - Label mismatches with actual elements
       - Grids and axes inconsistencies
       - Elevation or section reference errors

    3. Spatial and Functional Issues:
       - Room or space layout conflicts
       - Accessibility or code compliance concerns
       - Circulation errors (e.g., blocked pathways)
       - Conflicting equipment or furniture placement

    4. Layering, Hatching, and Symbol Issues:
       - Misuse of hatching patterns
       - Overlapping symbols or annotations
       - Missing legends or references

    Output Requirements:
    - Provide a structured list of detected defects
    - Include defect type, location (grid or room reference), severity, 
      and a clear textual description
    - Maintain exhaustiveness: every visible potential issue should be captured
    - Use spatial references and measurements wherever possible

    This signature is intended to generate a highly reliable defect report 
    that can guide downstream LLM reasoning or CAD review processes.
    """
    
    query: str = dspy.InputField(
        desc="The user query for defect analysis; may specify focus (e.g., 'structural issues' or 'dimension conflicts')"
    )
    context: str = dspy.InputField(
        desc="Supplementary context from the LLM to help ground defect identification (e.g., prior metadata, CAD conventions, building code references)"
    )
    image: dspy.Image = dspy.InputField(
        desc="The rasterized CAD drawing or design image to analyze for defects"
    )
    response: List["Defect"] = dspy.OutputField(
        desc="A structured list of all detected design defects, each including type, location, severity, and detailed description"
    )

class ComplianceAnalysisSignature(dspy.Signature):
    """
    Analyze a rasterized CAD drawing to identify potential building code, accessibility, 
    and safety compliance issues.

    The analysis should be exhaustive and structured, focusing on:
    
    1. Regulatory Compliance:
       - Adherence to local building codes (structural, fire safety, zoning)
       - Correct room sizes, ceiling heights, and occupancy limits
       - Fire exits, egress paths, and emergency access
       - Stair and ramp dimensions and slope regulations

    2. Accessibility Compliance:
       - Compliance with ADA or local accessibility standards
       - Door widths, corridor widths, turning radii
       - Elevator and lift access, wheelchair pathways

    3. Safety and Functional Compliance:
       - Placement of mechanical, electrical, and plumbing systems
       - Structural safety checks: column spacing, load paths, wall integrity
       - Hazard identification (e.g., blocked exits, tight circulation)

    4. Spatial and Dimensional Checks:
       - Grid and axis alignment
       - Consistency between dimensions, annotations, and actual layout
       - Overlaps or missing elements that affect compliance

    Output Requirements:
    - Provide a structured list of compliance issues
    - Each issue should include:
        * Issue type (e.g., accessibility, fire safety, structural)
        * Location (grid reference, room, or element)
        * Description of the violation
        * Severity or impact rating if possible
    - Maintain exhaustive coverage: capture every visible compliance concern
    - Include references to measured dimensions, annotations, and spatial relationships
    - Output should enable downstream reasoning or automated compliance reporting
    """
    
    query: str = dspy.InputField(
        desc="The user query specifying focus areas for compliance analysis (e.g., 'fire safety', 'accessibility')"
    )
    context: str = dspy.InputField(
        desc="Supplementary context from the LLM or prior metadata to ground compliance checking (e.g., code version, standards)"
    )
    image: dspy.Image = dspy.InputField(
        desc="The rasterized CAD drawing image to analyze for compliance issues"
    )
    response: List["ComplianceIssue"] = dspy.OutputField(
        desc="A structured list of all detected compliance issues, including type, location, description, and severity"
    )


class VisualAnalyzer:
    """
    A unified visual analyzer for CAD floor plans and related images.
    It exposes modules for descriptive analysis, design defect identification, and compliance checks.
    """
    def __init__(self):
        self.description_module = dspy.ChainOfThought(ImageAnalysisSignature)
        self.design_module = dspy.ChainOfThought(DesignDefectSignature)
        self.compliance_module = dspy.ChainOfThought(ComplianceAnalysisSignature)

    def describe(self, image_path: Union[str, Path], context: str, query: str = "Provide an exhaustively detailed description of this CAD drawing. Include all text labels, room names, grid axes, dimensions, structural elements (columns, walls, doors, windows), and their exact spatial relationships. Organize the response logically with clear markdown headings.") -> str:
        png_path = prepare_image_path(image_path)
        result = self.description_module(
            query=query,
            context=context,
            image=dspy.Image.from_file(png_path)
        )
        if getattr(result, "response", None) is None:
            return f"Model failed to generate a structured response. Reasoning trace:\n{getattr(result, 'reasoning', 'No output generated.')}"
        return result.response

    def analyze_design(self, image_path: Union[str, Path], context: str, query: str = "Identify any potential design errors, inconsistencies, or omissions in this floor plan.") -> List[Defect]:
        png_path = prepare_image_path(image_path)
        result = self.design_module(
            query=query,
            context=context,
            image=dspy.Image.from_file(png_path)
        )
        if getattr(result, "response", None) is None:
            raise ValueError(f"Model failed to generate a structured response. Reasoning trace:\n{getattr(result, 'reasoning', 'No output generated.')}")
        return result.response

    def analyze_compliance(self, image_path: Union[str, Path], context: str, query: str = "Check this floor plan for compliance issues regarding safety, accessibility, and standard building codes.") -> List[ComplianceIssue]:
        png_path = prepare_image_path(image_path)
        result = self.compliance_module(
            query=query,
            context=context,
            image=dspy.Image.from_file(png_path)
        )
        if getattr(result, "response", None) is None:
            raise ValueError(f"Model failed to generate a structured response. Reasoning trace:\n{getattr(result, 'reasoning', 'No output generated.')}")
        return result.response

if __name__ == "__main__":
    # Quick test of the module
    analyzer = VisualAnalyzer()
    
    # Mock context
    test_context = "This is a region of the first floor plan of a Production Comprehensive Building."
    
    # Check if the output_refined directory has files we can test
    test_image = Path("storage/images/roof_plan.svg")
    
    if test_image.exists():
        print(f"Testing description on: {test_image}")
        desc = analyzer.describe(
            image_path=test_image,
            context=test_context,
            query="Describe the main architectural features visible in this cropped region."
        )
        print("\n--- Description ---")
        print(desc)
    else:
        print(f"Test image not found: {test_image}")
