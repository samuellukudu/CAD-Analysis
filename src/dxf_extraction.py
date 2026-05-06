
import ezdxf
import math
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

class DXFEntity(BaseModel):
    """
    Standardized representation of a DXF entity.
    """
    type: str
    layer: str
    handle: Optional[str] = None
    geometry: Dict[str, Any] = Field(default_factory=dict)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None
    block_name: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    insert: Optional[List[float]] = None

    model_config = ConfigDict(extra="ignore")

class DXFExtractionResult(BaseModel):
    entities: List[DXFEntity]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    header: Dict[str, Any] = Field(default_factory=dict)
    layers: List[Dict[str, Any]] = Field(default_factory=list)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    linetypes: List[Dict[str, Any]] = Field(default_factory=list)
    styles: List[Dict[str, Any]] = Field(default_factory=list)
    dimstyles: List[Dict[str, Any]] = Field(default_factory=list)

class DXFExtractor:
    def __init__(self, dxf_path: str, encoding: Optional[str] = None):
        self.dxf_path = dxf_path
        self.encoding = encoding
        self.doc = self._load_dxf()

    def _load_dxf(self):
        """Loads the DXF file with encoding fallback."""
        encodings = [self.encoding] if self.encoding else ["utf-8", "gbk", "cp936", "gb18030", "latin1"]
        
        for enc in encodings:
            try:
                return ezdxf.readfile(self.dxf_path, encoding=enc)
            except (UnicodeDecodeError, ezdxf.DXFError):
                continue
        
        # Final attempt with default (cp1252 usually)
        try:
            return ezdxf.readfile(self.dxf_path)
        except Exception as e:
            logger.error(f"Failed to load DXF file {self.dxf_path}: {e}")
            raise

    def extract_all(self) -> DXFExtractionResult:
        """Extracts all supported entities from ModelSpace."""
        msp = self.doc.modelspace()
        entities = []
        
        for entity in msp:
            processed = self._process_entity(entity)
            if processed:
                entities.append(processed)
                
        return DXFExtractionResult(
            entities=entities,
            metadata=self._extract_metadata(),
            header=self._extract_header(),
            layers=self._extract_layers(),
            blocks=self._extract_blocks(),
            linetypes=self._extract_linetypes(),
            styles=self._extract_styles(),
            dimstyles=self._extract_dimstyles()
        )

    def extract_within_polygons(self, polygons: List[List[List[float]]]) -> DXFExtractionResult:
        """
        Extracts entities that intersect with the given polygons.
        polygons: List of polygons, where each polygon is a list of [x, y] points.
        """
        if not polygons:
            return DXFExtractionResult(entities=[])

        shapely_polys = []
        for poly in polygons:
            if len(poly) >= 3:
                shapely_polys.append(Polygon(poly))
        
        if not shapely_polys:
            return DXFExtractionResult(entities=[])

        roi_union = unary_union(shapely_polys)
        
        msp = self.doc.modelspace()
        entities = []
        
        for entity in msp:
            # First check if entity bbox intersects ROI (fast check)
            # ezdxf entities usually don't have bbox directly available without calculation
            # We will process geometry first then check intersection
            
            processed = self._process_entity(entity)
            if not processed:
                continue

            # Convert processed geometry to Shapely for intersection check
            geom_shape = self._to_shapely(processed.geometry)
            if geom_shape and geom_shape.intersects(roi_union):
                entities.append(processed)

        return DXFExtractionResult(
            entities=entities,
            metadata=self._extract_metadata(),
            header=self._extract_header(),
            layers=self._extract_layers()
        )

    def _process_entity(self, entity) -> Optional[DXFEntity]:
        try:
            dxftype = entity.dxftype()
            layer = getattr(entity.dxf, "layer", "0")
            handle = getattr(entity.dxf, "handle", None)
            
            geometry = self._extract_geometry(entity)
            attributes = self._extract_attributes(entity)
            text = self._extract_text(entity)
            block_name = getattr(entity.dxf, "name", None) if dxftype == "INSERT" else None
            extra = self._extract_extra(entity)
            insert = self._extract_insert(entity)

            return DXFEntity(
                type=dxftype,
                layer=layer,
                handle=handle,
                geometry=geometry,
                attributes=attributes,
                text=text,
                block_name=block_name,
                extra=extra,
                insert=insert
            )
        except Exception as e:
            logger.warning(f"Failed to process entity {entity}: {e}")
            return None

    def _extract_geometry(self, entity) -> Dict[str, Any]:
        dxftype = entity.dxftype()
        try:
            if dxftype == "LINE":
                return {
                    "type": "LINESTRING",
                    "coordinates": [
                        [float(entity.dxf.start.x), float(entity.dxf.start.y)], 
                        [float(entity.dxf.end.x), float(entity.dxf.end.y)]
                    ]
                }
            elif dxftype == "LWPOLYLINE":
                pts = [[float(x), float(y)] for x, y, *_ in entity.get_points()]
                if not pts: return {}
                if getattr(entity, "closed", False) and len(pts) >= 3:
                    pts = self._ensure_closed_ring(pts)
                    return {"type": "POLYGON", "coordinates": [pts]}
                return {"type": "LINESTRING", "coordinates": pts}
            elif dxftype == "POLYLINE":
                # 2D/3D Polyline
                pts = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in entity.vertices]
                if not pts: return {}
                if getattr(entity, "is_closed", False) and len(pts) >= 3:
                    pts = self._ensure_closed_ring(pts)
                    return {"type": "POLYGON", "coordinates": [pts]}
                return {"type": "LINESTRING", "coordinates": pts}
            elif dxftype == "CIRCLE":
                return {
                    "type": "CIRCLE",
                    "center": [float(entity.dxf.center.x), float(entity.dxf.center.y)],
                    "radius": float(entity.dxf.radius)
                }
            elif dxftype == "ARC":
                center = [float(entity.dxf.center.x), float(entity.dxf.center.y)]
                radius = float(entity.dxf.radius)
                start_angle = float(entity.dxf.start_angle)
                end_angle = float(entity.dxf.end_angle)
                pts = self._sample_arc_points(center, radius, start_angle, end_angle)
                return {
                    "type": "ARC",
                    "center": center,
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "coordinates": pts
                }
            elif dxftype in ("TEXT", "MTEXT", "INSERT"):
                if hasattr(entity.dxf, "insert"):
                    return {
                        "type": "POINT", 
                        "coordinates": [float(entity.dxf.insert.x), float(entity.dxf.insert.y)]
                    }
            elif dxftype == "HATCH":
                # Complex handling for hatch paths
                paths = []
                for path in getattr(entity, "paths", []):
                    if hasattr(path, "vertices"):
                        pts = [[float(v[0]), float(v[1])] for v in path.vertices]
                        if len(pts) >= 3:
                            paths.append(self._ensure_closed_ring(pts))
                if paths:
                    return {"type": "POLYGON", "coordinates": paths}
        except Exception:
            pass
        return {}

    def _extract_attributes(self, entity) -> Dict[str, Any]:
        try:
            return {k: self._serialize_value(v) for k, v in entity.dxfattribs().items()}
        except Exception:
            return {}

    def _extract_text(self, entity) -> Optional[str]:
        dxftype = entity.dxftype()
        if dxftype == "TEXT":
            return getattr(entity.dxf, "text", None)
        elif dxftype == "MTEXT":
            # MTEXT can have complex formatting, use ezdxf helper or raw text
            return getattr(entity, "text", None) # MTEXT usually exposes content via text property or plain_text()
        return None

    def _extract_extra(self, entity) -> Dict[str, Any]:
        dxftype = entity.dxftype()
        extra = {}
        try:
            if dxftype == "INSERT":
                extra = {
                    "name": getattr(entity.dxf, "name", None),
                    "rotation": float(getattr(entity.dxf, "rotation", 0)),
                    "x_scale": float(getattr(entity.dxf, "xscale", 1)),
                    "y_scale": float(getattr(entity.dxf, "yscale", 1)),
                    "z_scale": float(getattr(entity.dxf, "zscale", 1)),
                }
            elif dxftype in ("TEXT", "MTEXT"):
                extra = {
                    "height": float(getattr(entity.dxf, "height", 0)),
                    "rotation": float(getattr(entity.dxf, "rotation", 0)),
                    "style": getattr(entity.dxf, "style", None),
                    "width_factor": float(getattr(entity.dxf, "width", 1.0) if dxftype == "TEXT" else 1.0),
                }
                if dxftype == "MTEXT":
                    extra["char_height"] = float(getattr(entity.dxf, "char_height", 0))
                    extra["attachment_point"] = getattr(entity.dxf, "attachment_point", None)
            elif dxftype == "LWPOLYLINE":
                extra = {
                    "closed": getattr(entity, "closed", False),
                    "count": len(entity),
                    "has_width": any(w1 > 0 or w2 > 0 for _, _, w1, w2, _ in entity.get_points())
                }
            elif dxftype == "HATCH":
                extra = {
                    "pattern_name": getattr(entity.dxf, "pattern_name", None),
                    "solid_fill": bool(getattr(entity.dxf, "solid_fill", False)),
                    "pattern_scale": float(getattr(entity.dxf, "pattern_scale", 1.0)),
                    "pattern_angle": float(getattr(entity.dxf, "pattern_angle", 0.0)),
                }
            elif dxftype == "ARC":
                extra = {
                    "start_angle": float(entity.dxf.start_angle),
                    "end_angle": float(entity.dxf.end_angle),
                }
        except Exception:
            pass
        return extra

    def _extract_insert(self, entity) -> Optional[List[float]]:
        if hasattr(entity.dxf, "insert"):
            try:
                return [float(entity.dxf.insert.x), float(entity.dxf.insert.y)]
            except Exception:
                pass
        return None

    def _extract_metadata(self) -> Dict[str, Any]:
        return {
            "dxf_path": self.dxf_path,
            "encoding": self.encoding,
            "version": getattr(self.doc, "dxfversion", None)
        }
        
    def _extract_header(self) -> Dict[str, Any]:
        header = {}
        try:
            if hasattr(self.doc, "header"):
                for key, value in self.doc.header.items():
                    header[key] = self._serialize_value(value)
        except Exception:
            pass
        return header

    def _extract_layers(self) -> List[Dict[str, Any]]:
        layers = []
        try:
            for layer in self.doc.layers:
                layers.append({
                    "name": layer.dxf.name,
                    "color": layer.dxf.color,
                    "linetype": layer.dxf.linetype,
                    "is_frozen": layer.is_frozen(),
                    "is_locked": layer.is_locked(),
                    "is_off": layer.is_off()
                })
        except Exception:
            pass
        return layers

    def _extract_blocks(self) -> List[Dict[str, Any]]:
        blocks = []
        try:
            for block in self.doc.blocks:
                blocks.append({"name": block.name})
        except Exception:
            pass
        return blocks

    def _extract_linetypes(self) -> List[Dict[str, Any]]:
        linetypes = []
        try:
            for ltype in self.doc.linetypes:
                linetypes.append({"name": ltype.dxf.name, "description": ltype.dxf.description})
        except Exception:
            pass
        return linetypes

    def _extract_styles(self) -> List[Dict[str, Any]]:
        styles = []
        try:
            for style in self.doc.styles:
                styles.append({"name": style.dxf.name, "height": style.dxf.height})
        except Exception:
            pass
        return styles
        
    def _extract_dimstyles(self) -> List[Dict[str, Any]]:
        dimstyles = []
        try:
            for dimstyle in self.doc.dimstyles:
                dimstyles.append({"name": dimstyle.dxf.name})
        except Exception:
            pass
        return dimstyles


    @staticmethod
    def to_shapely(geometry_payload: Dict[str, Any]):
        if not geometry_payload:
            return None
        try:
            g_type = geometry_payload.get("type")
            coords = geometry_payload.get("coordinates")
            if g_type == "POINT" and coords:
                return Point(coords)
            elif g_type == "LINESTRING" and coords:
                return LineString(coords)
            elif g_type == "POLYGON" and coords:
                # Handle multipolygon (list of rings) or single polygon ring
                # Simplification: just take the first ring for simple cases or unary_union if multiple
                if isinstance(coords[0][0], list): # List of rings
                    polys = [Polygon(ring) for ring in coords if len(ring) >= 3]
                    return unary_union(polys) if polys else None
                return Polygon(coords)
            elif g_type == "CIRCLE":
                center = geometry_payload.get("center")
                radius = geometry_payload.get("radius")
                if center and radius:
                    return Point(center).buffer(radius)
            elif g_type == "ARC":
                coords = geometry_payload.get("coordinates")
                if coords and len(coords) >= 2:
                    return LineString(coords)
        except Exception:
            pass
        return None

    def _to_shapely(self, geometry_payload: Dict[str, Any]):
        return self.to_shapely(geometry_payload)

    @staticmethod
    def _sample_arc_points(center: List[float], radius: float, start_angle: float, end_angle: float, segments: int = 36) -> List[List[float]]:
        if radius <= 0:
            return []
        if end_angle < start_angle:
            end_angle += 360.0
        step = (end_angle - start_angle) / max(segments, 1)
        points: List[List[float]] = []
        angle = start_angle
        for _ in range(segments + 1):
            rad = math.radians(angle)
            points.append([center[0] + radius * math.cos(rad), center[1] + radius * math.sin(rad)])
            angle += step
        return points


    def _ensure_closed_ring(self, points: List[List[float]]) -> List[List[float]]:
        if len(points) < 3:
            return points
        if points[0] != points[-1]:
            return points + [points[0]]
        return points

    def _serialize_value(self, value: Any) -> Any:
        if hasattr(value, "x") and hasattr(value, "y"):
            return [float(value.x), float(value.y), float(getattr(value, "z", 0))]
        return str(value)

