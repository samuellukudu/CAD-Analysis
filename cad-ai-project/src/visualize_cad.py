import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import argparse
import os
import sys
import numpy as np
import re

import matplotlib.path as mpath

# Set non-interactive backend
plt.switch_backend('Agg')

# Matplotlib text: scale up vs raw CAD-derived height (+25%).
TEXT_FONT_SIZE_SCALE = 1.25

def get_system_cjk_font():
    env_font = os.environ.get("CAD_CJK_FONT")
    if env_font and os.path.exists(env_font):
        return env_font
    
    # Prioritize Simplified Chinese Fonts for macOS
    candidates = [
        # Project Local Fonts (For Deployment)
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Regular.otf"),
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Medium.otf"),
        
        # User Installed Noto Sans CJK SC (Best open source option - Explicitly Simplified)
        os.path.expanduser("~/Library/Fonts/NotoSansCJKsc-Regular.otf"),
        os.path.expanduser("~/Library/Fonts/NotoSansCJKsc-Medium.otf"),
        "/Library/Fonts/NotoSansCJKsc-Regular.otf",

        # macOS System Fonts (Simplified)
        "/System/Library/Fonts/PingFang SC.ttc", # Specific SC version
        "/System/Library/Fonts/Heiti SC.ttc",
        "/System/Library/Fonts/PingFang.ttc", # Generic
        "/System/Library/Fonts/STHeiti Light.ttc", # Can be ambiguous
        
        # Linux / Server Fonts
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        
        # Fallbacks
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
            
    try:
        fonts = set()
        for ext in ["ttf", "ttc", "otf"]:
            for fp in font_manager.findSystemFonts(fontext=ext):
                fonts.add(fp)
        
        # Keywords prioritized for Simplified Chinese
        keywords = [
            "sc", # Simplified Chinese
            "noto sans cjk sc",
            "notosanscjksc",
            "source han sans sc",
            "pingfang sc",
            "heiti sc",
            "simhei",
            "simsun",
            "msyh",
            "pingfang", # Generic fallback
            "heiti",
            "wenquanyi",
            "uming",
            "ukai",
            "ar pl",
        ]
        
        # Better search strategy: Iterate keywords (priority) and check all fonts
        for k in keywords:
            for path in fonts:
                name = os.path.basename(path).lower().replace("_", " ").replace("-", " ")
                if k in name:
                    return path

    except Exception:
        pass
    return None

def decode_cad_unicode(text):
    r"""
    Decodes \U+XXXX and strips AutoCAD MTEXT formatting.
    """
    if not text:
        return ""
    
    # 1. Strip MTEXT formatting
    # \A...; \C...; \H...; \Q...; \T...; \W...; \f...;
    text = re.sub(r'\\[ACFHQTWf].*?;', '', text)
    # \S...; (Stacking) - keeping just the first part if simpler, or just removing formatting
    # Stacking is complex like \S1^2; -> 1/2. simpler to just strip \S and ; and replace ^ with /
    # But for now let's just strip the tag wrapper?
    # Actually, \S...; content is valuable.
    # Simple approach: remove \S and ; and ^
    
    def clean_stacking(match):
        return match.group(0).replace(r'\S', '').replace(';', '').replace('^', '/')
    
    text = re.sub(r'\\S.*?;', clean_stacking, text)
    
    # \L, \O, \l, \o (Underline/Overline)
    text = re.sub(r'\\[LOlo]', '', text)
    # \~ (Non-breaking space)
    text = text.replace(r'\~', ' ')
    # {} (Braces)
    text = re.sub(r'[{}]', '', text)
    
    # 2. Manual Unicode Decoding for \U+XXXX
    def replace_unicode(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except:
            return match.group(0)

    # Regex to find \U+XXXX (Case insensitive)
    text = re.sub(r'\\U\+([0-9A-Fa-f]{4})', replace_unicode, text)
    
    # 3. Clean up newlines for MTEXT
    text = text.replace(r'\P', '\n')
    
    return text.strip()

# ACI (AutoCAD Color Index) mapping
ACI_COLORS = {
    1: '#FF0000', # Red
    2: '#FFFF00', # Yellow
    3: '#00FF00', # Green
    4: '#00FFFF', # Cyan
    5: '#0000FF', # Blue
    6: '#FF00FF', # Magenta
    7: '#FFFFFF', # White (on dark background)
    8: '#808080', # Gray
    9: '#C0C0C0', # Light Gray
    250: '#333333', # Dark Gray
    251: '#555555',
    252: '#777777',
    253: '#999999',
    254: '#BBBBBB',
    255: '#FFFFFF'
}

def get_aci_color(index):
    return ACI_COLORS.get(index, '#CCCCCC') # Default to light gray

def int_to_hex(rgb_int):
    """Convert integer RGB to hex string."""
    if rgb_int is None:
        return None
    return f'#{rgb_int:06x}'

def to_radians(angle):
    if angle is None:
        return 0.0
    try:
        a = float(angle)
    except Exception:
        return 0.0
    if abs(a) > (2 * np.pi + 1e-6):
        return np.radians(a)
    return a

def to_degrees(angle):
    if angle is None:
        return 0.0
    try:
        a = float(angle)
    except Exception:
        return 0.0
    if abs(a) <= (2 * np.pi + 1e-6):
        return np.degrees(a)
    return a

class CADVisualizer:
    def __init__(self, json_path):
        print(f"Loading {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.entities = self.data.get('entities', [])
        self.blocks = self.data.get('blocks', {})
        self.layers = self.data.get('tables', {}).get('layer', {}).get('layers', {})
        
        # Precompute layer colors
        self.layer_colors = {}
        for name, layer in self.layers.items():
            color = None
            layer_color = layer.get('color')
            if layer_color is not None:
                if isinstance(layer_color, int) and layer_color <= 255:
                    color = get_aci_color(layer_color)
                else:
                    color = int_to_hex(layer_color)
            elif layer.get('colorIndex'):
                color = get_aci_color(layer.get('colorIndex'))
            self.layer_colors[name] = color or '#FFFFFF'

        # Load Font
        self.font_path = get_system_cjk_font()
        self.font_prop = FontProperties(fname=self.font_path) if self.font_path else None
        if self.font_path:
            print(f"Using font: {self.font_path}")
        else:
            print("Warning: No CJK font found. Text rendering may be incorrect.")

    def get_entity_color(self, entity, parent_layer=None):
        """Determine color based on entity properties and layer."""
        # 1. Entity True Color
        if entity.get('trueColor'):
            return int_to_hex(entity.get('trueColor'))
        
        # 2. Entity ACI Color
        aci = entity.get('color')
        if aci is not None:
            if isinstance(aci, int) and aci > 255:
                return int_to_hex(aci)
            if aci == 0: # ByBlock
                return '#FFFFFF' # Simplification: Default to white for ByBlock
            if aci == 256: # ByLayer
                pass # Fall through to layer check
            else:
                return get_aci_color(aci)
        
        # 3. Layer Color
        layer_name = entity.get('layer') or parent_layer
        if layer_name and layer_name in self.layer_colors:
            return self.layer_colors[layer_name]
            
        return '#FFFFFF' # Default

    def _get_transform_matrix(self, tx, ty, sx, sy, rot):
        rot = to_radians(rot)
        c = np.cos(rot)
        s = np.sin(rot)
        return np.array([
            [sx*c, -sy*s, tx],
            [sx*s,  sy*c, ty],
            [   0,     0,  1]
        ])

    def transform_points(self, points, matrix):
        """
        Apply transformation matrix to a list of (x, y) points.
        points: list of [x, y] or np.array of shape (N, 2)
        matrix: 3x3 transformation matrix
        """
        pts = np.array(points)
        if pts.shape[0] == 0:
            return pts
            
        # Add homogeneous coordinate
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        
        # Transform: (M @ P.T).T
        transformed = (matrix @ pts_h.T).T
        
        return transformed[:, :2]

    def get_solid_points(self, entity):
        pts = []
        raw_points = entity.get('points')
        if isinstance(raw_points, list):
            for p in raw_points:
                if isinstance(p, dict) and 'x' in p and 'y' in p:
                    pts.append({'x': p['x'], 'y': p['y']})
        if not pts:
            for k in ['first', 'second', 'third', 'fourth']:
                p = entity.get(k)
                if p and 'x' in p and 'y' in p:
                    pts.append({'x': p['x'], 'y': p['y']})
        if len(pts) == 4:
            pts = [pts[0], pts[1], pts[3], pts[2]]
        return pts

    def get_line_points(self, entity):
        s = entity.get('start')
        e = entity.get('end')
        if s and e and 'x' in s and 'y' in s and 'x' in e and 'y' in e:
            return [[s['x'], s['y']], [e['x'], e['y']]]
        verts = entity.get('vertices')
        if isinstance(verts, list) and len(verts) >= 2:
            v0 = verts[0]
            v1 = verts[-1]
            if isinstance(v0, dict) and isinstance(v1, dict) and 'x' in v0 and 'y' in v0 and 'x' in v1 and 'y' in v1:
                return [[v0['x'], v0['y']], [v1['x'], v1['y']]]
        return None

    def _edge_angles_to_radians(self, start_angle, end_angle, counter_clockwise=True):
        start_rad = to_radians(start_angle)
        end_rad = to_radians(end_angle)
        if counter_clockwise:
            while end_rad < start_rad:
                end_rad += 2 * np.pi
        else:
            while end_rad > start_rad:
                end_rad -= 2 * np.pi
        return start_rad, end_rad

    def _extract_points_xy(self, points):
        out = []
        if not isinstance(points, list):
            return out
        for p in points:
            if isinstance(p, dict) and 'x' in p and 'y' in p:
                out.append([p['x'], p['y']])
        return out

    def approximate_spline_points(self, spline_like, max_points=180):
        fit_points = self._extract_points_xy(spline_like.get('fitPoints'))
        if len(fit_points) >= 2:
            return fit_points
        control_points = self._extract_points_xy(spline_like.get('controlPoints'))
        if len(control_points) < 2:
            return control_points
        pts = np.array(control_points, dtype=float)
        samples = int(min(max_points, max(2, len(pts) * 8)))
        t_src = np.arange(len(pts), dtype=float)
        t_dst = np.linspace(0.0, len(pts) - 1, samples)
        x = np.interp(t_dst, t_src, pts[:, 0])
        y = np.interp(t_dst, t_src, pts[:, 1])
        return np.column_stack([x, y]).tolist()

    def get_entity_lineweight(self, entity, parent_layer=None, default=0.5):
        lw = entity.get('lineweight')
        if lw is None:
            lw = entity.get('lineWeight')
        if lw is None:
            lw = entity.get('line_weight')
        if lw is None:
            lw = -1

        if lw == -1:
            layer_name = entity.get('layer') or parent_layer
            layer = self.layers.get(layer_name, {}) if layer_name else {}
            lw = layer.get('lineweight')
            if lw is None:
                lw = layer.get('lineWeight')

        if lw is None or lw in (-1, -2, -3):
            return default

        try:
            lw = float(lw)
        except Exception:
            return default

        if lw <= 0:
            return default

        mm = lw / 100.0 if lw > 3 else lw
        return float(np.clip(mm * 4.0, 0.2, 3.0))

    def render(self, output_path, bbox=None, text_size_min=5, text_size_max=16):
        if bbox:
            min_x, min_y, max_x, max_y = bbox
        else:
            print("Warning: No bbox provided. Using default limits.")
            min_x, min_y, max_x, max_y = 0, 0, 1000, 1000

        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate figure size based on aspect ratio (target ~24 inches max dimension)
        if width > height:
            fig_width = 24
            fig_height = 24 * (height / width)
        else:
            fig_height = 24
            fig_width = 24 * (width / height)
            
        # Calculate scale factor (points per drawing unit)
        # 1 inch = 72 points
        points_per_unit = (fig_height * 72) / height
        print(f"Scale: {points_per_unit:.4f} points/unit")
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Remove margins and axes for clean SVG
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')
        
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        print(f"Rendering to {output_path} with bbox {bbox}...")
        
        count = 0
        
        # Initial Transform (Identity)
        identity = np.eye(3)
        
        hatch_like = []
        others = []
        for entity in self.entities:
            if entity.get('type') == 'HATCH':
                hatch_like.append(entity)
            else:
                others.append(entity)

        draw_queue = others + hatch_like

        for entity in draw_queue:
            # Top-level filtering: if entity has a clear position/bbox, check it.
            # But skipping for now to ensure we don't miss blocks inserted at 0,0
            # that cover the area.
            # Optimization: Only filter simple entities (LINE, etc).
            # Always process INSERTs.
            
            etype = entity.get('type')
            if etype in ['LINE', 'LWPOLYLINE', 'POLYLINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'SOLID']:
                # Basic bbox check for primitives
                ebbox = self.get_simple_bbox(entity)
                if ebbox and not self.bbox_intersects(ebbox, bbox):
                    continue
            
            self.draw_entity(ax, entity, identity, bbox, points_per_unit=points_per_unit, 
                             text_size_min=text_size_min, text_size_max=text_size_max)
            count += 1
            
        print(f"Rendered {count} top-level entities. Saving...")
        # Remove bbox_inches='tight' to respect exact limits set by subplots_adjust and xlim/ylim
        plt.savefig(output_path, dpi=200, facecolor='black')
        plt.close()
        print("Done.")

    def get_simple_bbox(self, entity):
        """Get bbox for simple primitives (non-recursive)."""
        etype = entity.get('type')
        xs, ys = [], []
        if etype == 'LINE':
            line_pts = self.get_line_points(entity)
            if line_pts:
                xs = [line_pts[0][0], line_pts[1][0]]
                ys = [line_pts[0][1], line_pts[1][1]]
        elif etype == 'LWPOLYLINE' or etype == 'POLYLINE':
            verts = entity.get('vertices', [])
            if verts:
                xs = [v['x'] for v in verts]
                ys = [v['y'] for v in verts]
        elif etype in ['TEXT', 'MTEXT', 'ATTRIB', 'ATTDEF', 'INSERT', 'POINT']:
             p = entity.get('insertPoint') or entity.get('position') or entity.get('center') or entity.get('startPoint')
             if p:
                 xs, ys = [p['x']], [p['y']]
        elif etype == 'CIRCLE':
            c = entity.get('center')
            r = entity.get('radius')
            if c and r is not None:
                xs = [c['x'] - r, c['x'] + r]
                ys = [c['y'] - r, c['y'] + r]
        elif etype == 'ARC':
            c = entity.get('center')
            r = entity.get('radius')
            if c and r is not None:
                xs = [c['x'] - r, c['x'] + r]
                ys = [c['y'] - r, c['y'] + r]
        elif etype == 'ELLIPSE':
            c = entity.get('center')
            if c:
                # Approximate bbox for ellipse using major axis length
                # This is a loose bound
                major = entity.get('majorAxis', {'x': 100, 'y': 0})
                major_len = (major['x']**2 + major['y']**2)**0.5
                xs = [c['x'] - major_len, c['x'] + major_len]
                ys = [c['y'] - major_len, c['y'] + major_len]
        elif etype == 'SOLID':
            pts = self.get_solid_points(entity)
            if pts:
                xs = [p['x'] for p in pts]
                ys = [p['y'] for p in pts]
        
        if not xs: return None
        return min(xs), min(ys), max(xs), max(ys)

    def bbox_intersects(self, bbox1, bbox2):
        if not bbox1 or not bbox2: return True
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    def check_visibility(self, xs, ys, bbox):
        """Check if a set of points is visible within the bbox."""
        if not bbox: return True
        min_x, min_y, max_x, max_y = bbox
        # If all points are to the left, right, above, or below the bbox, it's invisible
        if np.max(xs) < min_x or np.min(xs) > max_x or np.max(ys) < min_y or np.min(ys) > max_y:
            return False
        return True

    def check_circle_visibility(self, cx, cy, r, bbox):
        """Check if a circle is visible within the bbox."""
        if not bbox: return True
        min_x, min_y, max_x, max_y = bbox
        if cx + r < min_x or cx - r > max_x or cy + r < min_y or cy - r > max_y:
            return False
        return True

    def draw_entity(self, ax, entity, transform, view_bbox, depth=0, points_per_unit=1.0, text_size_min=5, text_size_max=16):
        if depth > 30: return # Prevent infinite recursion
        
        etype = entity.get('type')
        color = self.get_entity_color(entity)
        
        if etype == 'INSERT' or etype == 'DIMENSION':
            if etype == 'DIMENSION':
                name = entity.get('block') # Dimensions use 'block' attribute for anonymous block
            else:
                name = entity.get('name')
                
            if not name or name not in self.blocks:
                return
            
            # Compute new transform
            # For DIMENSION, usually insert at 0,0 with identity scale unless specified
            # Dimensions are usually defined in WCS coordinates inside their anonymous block
            if etype == 'DIMENSION':
                p = entity.get('position') or entity.get('insertPoint') or {'x': 0, 'y': 0}
                sx = entity.get('xScale', 1)
                sy = entity.get('yScale', 1)
                rot = entity.get('rotation', 0)
            else:
                p = entity.get('position', {'x': 0, 'y': 0})
                sx = entity.get('xScale', 1)
                sy = entity.get('yScale', 1)
                rot = entity.get('rotation', 0)
            
            new_matrix = self._get_transform_matrix(p['x'], p['y'], sx, sy, rot)
            combined_transform = transform @ new_matrix
            
            # Retrieve block definition
            block_def = self.blocks[name]
            # Block definition might be a list of entities or a dict with 'entities'
            block_entities = []
            if isinstance(block_def, list):
                block_entities = block_def
            elif isinstance(block_def, dict):
                block_entities = block_def.get('entities', [])
                
            for child in block_entities:
                # Inherit layer color if child is ByBlock (color=0)
                # But get_entity_color handles ByBlock by returning white?
                # We should ideally pass down parent properties.
                # For now, simplistic recursion.
                self.draw_entity(ax, child, combined_transform, view_bbox, depth+1, points_per_unit=points_per_unit, text_size_min=text_size_min, text_size_max=text_size_max)
                
        elif etype == 'LINE':
            line_pts = self.get_line_points(entity)
            if line_pts:
                pts = np.array(line_pts)
                t_pts = self.transform_points(pts, transform)
                
                # Check visibility
                if not self.check_visibility(t_pts[:, 0], t_pts[:, 1], view_bbox):
                    return
                
                ax.plot(t_pts[:, 0], t_pts[:, 1], color=color, linewidth=self.get_entity_lineweight(entity, default=0.5), alpha=0.8)
                
        elif etype == 'LWPOLYLINE':
            verts = entity.get('vertices', [])
            if verts:
                pts = np.array([[v['x'], v['y']] for v in verts])
                t_pts = self.transform_points(pts, transform)
                
                is_closed = bool(entity.get('closed', False) or entity.get('shape', False))
                if is_closed:
                    t_pts = np.vstack([t_pts, t_pts[0]])
                
                # Check visibility
                if not self.check_visibility(t_pts[:, 0], t_pts[:, 1], view_bbox):
                    return
                
                ax.plot(t_pts[:, 0], t_pts[:, 1], color=color, linewidth=self.get_entity_lineweight(entity, default=0.5), alpha=0.8)

        elif etype == 'CIRCLE':
            c = entity.get('center')
            r = entity.get('radius')
            if c and r:
                # Transform center
                tc = self.transform_points([[c['x'], c['y']]], transform)[0]
                # Scale radius (approximation using transform scale)
                # Extract scale from matrix column lengths
                scale_x = np.linalg.norm(transform[:2, 0])
                tr = r * scale_x 
                
                # Check visibility
                if not self.check_circle_visibility(tc[0], tc[1], tr, view_bbox):
                    return
                
                circle = patches.Circle(tc, tr, color=color, fill=False, linewidth=self.get_entity_lineweight(entity, default=0.5))
                ax.add_patch(circle)

        elif etype == 'ARC':
            c = entity.get('center')
            r = entity.get('radius')
            start = entity.get('startAngle')
            end = entity.get('endAngle')
            if c and r:
                tc = self.transform_points([[c['x'], c['y']]], transform)[0]
                scale_x = np.linalg.norm(transform[:2, 0])
                tr = r * scale_x
                
                # Check visibility
                if not self.check_circle_visibility(tc[0], tc[1], tr, view_bbox):
                    return
                
                # Rotation from matrix
                mat_rot = np.arctan2(transform[1, 0], transform[0, 0])
                mat_rot_deg = np.degrees(mat_rot)
                
                if start is not None and end is not None:
                    # Adjust angles by rotation
                    # start/end are in radians usually in DXF/JSON? 
                    # JSON often has degrees for angles? 
                    # Standard DXF is degrees. Let's assume degrees.
                    arc = patches.Arc(tc, 2*tr, 2*tr, angle=mat_rot_deg, 
                                      theta1=to_degrees(start), theta2=to_degrees(end), 
                                      color=color, linewidth=self.get_entity_lineweight(entity, default=0.5))
                    ax.add_patch(arc)

        elif etype == 'HATCH':
            # Handle boundary loops
            loops = entity.get('boundaryLoops', [])
            
            # Collect all paths for this hatch
            all_codes = []
            all_verts = []
            
            for loop in loops:
                # Loop can have 'polyline' or 'edges'
                poly = loop.get('polyline')
                edges = loop.get('edges')
                
                loop_verts = []
                
                if poly:
                    verts = poly.get('vertices', [])
                    if verts:
                        loop_verts = [[v['x'], v['y']] for v in verts]
                
                elif edges:
                    # Construct loop from edges
                    # Assumes edges are contiguous
                    for i, edge in enumerate(edges):
                        edge_type = edge.get('type')
                        
                        # Handle both integer and string types for edges
                        is_line = (edge_type == 1 or str(edge_type).lower() == 'line')
                        is_arc = (edge_type == 2 or str(edge_type).lower() == 'arc')
                        is_ellipse = (edge_type == 3 or str(edge_type).lower() == 'ellipse')
                        is_spline = (edge_type == 4 or str(edge_type).lower() == 'spline')

                        if is_line: 
                            start = edge.get('start')
                            end = edge.get('end')
                            if start and end:
                                if i == 0:
                                    loop_verts.append([start['x'], start['y']])
                                loop_verts.append([end['x'], end['y']])
                        
                        elif is_arc: 
                            # Approximate arc with line segments
                            cx, cy = edge.get('center', {}).get('x'), edge.get('center', {}).get('y')
                            r = edge.get('radius')
                            start_ang = edge.get('startAngle')
                            end_ang = edge.get('endAngle')
                            ccw = edge.get('counterClockwise', True)
                            
                            if cx is not None and r is not None:
                                start_rad, end_rad = self._edge_angles_to_radians(start_ang, end_ang, ccw)
                                num_steps = max(2, int(abs(end_rad - start_rad) * r / 0.1))
                                num_steps = min(num_steps, 80)
                                theta = np.linspace(start_rad, end_rad, num_steps)
                                x = cx + r * np.cos(theta)
                                y = cy + r * np.sin(theta)
                                
                                pts = np.column_stack([x, y])
                                if i == 0:
                                    loop_verts.extend(pts.tolist())
                                else:
                                    loop_verts.extend(pts.tolist())
                        
                        elif is_ellipse:
                            center = edge.get('center') or {}
                            major = edge.get('majorAxisEndPoint') or edge.get('majorAxis') or {}
                            ratio = edge.get('minorMajorRatio')
                            if ratio is None:
                                ratio = edge.get('ratio')
                            cx, cy = center.get('x'), center.get('y')
                            mx, my = major.get('x'), major.get('y')
                            start_ang = edge.get('startAngle', 0.0)
                            end_ang = edge.get('endAngle', 2 * np.pi)
                            ccw = edge.get('counterClockwise', True)
                            if cx is not None and cy is not None and mx is not None and my is not None and ratio is not None:
                                major_len = float(np.hypot(mx, my))
                                if major_len > 0:
                                    minor_len = major_len * float(ratio)
                                    phi = np.arctan2(my, mx)
                                    start_rad, end_rad = self._edge_angles_to_radians(start_ang, end_ang, ccw)
                                    num_steps = max(8, int(abs(end_rad - start_rad) * major_len / 0.2))
                                    num_steps = min(num_steps, 120)
                                    theta = np.linspace(start_rad, end_rad, num_steps)
                                    x = cx + major_len * np.cos(theta) * np.cos(phi) - minor_len * np.sin(theta) * np.sin(phi)
                                    y = cy + major_len * np.cos(theta) * np.sin(phi) + minor_len * np.sin(theta) * np.cos(phi)
                                    pts = np.column_stack([x, y]).tolist()
                                    if i == 0:
                                        loop_verts.extend(pts)
                                    else:
                                        loop_verts.extend(pts)
                        
                        elif is_spline:
                            pts = self.approximate_spline_points(edge, max_points=120)
                            if pts:
                                if i == 0:
                                    loop_verts.extend(pts)
                                else:
                                    loop_verts.extend(pts)

                
                if loop_verts:
                    pts = np.array(loop_verts)
                    t_pts = self.transform_points(pts, transform)
                    
                    # Create Path codes
                    # MOVETO first, LINETO rest, CLOSEPOLY
                    if len(t_pts) > 0:
                        all_codes.append(mpath.Path.MOVETO)
                        all_verts.append(t_pts[0])
                        
                        for p in t_pts[1:]:
                            all_codes.append(mpath.Path.LINETO)
                            all_verts.append(p)
                        
                        all_codes.append(mpath.Path.CLOSEPOLY)
                        all_verts.append(t_pts[0]) 
            
            if all_verts:
                path = mpath.Path(all_verts, all_codes)
                
                if entity.get('isSolid'):
                    patch = patches.PathPatch(path, facecolor=color, alpha=0.5, linewidth=0)
                    ax.add_patch(patch)
                else:
                    # Hatching
                    pattern_name = entity.get('patternName', '').upper()
                    hatch_style = '///' # Default dense diagonal
                    if 'ANSI31' in pattern_name: hatch_style = '///'
                    elif 'ANSI37' in pattern_name: hatch_style = 'x'
                    elif 'SOLID' in pattern_name: hatch_style = None
                    elif 'CONCRETE' in pattern_name: hatch_style = '.'
                    elif 'AR-CONC' in pattern_name: hatch_style = 'o'
                    
                    # For non-solid, we want the boundary AND the hatch
                    # facecolor='none' makes it transparent
                    # edgecolor sets boundary color (and hatch color?)
                    # In matplotlib, hatch color follows edgecolor.
                    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                                            hatch=hatch_style, linewidth=self.get_entity_lineweight(entity, default=0.2), alpha=0.7)
                    ax.add_patch(patch)

        elif etype == 'POINT':
            p = entity.get('position')
            if p:
                tp = self.transform_points([[p['x'], p['y']]], transform)[0]
                # Draw as a small circle
                circle = patches.Circle(tp, radius=0.5, color=color, fill=True) # Radius in drawing units?
                ax.add_patch(circle)

        elif etype == 'SOLID':
            pts = self.get_solid_points(entity)
            if len(pts) >= 3:
                t_pts = self.transform_points([[pt['x'], pt['y']] for pt in pts], transform)
                
                # Check visibility
                if not self.check_visibility(t_pts[:, 0], t_pts[:, 1], view_bbox):
                    return
                
                poly = patches.Polygon(t_pts, closed=True, fill=True, color=color, alpha=0.9)
                ax.add_patch(poly)

        elif etype == 'ELLIPSE':
            c = entity.get('center')
            major = entity.get('majorAxis')
            ratio = entity.get('ratio')
            start = entity.get('startAngle')
            end = entity.get('endAngle')
            
            if c and major and ratio:
                tc = self.transform_points([[c['x'], c['y']]], transform)[0]
                
                # Calculate major axis length and angle
                major_len = (major['x']**2 + major['y']**2)**0.5
                scale_x = np.linalg.norm(transform[:2, 0])
                r_max = major_len * scale_x
                
                # Check visibility (using max radius)
                if not self.check_circle_visibility(tc[0], tc[1], r_max, view_bbox):
                    return
                
                minor_len = major_len * ratio
                
                angle = np.degrees(np.arctan2(major['y'], major['x']))
                
                # Apply transform rotation
                mat_rot = np.arctan2(transform[1, 0], transform[0, 0])
                total_angle = angle + np.degrees(mat_rot)
                
                scale_x = np.linalg.norm(transform[:2, 0])
                
                # Handle start/end angles if partial ellipse (Arc)
                start_deg = to_degrees(start) if start is not None else None
                end_deg = to_degrees(end) if end is not None else None
                if start_deg is not None and end_deg is not None and abs(end_deg - start_deg) < 359.999:
                    # Elliptical Arc
                    # Matplotlib Arc takes width/height, angle, theta1, theta2
                    # Note: Arc theta1/theta2 are relative to the ellipse axes, not global
                    arc = patches.Arc(tc, 2*major_len*scale_x, 2*minor_len*scale_x, angle=total_angle,
                                      theta1=start_deg, theta2=end_deg,
                                      color=color, linewidth=self.get_entity_lineweight(entity, default=0.5))
                    ax.add_patch(arc)
                else:
                    # Full Ellipse
                    ell = patches.Ellipse(tc, 2*major_len*scale_x, 2*minor_len*scale_x, angle=total_angle,
                                          color=color, fill=False, linewidth=self.get_entity_lineweight(entity, default=0.5))
                    ax.add_patch(ell)

        elif etype == 'POLYLINE':
            # Handle old POLYLINE entity (2D or 3D)
            verts = entity.get('vertices', [])
            if verts:
                pts = np.array([[v['x'], v['y']] for v in verts])
                t_pts = self.transform_points(pts, transform)
                
                # Check visibility
                if not self.check_visibility(t_pts[:, 0], t_pts[:, 1], view_bbox):
                    return
                
                is_closed = bool(entity.get('closed', False) or entity.get('shape', False))
                if is_closed:
                    t_pts = np.vstack([t_pts, t_pts[0]])
                
                ax.plot(t_pts[:, 0], t_pts[:, 1], color=color, linewidth=self.get_entity_lineweight(entity, default=0.5), alpha=0.8)

        elif etype == 'SPLINE':
            pts = self.approximate_spline_points(entity, max_points=240)
            if len(pts) >= 2:
                t_pts = self.transform_points(np.array(pts), transform)
                
                # Check visibility
                if not self.check_visibility(t_pts[:, 0], t_pts[:, 1], view_bbox):
                    return
                
                ax.plot(t_pts[:, 0], t_pts[:, 1], color=color, linewidth=self.get_entity_lineweight(entity, default=0.5), alpha=0.8)

        
        elif etype in ['TEXT', 'MTEXT', 'ATTRIB', 'ATTDEF']:
            p = entity.get('insertPoint') or entity.get('position') or entity.get('startPoint')
            if not p:
                s = entity.get('startPoint')
                e = entity.get('endPoint')
                if s and e and 'x' in s and 'y' in s and 'x' in e and 'y' in e:
                    p = {'x': 0.5 * (s['x'] + e['x']), 'y': 0.5 * (s['y'] + e['y'])}
            raw_text = entity.get('text', '')
            # Decode text
            text = decode_cad_unicode(raw_text)
            
            if p and text:
                tp = self.transform_points([[p['x'], p['y']]], transform)[0]
                
                # Check visibility
                if not self.check_visibility([tp[0]], [tp[1]], view_bbox):
                    return
                
                # Calculate rotation
                rot = to_degrees(entity.get('rotation', 0))
                mat_rot = np.arctan2(transform[1, 0], transform[0, 0])
                total_rot = rot + np.degrees(mat_rot)
                
                # Calculate dynamic font size
                # Default height 300 if missing (common for mm CAD drawings)
                h = entity.get('textHeight') or entity.get('height') or 300
                
                # Apply transformation scale to height
                scale_x = np.linalg.norm(transform[:2, 0])
                scaled_h = h * scale_x
                
                # Convert to points, then nudge larger for on-screen/SVG readability
                fontsize = scaled_h * points_per_unit * TEXT_FONT_SIZE_SCALE
                
                # Ensure minimum readability if needed, but respect scale primarily
                # For high-fidelity rendering, we want accurate size.
                # But if it's < 1pt, it's invisible.
                if fontsize < 0.5:
                    return # Skip invisible text
                
                fontsize = max(min(fontsize, text_size_max), text_size_min)
                
                # Use the loaded font properties
                ax.text(tp[0], tp[1], text, color=color, fontsize=fontsize, alpha=0.9, 
                        fontproperties=self.font_prop, rotation=total_rot, 
                        ha='left', va='bottom') # Default alignment
                        
        elif etype == '3DFACE':
            # 3DFACE has 4 corners (or 3 if two are same)
            # keys: 1stPoint, 2ndPoint, 3rdPoint, 4thPoint
            pts = []
            for k in ['1stPoint', '2ndPoint', '3rdPoint', '4thPoint']:
                pt = entity.get(k)
                if pt:
                    pts.append([pt['x'], pt['y']])
            
            if len(pts) >= 3:
                t_pts = self.transform_points(pts, transform)
                # Draw as polygon
                poly = patches.Polygon(t_pts, closed=True, fill=False, edgecolor=color, linewidth=self.get_entity_lineweight(entity, default=0.5))
                ax.add_patch(poly)

def main():
    parser = argparse.ArgumentParser(description='Visualize CAD JSON with Block Support')
    parser.add_argument('file_path')
    parser.add_argument('--bbox', type=str, help='min_x,min_y,max_x,max_y')
    parser.add_argument('--output', default='output_v2.svg')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print("File not found.")
        return
        
    viz = CADVisualizer(args.file_path)
    
    bbox = None
    if args.bbox:
        try:
            bbox = tuple(map(float, args.bbox.split(',')))
        except:
            print("Invalid bbox")
            return
            
    viz.render(args.output, bbox)

if __name__ == "__main__":
    main()
