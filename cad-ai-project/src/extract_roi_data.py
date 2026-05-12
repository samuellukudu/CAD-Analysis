import json
import numpy as np
import argparse
import sys
import tiktoken
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_transform_matrix(tx, ty, sx, sy, rot):
    c = np.cos(np.radians(rot))
    s = np.sin(np.radians(rot))
    return np.array([
        [sx*c, -sy*s, tx],
        [sx*s,  sy*c, ty],
        [   0,     0,  1]
    ])

def transform_points(points, matrix):
    pts = np.array(points)
    if pts.shape[0] == 0:
        return []
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    transformed = (matrix @ pts_h.T).T
    return transformed[:, :2].tolist()

def get_block_points(block_name, blocks, depth=0):
    if depth > 2: # Limit recursion depth
        return []
    
    block_def = blocks.get(block_name)
    if not block_def:
        return []
        
    points = []
    entities = []
    if isinstance(block_def, list):
        entities = block_def
    elif isinstance(block_def, dict):
        entities = block_def.get('entities', [])
        
    for entity in entities:
        pts = get_entity_points(entity, blocks, depth + 1)
        points.extend(pts)
        
    return points

MAX_ENTITY_DIM = 100000

def get_entity_points(entity, blocks=None, depth=0):
    points = []
    etype = entity.get('type')
    
    if etype == 'LINE':
        s = entity.get('start')
        e = entity.get('end')
        if s and e:
            points.extend([(s['x'], s['y']), (e['x'], e['y'])])
            
    elif etype == 'LWPOLYLINE':
        verts = entity.get('vertices', [])
        for v in verts:
            if isinstance(v, dict) and 'x' in v and 'y' in v:
                points.append((v['x'], v['y']))
                    
    elif etype == 'INSERT':
        p = entity.get('position') or entity.get('insertPoint')
        added_block_points = False
        if p:
            if blocks and depth < 2:
                name = entity.get('name')
                if name in blocks:
                    local_points = get_block_points(name, blocks, depth + 1)
                    if local_points:
                        sx = entity.get('xScale', 1)
                        sy = entity.get('yScale', 1)
                        rot = entity.get('rotation', 0)
                        matrix = get_transform_matrix(p['x'], p['y'], sx, sy, rot)
                        # Downsample if too many points to speed up bounding box calc
                        if len(local_points) > 20:
                             step = len(local_points) // 20
                             local_points = local_points[::step]
                        t_points = transform_points(local_points, matrix)
                        points.extend(t_points)
                        added_block_points = True
            
            if not added_block_points:
                points.append((p['x'], p['y']))

    elif etype in ['CIRCLE', 'ARC', 'ATTRIB', 'POINT']:
        p = entity.get('insertPoint') or entity.get('position') or entity.get('center') or entity.get('startPoint')
        if p:
            points.append((p['x'], p['y']))
            
    elif etype == 'SOLID':
        for k in ['first', 'second', 'third', 'fourth']:
            p = entity.get(k)
            if p: points.append((p['x'], p['y']))

    elif etype == 'ELLIPSE':
        c = entity.get('center')
        major = entity.get('majorAxis')
        if c and major:
            points.append((c['x'], c['y']))
            points.append((c['x'] + major['x'], c['y'] + major['y']))
            points.append((c['x'] - major['x'], c['y'] - major['y']))
            ratio = entity.get('ratio', 1.0)
            mx, my = -major['y'] * ratio, major['x'] * ratio
            points.append((c['x'] + mx, c['y'] + my))
            points.append((c['x'] - mx, c['y'] - my))

    elif etype == 'SPLINE':
        for k in ['controlPoints', 'fitPoints']:
            for p in entity.get(k, []):
                points.append((p['x'], p['y']))

    elif etype == 'POLYLINE':
        for v in entity.get('vertices', []):
            if isinstance(v, dict) and 'x' in v and 'y' in v:
                points.append((v['x'], v['y']))

    elif etype == 'HATCH':
        for loop in entity.get('boundaryLoops', []):
            poly = loop.get('polyline')
            if poly:
                for v in poly.get('vertices', []):
                    if isinstance(v, dict) and 'x' in v and 'y' in v:
                        points.append((v['x'], v['y']))
            for edge in loop.get('edges', []):
                 for k in ['start', 'end', 'center']:
                     p = edge.get(k)
                     if p: points.append((p['x'], p['y']))
                 for p in edge.get('controlPoints', []):
                     points.append((p['x'], p['y']))

    elif etype == 'DIMENSION':
        keys = ['anchorPoint', 'middleOfText', 'linearOrAngularPoint1', 'linearOrAngularPoint2', 'textPoint', 'defPoint', 'defPoint1', 'defPoint2']
        for k in keys:
            p = entity.get(k)
            if p and 'x' in p and 'y' in p:
                points.append((p['x'], p['y']))
                
    if not points and 'x' in entity and 'y' in entity:
        points.append((entity['x'], entity['y']))
    
    if depth == 0 and points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if (max(xs) - min(xs) > MAX_ENTITY_DIM) or (max(ys) - min(ys) > MAX_ENTITY_DIM):
            return []

    return points

def get_entity_bbox(entity, blocks=None):
    points = get_entity_points(entity, blocks)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_intersects(bbox1, bbox2):
    if not bbox1 or not bbox2: return False
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def minify_for_llm(obj, decimals=1):
    """
    Recursively cleans CAD entity data to reduce token count for LLMs:
    - Rounds floats to `decimals`
    - Removes visual and irrelevant properties
    """
    keys_to_drop = {
        # Visual & Styling
        'color', 'colorIndex', 'lineweight', 'linetype', 'thickness', 'extrusion', 'visible', 'lineWeight', 'textStyle', 'lineTypeScale',
        # Unnecessary 3D/CAD flags
        'z', 'zScale', 'hidden', 'constant', 'verificationRequired', 'preset', 'lockPositionFlag', 
        'attachmentPoint', 'horizontalJustification', 'endPoint', 'textHeight', 'shape', 'hasContinuousLinetypePattern', 'width',
        # Visual scaling properties (useless for semantic layout reasoning)
        'scale', 'xScale', 'yScale',
        # Bloated custom dictionary data from CAD plugins
        'xdata', 'xdata_dict'
    }
    
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, str):
        # Clean up CAD file prefixes (e.g. "230303生产综合楼_t8$0$COLUMN" -> "COLUMN")
        # This preserves the semantic layer/name but removes the repetitive file prefix
        if '$0$' in obj:
            return obj.split('$0$')[-1]
        return obj
    # 2. Re-key specific properties to single characters for massive token savings
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in keys_to_drop:
                continue
                
            # Drop ModelSpace root handles (usually '0', '2', or '3') as they provide no relational value
            if k == 'ownerHandle' and v in ('0', '2', '3'):
                continue
                
            # Drop rotation if it's 0 or 360 (default state)
            if k == 'rotation' and v in (0, 0.0, 360, 360.0):
                continue
                
            # Dictionary key minification
            new_k = k
            if k == 'type': new_k = 't'
            elif k == 'layer': new_k = 'l'
            elif k == 'text': new_k = 'txt'
            elif k == 'handle': new_k = 'id'
            elif k == 'ownerHandle': new_k = 'pid'  # parent id
            elif k == 'vertices': new_k = 'v'
            elif k == 'startPoint': new_k = 'p'
            elif k == 'position': new_k = 'p'
            elif k == 'rotation': new_k = 'r'
            elif k == 'name': new_k = 'n'
            elif k == 'tag': new_k = 'tg'
            
            cleaned[new_k] = minify_for_llm(v, decimals)
            
        # Point compression: {"x": 1.2, "y": 3.4, "bulge": 0} -> [1.2, 3.4] or [1.2, 3.4, bulge]
        if 'x' in cleaned and 'y' in cleaned and set(cleaned.keys()).issubset({'x', 'y', 'bulge'}):
            x = cleaned['x']
            y = cleaned['y']
            bulge = cleaned.get('bulge', 0)
            if bulge != 0:
                return [x, y, bulge]
            return [x, y]
            
        return cleaned
    elif isinstance(obj, list):
        return [minify_for_llm(item, decimals) for item in obj]
    return obj

def extract_entities_in_bbox(file_path, roi_bbox, llm_mode=False, text_only=False):
    """
    Extracts all entities from a CAD JSON file that intersect with the given bounding box.
    roi_bbox: tuple of (min_x, min_y, max_x, max_y)
    """
    data = load_json(file_path)
    entities = data.get('entities', [])
    blocks = data.get('blocks', {})
    
    # Entities that provide zero semantic value to an LLM and consume huge tokens
    LLM_IGNORED_TYPES = {'HATCH', 'SOLID', 'SPLINE', 'POINT'}
    TEXT_TYPES = {'TEXT', 'MTEXT', 'ATTRIB'}
    
    matched_entities = []
    for entity in entities:
        ent_type = entity.get('type')
        
        if text_only and ent_type not in TEXT_TYPES:
            continue
            
        if llm_mode:
            # Drop noisy entities
            if ent_type in LLM_IGNORED_TYPES:
                continue
            # Drop empty attributes or text
            if ent_type in TEXT_TYPES and not entity.get('text', '').strip():
                continue
                
            # For structural geometry, handles are useless since they don't have child attributes.
            # We only need handles for relational entities (INSERT, ATTRIB, TEXT, etc)
            if entity.get('type') in ('LWPOLYLINE', 'LINE', 'ARC', 'CIRCLE', 'POLYLINE'):
                entity.pop('handle', None)
                entity.pop('ownerHandle', None)
                
        ebbox = get_entity_bbox(entity, blocks)
        if ebbox and bbox_intersects(ebbox, roi_bbox):
            if llm_mode:
                matched_entities.append(minify_for_llm(entity))
            else:
                matched_entities.append(entity)
            
    return matched_entities

def count_tokens(data, model="gpt-4o"):
    """
    Counts the number of tokens in the JSON representation of the data.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    json_string = json.dumps(data, ensure_ascii=False)
    return len(encoding.encode(json_string))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract entities within a bounding box from a CAD JSON file.')
    parser.add_argument('--file', required=True, help='Path to the CAD JSON file')
    parser.add_argument('--bbox', required=True, help='Bounding box in format min_x,min_y,max_x,max_y')
    parser.add_argument('--output', help='Output JSON file to save the entities (optional)')
    parser.add_argument('--llm-mode', action='store_true', help='Minify data and drop noise entities (HATCH, etc) to save LLM tokens')
    parser.add_argument('--text-only', action='store_true', help='Only extract text entities (TEXT, MTEXT, ATTRIB)')
    parser.add_argument('--count-tokens', action='store_true', help='Count and print the number of tokens in the extracted data')
    
    args = parser.parse_args()
    
    try:
        bbox_parts = list(map(float, args.bbox.split(',')))
        if len(bbox_parts) != 4:
            raise ValueError
        roi_bbox = tuple(bbox_parts)
    except ValueError:
        print("Error: --bbox must be in format min_x,min_y,max_x,max_y", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading and filtering entities in bbox {roi_bbox}...", file=sys.stderr)
    matched = extract_entities_in_bbox(args.file, roi_bbox, llm_mode=args.llm_mode, text_only=args.text_only)
    print(f"Found {len(matched)} matching entities.", file=sys.stderr)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.llm_mode:
                # Provide key mapping at the top for LLM mode
                output_data = {
                    "key_map": {
                        "t": "type", "l": "layer", "txt": "text", "id": "handle", "pid": "ownerHandle",
                        "v": "vertices", "p": "startPoint/position", "r": "rotation", "n": "name", "tg": "tag"
                    },
                    "entities": matched
                }
                # LLM mode uses zero indentation to save whitespace tokens
                json.dump(output_data, f, ensure_ascii=False, separators=(',', ':'))
            else:
                json.dump(matched, f, ensure_ascii=False, indent=2)
        print(f"Saved matched entities to {args.output}", file=sys.stderr)
    else:
        # Just print a summary if no output file is specified, but also output the raw JSON to stdout so it can be piped or viewed
        try:
            if args.llm_mode:
                output_data = {
                    "key_map": {
                        "t": "type", "l": "layer", "txt": "text", "id": "handle", "pid": "ownerHandle",
                        "v": "vertices", "p": "startPoint/position", "r": "rotation", "n": "name", "tg": "tag"
                    },
                    "entities": matched
                }
                # Use zero indentation and tight separators
                print(json.dumps(output_data, ensure_ascii=False, separators=(',', ':')))
            else:
                print(json.dumps(matched, ensure_ascii=False, indent=2))
        except BrokenPipeError:
            # Handle early pipe closures (like when using | head)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            
    if args.count_tokens:
        token_count = count_tokens(matched)
        print(f"\n=========================================", file=sys.stderr)
        print(f"Estimated token count (gpt-4o): {token_count}", file=sys.stderr)
        print(f"=========================================\n", file=sys.stderr)
