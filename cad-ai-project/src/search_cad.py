import argparse
import json
import math
import os
import sys
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cad_utils import (
    TEXT_ENTITY_TYPES,
    extract_text_from_entity,
    get_block_entities,
    get_entity_position,
    normalize_text,
)

TEXT_ENTITIES = set(TEXT_ENTITY_TYPES)

def get_entity_text(entity: Dict[str, Any]) -> str:
    """Extract text content from an entity."""
    return extract_text_from_entity(entity)

def get_entity_location(entity: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Get a representative point for the entity using shared CAD utils."""
    pos = get_entity_position(entity)
    if pos:
        return pos
    if 'vertices' in entity and len(entity['vertices']) > 0:
        return entity['vertices'][0]
    if 'start' in entity:
        return entity['start']
    return None


def _transform_point(px: float, py: float, insert_entity: Dict[str, Any]) -> Tuple[float, float]:
    """Apply an INSERT's transform to a local point."""
    ip = get_entity_position(insert_entity) or {'x': 0, 'y': 0}
    tx, ty = ip.get('x', 0), ip.get('y', 0)
    sx = insert_entity.get('xScale', 1) or 1
    sy = insert_entity.get('yScale', 1) or 1
    rot = insert_entity.get('rotation', 0) or 0

    if abs(rot) > 2 * math.pi + 1e-6:
        rot = math.radians(rot)

    c = math.cos(rot)
    s = math.sin(rot)
    wx = tx + sx * c * px - sy * s * py
    wy = ty + sx * s * px + sy * c * py
    return wx, wy


def _world_position(entity: Dict[str, Any], transform_stack: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Resolve entity world position through nested INSERT transforms."""
    pos = get_entity_position(entity)
    if not pos:
        return None

    wx, wy = pos.get('x', 0), pos.get('y', 0)
    for parent in reversed(transform_stack):
        wx, wy = _transform_point(wx, wy, parent)
    return {'x': wx, 'y': wy}


def _iter_entities(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    blocks: Dict[str, Any],
    transform_stack: Optional[List[Dict[str, Any]]] = None,
    depth: int = 0,
    max_depth: int = 5,
):
    """Yield (entity, world_location) from top-level and nested block entities."""
    if depth > max_depth:
        return

    if transform_stack is None:
        transform_stack = []

    if isinstance(data, dict):
        entities = data.get('entities', [])
    elif isinstance(data, list):
        entities = data
    else:
        return

    for ent in entities:
        yield ent, _world_position(ent, transform_stack)

        etype = ent.get('type', '')
        if etype in ('INSERT', 'DIMENSION'):
            name = ent.get('name') if etype == 'INSERT' else (ent.get('block') or ent.get('name'))
            if name and name in blocks:
                block_ents = get_block_entities(blocks[name])
                yield from _iter_entities(
                    block_ents,
                    blocks,
                    transform_stack=transform_stack + [ent],
                    depth=depth + 1,
                    max_depth=max_depth,
                )


def _collect_json_files(paths: List[str]) -> List[str]:
    """Collect JSON files from files and directories."""
    json_files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for file_name in files:
                    if file_name.endswith('.json'):
                        json_files.append(os.path.join(root, file_name))
        elif os.path.isfile(p) and p.endswith('.json'):
            json_files.append(p)
    return sorted(set(json_files))


def _prepare_queries(queries: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (original_query, normalized_lower_query) with empty queries removed.
    """
    prepared = []
    for q in queries:
        normalized = normalize_text(q or '').strip()
        if normalized:
            prepared.append((q, normalized.lower()))
    return prepared


def _build_multiline_candidates(text_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge nearby text records so multi-line annotations can be matched as one unit.
    """
    if not text_records:
        return []

    heights = [
        rec.get('entity', {}).get('textHeight', 0)
        for rec in text_records
        if isinstance(rec.get('entity', {}).get('textHeight'), (int, float))
        and rec.get('entity', {}).get('textHeight', 0) > 0
    ]
    median_h = statistics.median(heights) if heights else 80.0
    x_threshold = max(300.0, median_h * 3.0)
    line_gap_threshold = max(250.0, median_h * 2.5)

    sorted_records = sorted(
        text_records,
        key=lambda rec: (
            round((rec['location']['x'] if rec.get('location') else 0.0) / x_threshold),
            -(rec['location']['y'] if rec.get('location') else 0.0),
        ),
    )

    groups = []
    for rec in sorted_records:
        loc = rec.get('location')
        if not loc:
            continue

        rx, ry = loc.get('x', 0.0), loc.get('y', 0.0)
        attached = False
        for group in groups:
            if abs(group['avg_x'] - rx) > x_threshold:
                continue
            if group['last_y'] - ry > line_gap_threshold:
                continue
            if ry > group['last_y'] + line_gap_threshold:
                continue

            group['records'].append(rec)
            group['avg_x'] = (group['avg_x'] * (len(group['records']) - 1) + rx) / len(group['records'])
            group['last_y'] = ry
            attached = True
            break

        if not attached:
            groups.append({
                'records': [rec],
                'avg_x': rx,
                'last_y': ry,
            })

    merged = []
    for group in groups:
        records = sorted(group['records'], key=lambda r: -(r['location'].get('y', 0.0)))
        if len(records) < 2:
            continue

        merged_text = ' '.join(r['text'] for r in records if r.get('text')).strip()
        if not merged_text:
            continue

        merged.append({
            'type': 'MULTILINE_TEXT_GROUP',
            'handle': records[0]['entity'].get('handle'),
            'group_handles': [r['entity'].get('handle') for r in records if r['entity'].get('handle')],
            'group_size': len(records),
            'text': merged_text,
            'normalized_text_lower': normalize_text(merged_text).lower(),
            'location': records[0].get('location'),
        })

    return merged

def get_entity_bbox(entity: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounding box for an entity.
    Returns (min_x, min_y, max_x, max_y).
    """
    etype = entity.get('type')
    
    if etype == 'LINE':
        s = entity.get('startPoint')
        e = entity.get('endPoint')
        if s and e:
            return (min(s['x'], e['x']), min(s['y'], e['y']), max(s['x'], e['x']), max(s['y'], e['y']))
            
    elif etype == 'LWPOLYLINE':
        verts = entity.get('vertices', [])
        if verts:
            xs = [v['x'] for v in verts]
            ys = [v['y'] for v in verts]
            return (min(xs), min(ys), max(xs), max(ys))
            
    elif etype in TEXT_ENTITIES:
        # Approximate bbox
        ins = entity.get('startPoint') or entity.get('insertionPoint')
        if ins:
            height = entity.get('textHeight', 0)
            width = entity.get('width', 0)
            
            # If width is not explicit, estimate from text length
            if width == 0:
                text = entity.get('text', '')
                # Rough estimate: char width ~ 0.7 * height
                width = len(text) * height * 0.7
            
            # Assuming simple orientation (rotation=0) for basic containment check
            # TODO: Handle rotation if needed for precise bbox
            return (ins['x'], ins['y'], ins['x'] + width, ins['y'] + height)

    elif etype == 'INSERT':
        ins = entity.get('insertionPoint')
        if ins:
            # Treat INSERT as a point for now, or very small box
            return (ins['x'], ins['y'], ins['x'], ins['y'])
            
    elif etype == 'CIRCLE':
        c = entity.get('center')
        r = entity.get('radius', 0)
        if c:
            return (c['x'] - r, c['y'] - r, c['x'] + r, c['y'] + r)

    return None

def search_text(queries: List[str], paths: List[str]) -> List[Dict[str, Any]]:
    """
    Search for text in files.
    Uses grep to filter files, then parses JSON to find specific entities.
    Matches ANY of the queries.
    """
    results = []
    prepared_queries = _prepare_queries(queries)
    if not prepared_queries:
        return results

    # Parse all candidate JSON files, then perform text matching in-memory.
    # This allows multi-line reconstruction before matching.
    json_files = _collect_json_files(paths)

    for file_path in json_files:
        if not file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            blocks = data.get('blocks', {})
            text_records = []
            for ent, world_loc in _iter_entities(data, blocks):
                text = get_entity_text(ent)
                if not text:
                    continue
                text_records.append({
                    'entity': ent,
                    'text': text,
                    'normalized_text_lower': normalize_text(text).lower(),
                    'location': world_loc or get_entity_location(ent),
                })

            candidates = []
            for rec in text_records:
                candidates.append({
                    'type': rec['entity'].get('type'),
                    'handle': rec['entity'].get('handle'),
                    'text': rec['text'],
                    'normalized_text_lower': rec['normalized_text_lower'],
                    'location': rec['location'],
                })

            candidates.extend(_build_multiline_candidates(text_records))

            for cand in candidates:
                matches = [
                    original
                    for original, normalized_q in prepared_queries
                    if normalized_q in cand['normalized_text_lower']
                ]
                if matches:
                    results.append({
                        'file': file_path,
                        'type': cand.get('type'),
                        'handle': cand.get('handle'),
                        'match_text': cand['text'],
                        'matched_queries': matches,
                        'location': cand.get('location'),
                        'group_size': cand.get('group_size', 1),
                        'group_handles': cand.get('group_handles', []),
                    })

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    # Deduplicate in case single-line and merged candidates are text-identical.
    deduped = []
    seen = set()
    for item in results:
        loc = item.get('location') or {}
        key = (
            item.get('file'),
            item.get('type'),
            item.get('match_text'),
            tuple(item.get('matched_queries', [])),
            round(loc.get('x', 0.0), 3) if isinstance(loc, dict) else None,
            round(loc.get('y', 0.0), 3) if isinstance(loc, dict) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped

def search_spatial(
    target: Union[Tuple[float, float, float, float], Tuple[float, float]], 
    paths: List[str],
    mode: str = 'bbox'
) -> List[Dict[str, Any]]:
    """
    Search for entities intersecting a bbox or containing a point.
    target: (min_x, min_y, max_x, max_y) or (x, y)
    """
    results = []
    files_to_check = _collect_json_files(paths)

    for fp in files_to_check:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)

            blocks = data.get('blocks', {})
            for ent, world_loc in _iter_entities(data, blocks):
                ent_bbox = get_entity_bbox(ent)
                if not ent_bbox:
                    # Fallback to point bbox when no geometric bbox exists.
                    loc = world_loc or get_entity_location(ent)
                    if not loc:
                        continue
                    ent_bbox = (loc['x'], loc['y'], loc['x'], loc['y'])
                else:
                    # When nested in blocks, prefer world position point bbox for text.
                    if world_loc and ent.get('type') in TEXT_ENTITIES:
                        ent_bbox = (
                            world_loc['x'],
                            world_loc['y'],
                            world_loc['x'],
                            world_loc['y'],
                        )

                e_min_x, e_min_y, e_max_x, e_max_y = ent_bbox

                match = False
                if mode == 'bbox':
                    t_min_x, t_min_y, t_max_x, t_max_y = target
                    if (e_min_x <= t_max_x and e_max_x >= t_min_x and
                        e_min_y <= t_max_y and e_max_y >= t_min_y):
                        match = True
                elif mode == 'point':
                    px, py = target
                    if (e_min_x <= px <= e_max_x and
                        e_min_y <= py <= e_max_y):
                        match = True

                if match:
                    results.append({
                        'file': fp,
                        'type': ent.get('type'),
                        'handle': ent.get('handle'),
                        'location': world_loc or get_entity_location(ent),
                        'bbox': ent_bbox
                    })

        except Exception as e:
            print(f"Error processing {fp}: {e}", file=sys.stderr)

    return results

def main():
    parser = argparse.ArgumentParser(description="Search CAD JSON data.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Text search
    text_parser = subparsers.add_parser('text', help='Search for text')
    text_parser.add_argument('-q', '--query', type=str, action='append', required=True, help='Text(s) to search for')
    text_parser.add_argument('paths', nargs='+', help='Files or folders to search in')
    
    # Bbox search
    bbox_parser = subparsers.add_parser('bbox', help='Search by bounding box')
    bbox_parser.add_argument('min_x', type=float)
    bbox_parser.add_argument('min_y', type=float)
    bbox_parser.add_argument('max_x', type=float)
    bbox_parser.add_argument('max_y', type=float)
    bbox_parser.add_argument('paths', nargs='+', help='Files to search in')
    
    # Point search
    point_parser = subparsers.add_parser('point', help='Search by point')
    point_parser.add_argument('x', type=float)
    point_parser.add_argument('y', type=float)
    point_parser.add_argument('paths', nargs='+', help='Files to search in')
    
    args = parser.parse_args()
    
    results = []
    if args.command == 'text':
        results = search_text(args.query, args.paths)
    elif args.command == 'bbox':
        bbox = (args.min_x, args.min_y, args.max_x, args.max_y)
        results = search_spatial(bbox, args.paths, mode='bbox')
    elif args.command == 'point':
        point = (args.x, args.y)
        results = search_spatial(point, args.paths, mode='point')
        
    # Output results
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
