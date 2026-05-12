"""
Shared CAD utilities used by verify_roi.py, visualize_cad.py, and the python tool.
Centralizes text decoding, entity traversal, and position resolution so all tools
agree on how to interpret CAD JSON data.
"""

import re
import unicodedata


def decode_cad_unicode(text):
    r"""
    Decode AutoCAD MTEXT formatting codes and \U+XXXX Unicode escapes.

    Handles: \A...; \C...; \H...; \Q...; \T...; \W...; \f...;  (formatting)
             \S...;  (stacking, e.g. \S1^2; -> 1/2)
             \L \O \l \o  (underline/overline toggles)
             \~  (non-breaking space)
             {}  (grouping braces)
             \U+XXXX  (Unicode code point)
             \P  (paragraph / newline)
    """
    if not text:
        return ""

    text = re.sub(r"\\[ACFHQTWf].*?;", "", text)

    def clean_stacking(match):
        return match.group(0).replace(r"\S", "").replace(";", "").replace("^", "/")

    text = re.sub(r"\\S.*?;", clean_stacking, text)
    text = re.sub(r"\\[LOlo]", "", text)
    text = text.replace(r"\~", " ")
    text = re.sub(r"[{}]", "", text)

    def replace_unicode(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except Exception:
            return match.group(0)

    text = re.sub(r"\\U\+([0-9A-Fa-f]{4})", replace_unicode, text)
    text = text.replace(r"\P", "\n")

    return text.strip()


def normalize_text(text):
    """NFKC-normalize for consistent keyword matching across Unicode variants."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def get_entity_position(entity):
    """
    Return the position dict ({'x': ..., 'y': ...}) for any entity type,
    checking all field-name variants used by different DXF exporters.
    Returns None if no position can be determined.
    """
    return (
        entity.get("insertPoint")
        or entity.get("insertionPoint")
        or entity.get("position")
        or entity.get("alignmentPoint")
        or entity.get("firstLine")
        or entity.get("secondLine")
        or entity.get("center")
        or entity.get("startPoint")
    )


def get_block_entities(block_def):
    """Block definition can be a list of entities or a dict with 'entities' key."""
    if isinstance(block_def, list):
        return block_def
    if isinstance(block_def, dict):
        return block_def.get("entities", [])
    return []


def extract_entities(data, blocks=None, depth=0, max_depth=5):
    """
    Recursively yield (entity, parent_insert_or_None) from top-level entities
    and nested block definitions (INSERT / DIMENSION).

    Yields tuples of (entity, parent_entity) so callers can determine whether
    an entity came from a block and access the parent's position for transforms.
    """
    if depth > max_depth:
        return

    if isinstance(data, dict):
        entities = data.get("entities", [])
    elif isinstance(data, list):
        entities = data
    else:
        return

    for ent in entities:
        yield ent
        etype = ent.get("type")
        if etype in ("INSERT", "DIMENSION") and blocks:
            name = (
                ent.get("name")
                if etype == "INSERT"
                else (ent.get("block") or ent.get("name"))
            )
            if name and name in blocks:
                block_ents = get_block_entities(blocks[name])
                yield from extract_entities(block_ents, blocks, depth + 1, max_depth)


TEXT_ENTITY_TYPES = frozenset(["TEXT", "MTEXT", "ATTRIB", "ATTDEF", "LEADER"])


def extract_text_from_entity(entity):
    """
    Extract and decode text content from a text-bearing entity.
    Returns decoded, NFKC-normalized text or empty string.
    """
    etype = entity.get("type", "")
    if etype not in TEXT_ENTITY_TYPES:
        return ""
    raw = entity.get("text", "")
    if not raw:
        return ""
    return normalize_text(decode_cad_unicode(raw))


def iter_text_entities(data, blocks=None, max_depth=5):
    """
    Yield text-bearing entities from top-level and nested blocks.
    """
    if blocks is None and isinstance(data, dict):
        blocks = data.get("blocks")
    for entity in extract_entities(data, blocks=blocks, max_depth=max_depth):
        if entity.get("type", "") in TEXT_ENTITY_TYPES:
            yield entity


def extract_text_annotations(
    data,
    blocks=None,
    keywords=None,
    pattern=None,
    include_position=True,
    max_depth=5,
):
    """
    Extract normalized text annotation records, optionally filtered by
    keyword list and/or regex pattern. Uses correct block coordinate transformations.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = keywords or []
    normalized_keywords = [normalize_text(str(k)) for k in keywords if str(k).strip()]
    normalized_keywords_lower = [k.lower() for k in normalized_keywords]
    regex = re.compile(pattern) if pattern else None

    from src.occupancy_grids import _traverse_entity_tree, _get_text_center
    
    if blocks is None and isinstance(data, dict):
        blocks = data.get("blocks", {})
    
    entities = data.get("entities", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    records = []

    def _apply_matrix_xy(m, x, y):
        return (
            float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
            float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
        )

    def on_leaf(ent, etype, m):
        text = extract_text_from_entity(ent)
        if not text:
            return

        text_lower = text.lower()
        matched_keywords = [
            kw for kw, kw_lower in zip(normalized_keywords, normalized_keywords_lower)
            if kw_lower in text_lower
        ]
        regex_match = regex.search(text) if regex else None

        if normalized_keywords and not matched_keywords:
            return
        if regex and not regex_match:
            return

        record = {
            "text": text,
            "entity_type": etype,
            "matched_keywords": matched_keywords,
        }
        if regex_match:
            record["regex_match"] = regex_match.group(0)
            
        if include_position:
            # First try get_entity_position for standard raw values
            pos = get_entity_position(ent)
            # Or fallback to _get_text_center which handles start/end lines
            if not pos:
                pos = _get_text_center(ent)
                
            if pos and "x" in pos and "y" in pos:
                xw, yw = _apply_matrix_xy(m, float(pos["x"]), float(pos["y"]))
                record["position"] = {"x": xw, "y": yw}
            else:
                record["position"] = pos
                
        records.append(record)

    _traverse_entity_tree(entities, blocks, on_leaf, max_depth=max_depth)
    return records
