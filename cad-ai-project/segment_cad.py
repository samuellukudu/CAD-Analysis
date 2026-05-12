import json
import argparse
import sys
import re


def decode_cad_unicode(text):
    if not text:
        return ""
    # Decode AutoCAD style \U+XXXX escapes.
    def _replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)
    text = re.sub(r'\\U\+([0-9A-Fa-f]{4})', _replace_unicode, text)
    # Basic MTEXT cleanup.
    text = text.replace(r'\P', '\n')
    text = re.sub(r'\\[ACFHQTWf].*?;', '', text)
    text = re.sub(r'[{}]', '', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Find matching text labels in CAD JSON and output seed points."
    )
    parser.add_argument("--file", required=True, help="Path to the JSON CAD file")
    parser.add_argument("--name", required=True, help="Name/keyword to search for (e.g., '一层平面图')")
    args = parser.parse_args()

    print(f"Loading {args.file}...")
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    results = []
    for ent in data.get('entities', []):
        if ent.get('type') in ['TEXT', 'MTEXT']:
            raw_text = ent.get('text', '')
            text = decode_cad_unicode(raw_text)
            if args.name in text:
                p = ent.get('insertPoint') or ent.get('position') or ent.get('startPoint')
                layer = ent.get('layer', 'Unknown')
                if p:
                    x, y = p['x'], p['y']
                    seed_str = f"{x:.2f},{y:.2f}"

                    results.append({
                        'text': text,
                        'x': x,
                        'y': y,
                        'layer': layer,
                        'seed': seed_str,
                    })

    if not results:
        print(f"\nNo text matching '{args.name}' found.")
        sys.exit(0)

    # Sort primarily by X-column (grouped by ~10m to align stacks left-to-right)
    # Sort secondarily by Y-elevation descending (top floor to bottom floor)
    results.sort(key=lambda r: (round(r['x'] / 10000), -r['y']))

    print(f"\nFound {len(results)} matches for '{args.name}':")
    print("=" * 100)
    for r in results:
        print(f"Text:  {r['text']}")
        print(f"Coord: X: {r['x']:10.2f}, Y: {r['y']:10.2f}  | Layer: {r['layer']}")
        print(f"-> Suggested --seed_point:  \"{r['seed']}\"")
        print("-" * 100)

if __name__ == "__main__":
    main()
