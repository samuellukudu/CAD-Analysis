import json
import sys

def find_pattern_hatch(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Searching for HATCH with ARC edges...")
    
    def check_entities(entities, location):
        for entity in entities:
            if entity.get('type') == 'HATCH':
                loops = entity.get('boundaryLoops', [])
                for loop in loops:
                    edges = loop.get('edges', [])
                    for edge in edges:
                        if edge.get('type') == 2: # Arc
                            print(f"Found HATCH with ARC edge in {location}: {entity.get('handle')}")
                            print(json.dumps(entity, indent=2))
                            return True
        return False

    if check_entities(data.get('entities', []), "Modelspace"):
        return

    # Check blocks too
    blocks = data.get('blocks', {})
    for block_name, block_content in blocks.items():
        block_entities = []
        if isinstance(block_content, list):
             block_entities = block_content
        elif isinstance(block_content, dict):
             block_entities = block_content.get('entities', [])
        
        if check_entities(block_entities, f"Block {block_name}"):
            return

    print("No HATCH with ARC edges found.")

if __name__ == "__main__":
    find_pattern_hatch(sys.argv[1])
