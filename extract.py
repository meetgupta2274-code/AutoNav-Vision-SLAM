import json
import sys

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            code_cells.append("".join(source))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# --- CELL ---\n\n".join(code_cells))

if __name__ == '__main__':
    extract_code(sys.argv[1], sys.argv[2])
