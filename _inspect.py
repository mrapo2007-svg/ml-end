"""Inspect T5.ipynb and save details"""
import json
with open(r'C:\Users\Senya\ml-end\notebook\T5.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open(r'C:\Users\Senya\ml-end\inspect_out.txt', 'w', encoding='utf-8') as f:
    f.write(f"cells: {len(nb['cells'])}\n")
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])[:200]
        f.write(f"Cell {i} type={c['cell_type']}\n{src}\n---\n")
