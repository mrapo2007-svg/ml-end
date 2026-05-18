"""Verify notebook is valid, save to file"""
import json, os

with open(r'C:\Users\Senya\ml-end\notebook\T5.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'C:\Users\Senya\ml-end\_verify_out.txt', 'w', encoding='utf-8') as f:
    f.write(f'cells: {len(nb["cells"])}\n')
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])[:120].replace('\n', ' ')
        f.write(f'  {i}  [{c["cell_type"]}]: {src}\n')
    # Verify JSON round-trip
    f.write('\nNotebook is valid JSON.\n')

# check for duplicate Step 3 in code
cell_sources = [''.join(c['source']) for c in nb['cells']]
step3_count = sum(1 for s in cell_sources if 'Step 3' in s)
step0_count = sum(1 for s in cell_sources if 'Step 0' in s)
print(f'Steps: Step0 x{step0_count}, Step1 x{sum(1 for s in cell_sources if "Step 1" in s)}, '
      f'Step2 x{sum(1 for s in cell_sources if "Step 2" in s)}, '
      f'Step3 x{step3_count}')
print('All cells:', len(cell_sources))
