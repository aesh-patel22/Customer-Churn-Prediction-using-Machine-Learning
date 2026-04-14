import json

with open('Telecom_customer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_code.py', 'w', encoding='utf-8') as f:
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            f.write(f"# CELL {i}\n")
            f.write("".join(c['source']) + "\n\n")
