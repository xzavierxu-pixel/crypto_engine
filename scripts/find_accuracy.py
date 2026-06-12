import os
import json
import re

target_dir = r'C:\Users\ROG\Desktop\crypto_engine_version1\artifacts\data_v2\reports\reversal_hybrid'
results = []

def get_metrics(data):
    val_acc = None
    acc2 = None
    def find_keys(d, path=""):
        nonlocal val_acc, acc2
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'reversal_accepted_accuracy' and 'validation' in path.lower():
                    val_acc = v
                elif k == 'reversal_accepted_accuracy' and val_acc is None:
                    val_acc = v # fallback
                
                if k == 'accepted_sample_accuracy':
                    acc2 = v
                find_keys(v, path + "." + k)
        elif isinstance(d, list):
            for i, item in enumerate(d):
                find_keys(item, path + f"[{i}]")
    find_keys(data, "root")
    return val_acc, acc2

for root, dirs, files in os.walk(target_dir):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if 'reversal_accepted_accuracy' in content:
                        data = json.loads(content)
                        val_acc, acc2 = get_metrics(data)
                        
                        if val_acc is None:
                            match = re.search(r'"reversal_accepted_accuracy"\s*:\s*([\d\.]+)', content)
                            if match: val_acc = float(match.group(1))
                        
                        if acc2 is None:
                            match2 = re.search(r'"accepted_sample_accuracy"\s*:\s*([\d\.]+)', content)
                            if match2: acc2 = float(match2.group(1))

                        if val_acc is not None and acc2 is not None:
                            results.append((float(val_acc), float(acc2), os.path.basename(root), path))
            except Exception as e:
                pass

results.sort(reverse=True, key=lambda x: x[0])
seen = set()
unique_results = []
for item in results:
    if item[2] not in seen:
        seen.add(item[2])
        unique_results.append(item)

for i, item in enumerate(unique_results[:10]):
    print(f'Folder: {item[2]} | validation reversal_accepted_accuracy: {item[0]} | accepted_sample_accuracy: {item[1]}')
