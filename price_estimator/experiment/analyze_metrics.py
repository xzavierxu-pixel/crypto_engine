import json
import glob
import os

files = glob.glob('price_estimator/experiment/**/*.json', recursive=True)
results = []
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            # Find validation metrics
            if 'validation' in data:
                val = data['validation']
            elif 'q90' in data: 
                for q_key in ['q70', 'q80', 'q90']:
                    if q_key in data:
                        exp_name = os.path.basename(os.path.dirname(os.path.dirname(f))) + f'_{q_key}'
                        results.append({
                            'name': exp_name, 
                            'coverage': data[q_key].get('coverage'), 
                            'mean_gap': data[q_key].get('mean_gap')
                        })
                continue
            else:
                val = data
            
            if val and isinstance(val, dict):
                exp_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
                if os.path.basename(f) in ['summary_metrics.json', 'quantile_metrics.json'] or 'summary' in f:
                    coverage = val.get('coverage')
                    mean_gap = val.get('mean_gap')
                    if coverage is not None and mean_gap is not None:
                         results.append({'name': exp_name, 'coverage': coverage, 'mean_gap': mean_gap, 'file': f})
    except Exception as e:
        pass

results = [r for r in results if r['mean_gap'] is not None and r['coverage'] is not None]

# Sort by mean_gap ascending
print("Sorted by mean_gap (ascending):")
results_gap = sorted(results, key=lambda x: x['mean_gap'])
for i, r in enumerate(results_gap[:10]):
    print(f"{i+1}. {r['name']} | mean_gap: {r['mean_gap']:.6f} | coverage: {r['coverage']:.6f} | file: {r['file']}")

print("\nSorted by mean_gap (ascending), ONLY coverage >= 0.90:")
results_90 = sorted([r for r in results if r['coverage'] >= 0.90], key=lambda x: x['mean_gap'])
for i, r in enumerate(results_90[:5]):
    print(f"{i+1}. {r['name']} | mean_gap: {r['mean_gap']:.6f} | coverage: {r['coverage']:.6f} | file: {r['file']}")
