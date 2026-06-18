import pandas as pd
import numpy as np

df = pd.read_parquet('price_estimator/safe_lowest_price_gap/reports/predictions_validation.parquet')
tolerance = 1e-6

# Definitions
covered = (df['p_pred'] + tolerance >= df['target_safe']) & (df['p_pred'] <= df['p_side'] + tolerance)
df['covered'] = covered
df['covered_feasible'] = covered & df['feasible']

# Bins
bins = np.arange(0.0, 1.1, 0.1)
df['bucket'] = pd.cut(df['p_side'], bins=bins, right=False)

# Group metrics
results = []
for b, group in df.groupby('bucket'):
    count = len(group)
    if count == 0:
        continue
    
    active_mask = group['action'] == 'active'
    active_count = active_mask.sum()
    active_share = active_count / count
    
    feasible_count = group['feasible'].sum()
    covered_count = group['covered'].sum()
    covered_feasible_count = group['covered_feasible'].sum()
    
    coverage_overall = covered_count / count if count > 0 else np.nan
    coverage_feasible = covered_feasible_count / feasible_count if feasible_count > 0 else np.nan
    
    covered_gap_norm_mean = group.loc[group['covered_feasible'], 'gap_norm'].mean()
    
    mean_pside = group['p_side'].mean()
    mean_ppred = group['p_pred'].mean()
    
    results.append({
        'Bucket (p_side)': str(b),
        'Total Count': count,
        'Active Count': active_count,
        'Active Share': f"{active_share:.2%}",
        'Feasible Count': feasible_count,
        'Coverage Overall': f"{coverage_overall:.2%}",
        'Coverage Feasible': f"{coverage_feasible:.2%}",
        'Covered Gap Norm Mean': f"{covered_gap_norm_mean:.4f}",
        'Mean p_side': f"{mean_pside:.4f}",
        'Mean p_pred': f"{mean_ppred:.4f}"
    })

res_df = pd.DataFrame(results)

markdown_table = res_df.to_markdown(index=False)
with open('price_estimator/safe_lowest_price_gap/reports/validation_pside_buckets.md', 'w') as f:
    f.write("# Validation Set Analysis by p_side Buckets\n\n")
    f.write("This report provides a detailed breakdown of the safe_lowest_price_gap model's performance on the validation set, segmented by the baseline probability (p_side) in increments of 0.1.\n\n")
    f.write("## Metrics Definitions\n")
    f.write("- **Active Count**: Number of samples where the model made a prediction rather than falling back to the baseline.\n")
    f.write("- **Coverage Overall**: Proportion of total samples successfully covered by the prediction.\n")
    f.write("- **Coverage Feasible**: Proportion of *theoretically feasible* samples successfully covered.\n")
    f.write("- **Covered Gap Norm Mean**: The mean normalized gap between the predicted price and the safe target (lower is better, 1.0 means no gap compression).\n\n")
    f.write("## Bucket Breakdown\n\n")
    f.write(markdown_table)
    f.write("\n")

print("Generated markdown file.")
