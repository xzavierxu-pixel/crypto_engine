import pandas as pd
import numpy as np

# Load predictions from the baseline model
df = pd.read_parquet('artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features/validation_predictions.parquet')

# Create bins of 0.1 for p_up
bins = np.arange(0, 1.1, 0.1)
df['bin'] = pd.cut(df['p_up'], bins=bins, right=False)

# Assuming simple threshold at 0.5 for raw accuracy checking
df['hard_pred'] = (df['p_up'] >= 0.5).astype(int)
df['correct'] = (df['hard_pred'] == df['target'])

results = df.groupby('bin', observed=False).agg(
    count=('p_up', 'count'),
    accuracy=('correct', 'mean')
).reset_index()

print('Baseline Model Probability Bins (p_up):')
print('-' * 50)
print(f"{'Bin':<15} | {'Count':<10} | {'Accuracy':<10}")
print('-' * 50)
for _, row in results.iterrows():
    acc = f"{row['accuracy']:.4f}" if pd.notna(row['accuracy']) else "N/A"
    print(f"{str(row['bin']):<15} | {row['count']:<10} | {acc:<10}")
print('-' * 50)

# Check specific model if the user meant catboost_calendar_coordinate_search
# Note: That experiment didn't save predictions to disk natively, so we are checking the baseline.
