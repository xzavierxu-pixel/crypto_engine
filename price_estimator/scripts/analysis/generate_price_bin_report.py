import pandas as pd
import numpy as np

file_path = 'price_estimator/data/predictions_validation.parquet'
df = pd.read_parquet(file_path)

# 计算差值
df['diff'] = df['lowest_trade_price_next4'] - df['pred_q80']

# 按照 0.05 进行价格分桶
bins = np.arange(0, 1.05, 0.05)
df['price_bin'] = pd.cut(df['lowest_trade_price_next4'], bins=bins)

# 聚合统计
report = df.groupby('price_bin', observed=True)['diff'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('p25', lambda x: x.quantile(0.25)),
    ('median', 'median'),
    ('p75', lambda x: x.quantile(0.75)),
    ('max', 'max')
]).reset_index()

# 增加覆盖率计算 (实际最低价 <= 预测价的比例，即 diff <= 0)
report['coverage_q80'] = df.groupby('price_bin', observed=True).apply(
    lambda x: (x['diff'] <= 0).mean()
).values

# 保存为 CSV
output_path = 'price_estimator/reports/analysis/price_diff_distribution_by_price_bin.csv'
report.to_csv(output_path, index=False)

print(f"Report saved to {output_path}")
print("\nSummary of the distribution by price bin:")
print(report[['price_bin', 'count', 'mean', 'median', 'coverage_q80']])
