import pandas as pd
import numpy as np

file_path = 'price_estimator/data/predictions_validation.parquet'
df = pd.read_parquet(file_path)

# 计算差值: 实际最低价 - Q80预测价
# 这里的含义是：如果是正数，说明实际价格高于预测，即预测成功（价格没跌破预测）
# 如果是负数，说明实际价格跌破了预测价格
diff = df['lowest_trade_price_next4'] - df['pred_q80']

stats = diff.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99])

print("Difference (Actual Lowest - Q80 Prediction) Statistics:")
print(stats)

print("\nDistribution Histogram:")
counts, bin_edges = np.histogram(diff.dropna(), bins=10)
for i in range(len(counts)):
    print(f"{bin_edges[i]:>8.4f} to {bin_edges[i+1]:>8.4f}: {'#' * int(counts[i] * 50 / counts.max())} ({counts[i]})")

# 额外计算下覆盖率
coverage = (diff >= 0).mean()
print(f"\nCalculated Q80 Coverage: {coverage:.4%}")
