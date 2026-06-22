import pandas as pd
df = pd.read_parquet('price_estimator/data/price_estimator_train.parquet')
print("Total rows:", len(df))
if 'lowest_trade_price_next4' in df.columns:
    print("Missing lowest_trade_price_next4:", df['lowest_trade_price_next4'].isna().sum())
