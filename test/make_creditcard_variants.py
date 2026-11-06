"""Build anomaly-rate variants (1%, 2%, 5%) and a deduplicated full version of the
Kaggle Credit Card Fraud data (mlg-ulb/creditcardfraud).
Usage: python make_creditcard_variants.py creditcard.csv OUT_DIR"""
import sys, numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src).drop_duplicates().reset_index(drop=True)
hr = (df.Time % 86400) / 3600
df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
df["log_amount"] = np.log1p(df.Amount)
feat = [f"V{i}" for i in range(1, 29)] + ["hour_sin", "hour_cos", "log_amount"]
X = df[feat].copy()
X["log_amount"] = (X.log_amount - X.log_amount.mean()) / X.log_amount.std()
y = df.Class.values
fraud, normal = np.where(y == 1)[0], np.where(y == 0)[0]
print("after dedup:", X.shape, "frauds:", y.sum())
np.savez(f"{out}/creditcard_full.npz", X=X.values.astype("float32"), y=y)
rng = np.random.default_rng(0)
for name, r in [("1pct", .01), ("2pct", .02), ("5pct", .05)]:
    n = int(round(len(fraud) * (1 - r) / r))
    idx = np.concatenate([fraud, rng.choice(normal, n, replace=False)]); rng.shuffle(idx)
    out_df = X.iloc[idx].reset_index(drop=True); out_df["label"] = y[idx]
    out_df.to_csv(f"{out}/creditcard_{name}.csv", index=False)
    np.savez(f"{out}/creditcard_{name}.npz", X=X.iloc[idx].values.astype("float32"), y=y[idx])
    print(name, out_df.shape, int(y[idx].sum()))
