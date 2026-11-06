"""iForest baseline (scikit-learn) on the constructed datasets, 15 seeds, mean/std of
AUC-ROC and AUC-PR (average precision). Usage: python run_iforest_baseline.py DATA_DIR OUT.csv"""
import sys, glob, os, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    d = np.load(f); X, y = d["X"], d["y"]; roc, ap = [], []
    for seed in range(15):
        m = IsolationForest(n_estimators=100, max_samples=256, random_state=seed).fit(X)
        s = -m.score_samples(X)               # higher = more anomalous
        roc.append(roc_auc_score(y, s)); ap.append(average_precision_score(y, s))
    rows.append(dict(dataset=os.path.basename(f)[:-4], n=len(y), d=X.shape[1], rate=100*y.mean(),
                     roc_mean=100*np.mean(roc), roc_std=100*np.std(roc),
                     ap_mean=100*np.mean(ap), ap_std=100*np.std(ap)))
    print(rows[-1])
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
