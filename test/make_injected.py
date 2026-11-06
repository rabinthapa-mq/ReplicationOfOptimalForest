"""Build injected-anomaly datasets (global/local/clustered/dependency) on top of
normal credit-card transactions (CC-Inject). Usage: python make_injected.py IN.npz OUT_DIR"""
import sys, numpy as np
from sklearn.mixture import GaussianMixture

data = np.load(sys.argv[1]); out = sys.argv[2]
Xn = data["X"][data["y"] == 0].astype("float64")
rng = np.random.default_rng(0)
Xn = Xn[rng.choice(len(Xn), 10000, replace=False)]
n_anom = int(round(len(Xn) * 0.05 / 0.95))          # 5% anomaly rate overall
lo, hi = Xn.min(0), Xn.max(0); mid, half = (lo + hi) / 2, (hi - lo) / 2

def global_(n):      # uniform over the 1.1x bounding box
    return rng.uniform(mid - 1.1 * half, mid + 1.1 * half, size=(n, Xn.shape[1]))
def local_(n):       # GMM fitted to normals, covariance inflated x5
    g = GaussianMixture(5, covariance_type="diag", random_state=0).fit(Xn)
    comp = rng.choice(5, n, p=g.weights_)
    return g.means_[comp] + rng.normal(size=(n, Xn.shape[1])) * np.sqrt(5 * g.covariances_[comp])
def clustered_(n):   # one tight cluster displaced ~1 sd per feature from the data centre
    d = rng.normal(size=Xn.shape[1]); d /= np.linalg.norm(d)
    centre = Xn.mean(0) + Xn.std(0) * d * np.sqrt(Xn.shape[1])
    return centre + rng.normal(size=(n, Xn.shape[1])) * Xn.std(0) * 0.3
def dependency_(n):  # marginals kept, feature correlations broken by column shuffling
    A = Xn[rng.choice(len(Xn), n, replace=False)].copy()
    for j in range(A.shape[1]): A[:, j] = rng.permutation(A[:, j])
    return A

for name, fn in [("global", global_), ("local", local_), ("clustered", clustered_), ("dependency", dependency_)]:
    A = fn(n_anom); X = np.vstack([Xn, A]); y = np.r_[np.zeros(len(Xn)), np.ones(n_anom)]
    p = rng.permutation(len(X))
    np.savez(f"{out}/ccinject_{name}.npz", X=X[p].astype("float32"), y=y[p].astype(int))
    print(name, X.shape, int(y.sum()))
