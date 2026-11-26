"""NormGuard on CICIoV2024: experiment E7 (alert aggregation)."""
# ==========================================================================
# NormGuard on CICIoV2024 (decimal) -- full experiment
#   C1  Self-configuring guard : per-ID, per-byte rules learned from NORMAL
#       traffic only (fixed value / allowed set / range / any), relaxed greedily
#       until a false-alarm budget beta is met on held-out normal traffic.
#   C2  Open-set attack naming : nearest known-attack signature (Hamming
#       distance over ID + 8 bytes) with a calibrated reject radius ->
#       "known type X" or "UNKNOWN attack".
#   C3  Evasion analysis        : masquerading attacker may resend the attack
#       payload under any legitimate ID and change up to k bytes.
# Experiments (all saved as CSV):
#   E1 seen attacks (usual protocol)     E2 leave-one-attack-out (unseen)
#   E3 evasion vs k                      E4 budget sweep / ablation
#   E5 open-set naming                   E6 cost (size, speed)
# CPU only. ~20-40 min in Colab.
# ==========================================================================
import os, glob, time, pickle, warnings, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, f1_score
warnings.filterwarnings("ignore")
import argparse, sys
# ---- paths: command line (python src/...py --data DIR --out DIR) or Colab globals ----
if "ipykernel" not in sys.modules and __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--data", default="data/decimal", help="folder with the CICIoV2024 decimal CSV files")
    _ap.add_argument("--out", default="results", help="folder for result CSVs")
    _a = _ap.parse_args()
    DATA_DIR, OUT_DIR = _a.data, _a.out
else:
    DATA_DIR = globals().get("DATA_DIR", "/content/drive/MyDrive/PhD/CICIoV2024/decimal")
    OUT_DIR = globals().get("OUT_DIR", "/content/drive/MyDrive/PhD/CICIoV2024/normguard_results")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
BETA = 0.001                      # default false-alarm budget (0.1%)
BYTES = [f"DATA_{i}" for i in range(8)]
FEATS = ["ID"] + BYTES
ATTACKS = {"DoS": "decimal_DoS.csv", "GAS": "decimal_spoofing-GAS.csv", "RPM": "decimal_spoofing-RPM.csv",
           "SPEED": "decimal_spoofing-SPEED.csv", "STEERING_WHEEL": "decimal_spoofing-STEERING_WHEEL.csv"}
SET_MAX = 32                      # an allowed-value set is used only up to 32 values
LEVELS = ["fixed", "set", "range", "range+", "any"]


def load(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        hits = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True)
        if not hits:
            raise FileNotFoundError(p)
        p = hits[0]
    return pd.read_csv(p)[FEATS].astype(np.int64)


# ------------------------------------------------------------------ data ----
ben = load("decimal_benign.csv"); n = len(ben)
ben_tr, ben_val, ben_te = ben.iloc[:int(.6 * n)], ben.iloc[int(.6 * n):int(.7 * n)], ben.iloc[int(.7 * n):]
atk_tr, atk_val, atk_te = {}, {}, {}
for a, f in ATTACKS.items():
    d = load(f); m1, m2 = int(.56 * len(d)), int(.7 * len(d))     # train 56% / val 14% / test 30%
    atk_tr[a], atk_val[a], atk_te[a] = d.iloc[:m1], d.iloc[m1:m2], d.iloc[m2:]
test_parts = [(ben_te, "BENIGN")] + [(atk_te[a], a) for a in ATTACKS]
X_test = pd.concat([p for p, _ in test_parts], ignore_index=True)
c_test = np.concatenate([[c] * len(p) for p, c in test_parts])
y_test = (c_test != "BENIGN").astype(int)
uniq_idx = pd.DataFrame(X_test.values).assign(c=c_test).drop_duplicates().index.values   # unique (frame, class)
NORMAL_FRAMES = set(map(tuple, ben_tr.values))
print(f"benign train/val/test {len(ben_tr)}/{len(ben_val)}/{len(ben_te)}; test unique frames {len(uniq_idx)}")


# ======================================================= C1: the guard ====
class NormGuard:
    """Per-ID, per-byte rules learned from normal traffic, tuned to a false-alarm budget."""

    def __init__(self, beta=BETA, tune=True, use_bytes=True):
        self.beta, self.tune, self.use_bytes = beta, tune, use_bytes

    # -- rule construction for one byte at a given relaxation level
    @staticmethod
    def _rule(vals, level):
        u = np.unique(vals)
        if level == 0 and len(u) == 1:
            return ("fixed", int(u[0]))
        if level <= 1 and len(u) <= SET_MAX:
            return ("set", frozenset(u.tolist()))
        if level <= 2:
            return ("range", (int(u.min()), int(u.max())))
        if level == 3:
            pad = max(1, int(0.1 * (u.max() - u.min() + 1)))
            return ("range+", (max(0, int(u.min()) - pad), min(255, int(u.max()) + pad)))
        return ("any", None)

    @staticmethod
    def _viol(rule, col):
        kind, v = rule
        if kind == "fixed":
            return col != v
        if kind == "set":
            return ~np.isin(col, list(v))
        if kind in ("range", "range+"):
            return (col < v[0]) | (col > v[1])
        return np.zeros(len(col), dtype=bool)

    def fit(self, normal_train, normal_val):
        self.vals = {int(c): g[BYTES].values for c, g in normal_train.groupby("ID")}
        self.level = {(c, j): 0 for c in self.vals for j in range(8)}
        self._build()
        self.history = []
        if self.tune and self.use_bytes:
            Xv = normal_val[FEATS].values
            for _ in range(2000):
                alarms, per_rule = self._alarms_with_blame(Xv)
                far = alarms.mean()
                self.history.append(far)
                if far <= self.beta or not per_rule:
                    break
                worst = max(per_rule, key=per_rule.get)     # rule causing the most false alarms
                if self.level[worst] >= len(LEVELS) - 1:
                    break
                self.level[worst] += 1
                self._build_one(*worst)
        return self

    def _build(self):
        self.rules = {c: [self._rule(self.vals[c][:, j], self.level[(c, j)]) for j in range(8)] for c in self.vals}

    def _build_one(self, c, j):
        self.rules[c][j] = self._rule(self.vals[c][:, j], self.level[(c, j)])

    def _alarms_with_blame(self, X):
        ids = X[:, 0]; alarms = ~np.isin(ids, list(self.rules)); blame = {}
        for c in np.unique(ids):
            if int(c) not in self.rules:
                continue
            m = ids == c; sub = X[m, 1:]; bad = np.zeros(m.sum(), dtype=bool)
            for j, r in enumerate(self.rules[int(c)]):
                v = self._viol(r, sub[:, j])
                if v.any():
                    blame[(int(c), j)] = int(v.sum())
                bad |= v
            alarms[np.where(m)[0]] = bad
        return alarms, blame

    def predict(self, df_or_X):
        X = df_or_X[FEATS].values if isinstance(df_or_X, pd.DataFrame) else np.asarray(df_or_X)
        if not self.use_bytes:
            return (~np.isin(X[:, 0], list(self.rules))).astype(int)
        return self._alarms_with_blame(X)[0].astype(int)

    def violations(self, X):
        """number of rule violations per frame (unknown ID counts as 9)"""
        X = np.asarray(X); out = np.zeros(len(X), dtype=int)
        for i, row in enumerate(X):
            c = int(row[0])
            if c not in self.rules:
                out[i] = 9; continue
            if self.use_bytes:
                out[i] = sum(bool(self._viol(r, np.array([row[j + 1]]))[0]) for j, r in enumerate(self.rules[c]))
        return out

    def explain(self, row):
        c = int(row[0])
        if c not in self.rules:
            return f"unknown ID {c}"
        for j, (kind, v) in enumerate(self.rules[c]):
            x = int(row[j + 1])
            if self._viol((kind, v), np.array([x]))[0]:
                return f"ID {c} byte {j} = {x} breaks '{kind}' rule {sorted(v) if kind == 'set' else v}"
        return ""

    def rule_summary(self):
        kinds = [r[0] for rs in self.rules.values() for r in rs]
        return {k: kinds.count(k) for k in LEVELS}

    def size_values(self):
        s = len(self.rules)
        if not self.use_bytes:
            return s
        for rs in self.rules.values():
            for kind, v in rs:
                s += 1 if kind == "fixed" else len(v) if kind == "set" else 0 if kind == "any" else 2
        return s


# ============================================== C2: open-set naming =====
class OpenSetNamer:
    """Known attack types are remembered as signatures (target ID + 8 payload bytes).
    A flagged frame gets the type of the nearest signature on the SAME ID; it is
    called UNKNOWN if no known attack used that ID, or if it is farther (in bytes)
    than that type's calibrated radius (95% of the type's validation frames)."""

    def fit(self, attack_tr, attack_val, accept=0.95):
        self.sig, self.lab = [], []
        for a, d in attack_tr.items():
            u = d[FEATS].drop_duplicates().values
            self.sig.append(u); self.lab += [a] * len(u)
        self.sig = np.vstack(self.sig); self.lab = np.array(self.lab)
        self.radius = {}
        for a in attack_tr:
            dv = self._nearest(attack_val[a][FEATS].values)[1] if a in attack_val and len(attack_val[a]) else np.array([0])
            dv = dv[dv <= 8]
            self.radius[a] = int(np.quantile(dv, accept)) if len(dv) else 0
        return self

    def _nearest(self, X):
        X = np.asarray(X)
        same_id = X[:, None, 0] == self.sig[None, :, 0]
        D = (X[:, None, 1:] != self.sig[None, :, 1:]).sum(-1) + np.where(same_id, 0, 100)
        k = D.argmin(1)
        return self.lab[k], D[np.arange(len(X)), k]

    def name(self, X):
        lab, dist = self._nearest(X)
        r = np.array([self.radius[l] for l in lab])
        return np.where(dist <= r, lab, "UNKNOWN")



# ==========================================================================
# E7: ALERT AGGREGATION -- cut false alarms from rare-but-normal messages
#   unknown ID           -> alarm immediately
#   byte-rule violation  -> alarm only if >= m of the last W frames of that ID
#                           violated a byte rule (attacks repeat; rare normal
#                           variants do not)
# Streams are evaluated in recorded order. Reports false alarms per 1M normal
# frames, attack frame recall and detection delay (frames until first alarm).
# ==========================================================================
from collections import deque, defaultdict
OUT7 = f"{OUT_DIR}/E7_aggregation.csv"
rows7 = []


def stream_alarms(g, X, m, W):
    X = np.asarray(X); ids = X[:, 0]
    known = np.isin(ids, list(g.rules))
    viol = np.zeros(len(X), dtype=bool)
    for c in np.unique(ids[known]):
        mk = ids == c; sub = X[mk, 1:]; bad = np.zeros(mk.sum(), dtype=bool)
        for j, r in enumerate(g.rules[int(c)]):
            bad |= g._viol(r, sub[:, j])
        viol[np.where(mk)[0]] = bad
    out = np.zeros(len(X), dtype=bool); hist = defaultdict(lambda: deque(maxlen=W))
    for i in range(len(X)):
        if not known[i]:
            out[i] = True; continue
        h = hist[int(ids[i])]; h.append(bool(viol[i]))
        out[i] = viol[i] and sum(h) >= m
    return out


for beta in [0.001, 0.0001]:
    g = NormGuard(beta=beta).fit(ben_tr, ben_val)
    for m, W in [(1, 1), (2, 5), (3, 10), (5, 20), (10, 50)]:
        fa = stream_alarms(g, ben_te[FEATS].values, m, W)
        row = {"beta": beta, "m": m, "W": W, "benign_frames": len(ben_te),
               "false_alarm_rate": float(fa.mean()), "false_alarms_per_1M": float(fa.mean() * 1e6)}
        for a in ATTACKS:
            al = stream_alarms(g, atk_te[a][FEATS].values, m, W)
            row[f"recall_{a}"] = float(al.mean())
            row[f"delay_{a}"] = int(np.argmax(al)) if al.any() else -1
        rows7.append(row); print(row)
pd.DataFrame(rows7).to_csv(OUT7, index=False)
print(pd.DataFrame(rows7).round(5).to_string(index=False))
print("Saved", OUT7)
