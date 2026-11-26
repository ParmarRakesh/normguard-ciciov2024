"""NormGuard on CICIoV2024: experiments E1-E6 (seen/unseen attacks, evasion, budget ablation, open-set naming, cost)."""
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


# ================================================ supervised baselines ===
def sup_models(multiclass=False):
    m = {"DecisionTree": DecisionTreeClassifier(random_state=SEED),
         "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
         "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, random_state=SEED)}
    if XGBClassifier is not None:
        m["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1, verbosity=0)
    return m


class Sup:
    def __init__(self, name, mdl):
        self.name, self.mdl, self.sc = name, mdl, None

    def fit(self, X, y):
        if self.name == "MLP":
            self.sc = StandardScaler().fit(X)
        self.mdl.fit(self._t(X), y); return self

    def _t(self, X):
        X = X[FEATS].values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.sc.transform(X) if self.sc is not None else X

    def predict(self, X):
        return self.mdl.predict(self._t(X)).astype(int)

    def p_attack(self, X):
        return self.mdl.predict_proba(self._t(X))[:, 1]


ben_sup = ben_tr.sample(n=min(300_000, len(ben_tr)), random_state=SEED)


def train_sup(held=None):
    known = [a for a in ATTACKS if a != held]
    Xtr = pd.concat([ben_sup] + [atk_tr[a] for a in known] + [atk_val[a] for a in known], ignore_index=True)
    ytr = np.r_[np.zeros(len(ben_sup)), np.ones(len(Xtr) - len(ben_sup))].astype(int)
    return {nm: Sup(nm, m).fit(Xtr, ytr) for nm, m in sup_models().items()}


def binary_rows(model, protocol, held, pred, idx=None, tag="all_frames"):
    rows = []
    sel = np.arange(len(y_test)) if idx is None else idx
    fpr = float(pred[sel][c_test[sel] == "BENIGN"].mean())
    for a in ATTACKS:
        if held is not None and a != held:
            continue
        m = sel[(c_test[sel] == "BENIGN") | (c_test[sel] == a)]
        p, r, f1, _ = precision_recall_fscore_support(y_test[m], pred[m], average="binary", zero_division=0)
        rows.append({"model": model, "protocol": protocol, "attack": a, "frames": tag,
                     "recall": r, "precision": p, "f1": f1, "false_alarm_rate": fpr})
    return rows


R1, R3, R4, R5, R6 = [], [], [], [], []

# ============================== E1/E2: seen and unseen (leave-one-out) ====
guard = NormGuard(beta=BETA).fit(ben_tr, ben_val)
t = time.perf_counter(); g_pred = guard.predict(X_test); g_us = (time.perf_counter() - t) / len(X_test) * 1e6
print("guard rules:", guard.rule_summary(), "| val FAR", round(guard.history[-1] if guard.history else 0, 5),
      "| stored values", guard.size_values())
for prot, held in [("SEEN", None)] + [("UNSEEN", a) for a in ATTACKS]:
    for tag, idx in [("all_frames", None), ("unique_frames", uniq_idx)]:
        R1 += binary_rows("NormGuard", prot, held, g_pred, idx, tag)
sup_seen = None
for prot, held in [("SEEN", None)] + [("UNSEEN", a) for a in ATTACKS]:
    models = train_sup(held)
    if held is None:
        sup_seen = models
    for nm, mdl in models.items():
        p = mdl.predict(X_test)
        for tag, idx in [("all_frames", None), ("unique_frames", uniq_idx)]:
            R1 += binary_rows(nm, prot, held, p, idx, tag)
    print(f"E1/E2 {prot} held-out={held} done")
pd.DataFrame(R1).to_csv(f"{OUT_DIR}/E1_E2_seen_unseen.csv", index=False)

# ================================================== E3: evasion vs k =====
legit_ids = np.array(sorted(guard.rules))
atk_unique = {a: atk_te[a][FEATS].drop_duplicates().values for a in ATTACKS}


def evade_guard(g, frame, k):
    """White-box attacker: pick the legitimate ID needing the fewest byte changes; evades iff changes <= k.
    Returns (evaded, bytes_changed, evaded_frame_is_a_normal_frame)."""
    best = None
    for cid in legit_ids:
        cand = frame.copy(); cand[0] = cid
        bad = [j for j, r in enumerate(g.rules[int(cid)]) if g._viol(r, np.array([cand[j + 1]]))[0]] if g.use_bytes else []
        if best is None or len(bad) < len(best[1]):
            best = (cand, bad)
    cand, bad = best
    if len(bad) > k:
        return False, len(bad), False
    for j in bad:                                       # set each violating byte to an allowed value
        kind, v = g.rules[int(cand[0])][j]
        cand[j + 1] = v if kind == "fixed" else (min(v) if kind == "set" else v[0])
    return True, len(bad), tuple(cand) in NORMAL_FRAMES


def evade_sup(mdl, frame, k):
    """Black-box query attacker: best legitimate ID, then greedily change up to k bytes to lower p(attack)."""
    cands = np.tile(frame, (len(legit_ids), 1)); cands[:, 0] = legit_ids
    p = mdl.p_attack(cands); cur = cands[p.argmin()].copy(); pc = p.min(); changed = set()
    if pc < 0.5:
        return True, 0, tuple(cur) in NORMAL_FRAMES
    for _ in range(k):
        best = (pc, None, None)
        for j in range(1, 9):
            if j in changed:
                continue
            trial = np.tile(cur, (256, 1)); trial[:, j] = np.arange(256)
            pt = mdl.p_attack(trial); i = pt.argmin()
            if pt[i] < best[0]:
                best = (pt[i], j, i)
        if best[1] is None:
            break
        cur[best[1]] = best[2]; changed.add(best[1]); pc = best[0]
        if pc < 0.5:
            return True, len(changed), tuple(cur) in NORMAL_FRAMES
    return False, len(changed), False


id_only = NormGuard(use_bytes=False).fit(ben_tr, ben_val)
detectors = {"NormGuard": ("guard", guard), "ID-allowlist only": ("guard", id_only)}
detectors.update({nm: ("sup", m) for nm, m in sup_seen.items() if nm in ("RandomForest", "XGBoost", "MLP")})
for k in range(0, 9):
    for dname, (kind, det) in detectors.items():
        for a, U in atk_unique.items():
            res = [evade_guard(det, f.copy(), k) if kind == "guard" else evade_sup(det, f.copy(), k) for f in U]
            ev = np.array([r[0] for r in res]); nul = np.array([r[2] for r in res])
            ch = np.array([r[1] for r in res], dtype=float)
            R3.append({"detector": dname, "k_bytes": k, "attack": a, "n_unique_frames": len(U),
                       "evasion_rate": ev.mean(), "evaded_and_nullified": (ev & nul).mean(),
                       "evaded_with_attack_content": (ev & ~nul).mean(),
                       "mean_bytes_changed_when_evaded": ch[ev].mean() if ev.any() else np.nan,
                       "mean_payload_bytes_kept_when_evaded": (8 - ch[ev]).mean() if ev.any() else np.nan})
    print(f"E3 k={k} done")
pd.DataFrame(R3).to_csv(f"{OUT_DIR}/E3_evasion.csv", index=False)

# ======================================= E4: budget sweep + ablation =====
for beta in [0.0001, 0.001, 0.01]:
    for label, g in [(f"NormGuard beta={beta}", NormGuard(beta=beta).fit(ben_tr, ben_val))]:
        p = g.predict(X_test)
        R4.append({"variant": label, "val_far": g.history[-1] if g.history else np.nan,
                   "test_far": float(p[y_test == 0].mean()), "recall": float(p[y_test == 1].mean()),
                   "rules": json.dumps(g.rule_summary()), "stored_values": g.size_values()})
for label, g in [("no budget tuning (exact rules)", NormGuard(tune=False).fit(ben_tr, ben_val)),
                 ("ID allow-list only (no byte rules)", id_only)]:
    p = g.predict(X_test)
    R4.append({"variant": label, "val_far": np.nan, "test_far": float(p[y_test == 0].mean()),
               "recall": float(p[y_test == 1].mean()), "rules": json.dumps(g.rule_summary()), "stored_values": g.size_values()})
pd.DataFrame(R4).to_csv(f"{OUT_DIR}/E4_budget_ablation.csv", index=False)
print(pd.DataFrame(R4).to_string(index=False))

# ============================================== E5: open-set naming ======
for held in [None] + list(ATTACKS):
    known = {a: atk_tr[a] for a in ATTACKS if a != held}
    namer = OpenSetNamer().fit(known, {a: atk_val[a] for a in known})
    flagged = np.where(g_pred == 1)[0]
    for a in ATTACKS:
        idx = flagged[c_test[flagged] == a]
        if len(idx) == 0:
            continue
        names = namer.name(X_test.values[idx])
        truth = "UNKNOWN" if a == held else a
        R5.append({"protocol": "SEEN" if held is None else "UNSEEN", "held_out": held, "attack": a,
                   "is_held_out": a == held, "frames": len(idx), "radius": json.dumps(namer.radius),
                   "correct_rate": float((names == truth).mean()),
                   "said_unknown": float((names == "UNKNOWN").mean())})
    # closed-set baseline: RF multi-class over known attacks only (cannot say UNKNOWN)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    Xk = pd.concat([known[a] for a in known]); yk = np.concatenate([[a] * len(known[a]) for a in known])
    rf.fit(Xk[FEATS].values, yk)
    for a in ATTACKS:
        idx = np.where(c_test == a)[0]
        pr = rf.predict(X_test.values[idx]); truth = "UNKNOWN" if a == held else a
        R5.append({"protocol": "SEEN" if held is None else "UNSEEN", "held_out": held, "attack": a,
                   "is_held_out": a == held, "frames": len(idx), "radius": "closed-set RF",
                   "correct_rate": float((pr == truth).mean()), "said_unknown": 0.0})
pd.DataFrame(R5).to_csv(f"{OUT_DIR}/E5_open_set_naming.csv", index=False)

# ======================================================= E6: cost ========
R6.append({"model": "NormGuard", "size_bytes": len(pickle.dumps(guard.rules)), "us_per_frame": g_us})
for nm, m in sup_seen.items():
    t = time.perf_counter(); m.predict(X_test); us = (time.perf_counter() - t) / len(X_test) * 1e6
    R6.append({"model": nm, "size_bytes": len(pickle.dumps(m.mdl)), "us_per_frame": us})
pd.DataFrame(R6).to_csv(f"{OUT_DIR}/E6_cost.csv", index=False)
ex = [{"class": c_test[i], "frame": X_test.values[i].tolist(), "explanation": guard.explain(X_test.values[i])}
      for i in np.where(y_test == 1)[0][::max(1, int(y_test.sum()) // 20)][:20]]
pd.DataFrame(ex).to_csv(f"{OUT_DIR}/alert_explanations.csv", index=False)
json.dump({"rule_summary": guard.rule_summary(), "far_history": guard.history[:200]},
          open(f"{OUT_DIR}/guard_rules.json", "w"))

# ================================================= console summary ======
D1 = pd.DataFrame(R1); D1 = D1[D1.frames == "all_frames"]
print("\n=== E1/E2 recall (SEEN vs UNSEEN) ===")
print(D1.pivot_table(index="model", columns=["protocol", "attack"], values="recall").round(3).to_string())
D3 = pd.DataFrame(R3)
print("\n=== E3 evasion rate with attack content kept, by k ===")
print(D3.pivot_table(index="detector", columns="k_bytes", values="evaded_with_attack_content").round(3).to_string())
print("\n=== E5 open-set naming ===")
print(pd.DataFrame(R5).to_string(index=False))
print("\n=== E6 cost ===\n", pd.DataFrame(R6).round(3).to_string(index=False))
print(f"\nAll CSVs in {OUT_DIR}")
