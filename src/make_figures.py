"""Re-create Fig. 2 of the paper (evasion vs. number of changed bytes) from results/E3_evasion.csv."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
E = pd.read_csv(f"{a.results}/E3_evasion.csv")
P = E.pivot_table(index="detector", columns="k_bytes", values="evaded_with_attack_content")
style = {"RandomForest": ("#257de0", "-", "o", "Random Forest", 0.03),
         "XGBoost": ("#f47230", "--", "s", "XGBoost", 0.015),
         "MLP": ("#8a5cd1", ":", "^", "MLP", 0.0),
         "ID-allowlist only": ("#7a7a7a", "-.", "x", "ID allow-list only", -0.015),
         "NormGuard": ("#00a383", "-", "D", "NormGuard (ours)", 0.0)}
fig, ax = plt.subplots(figsize=(3.45, 2.0))
for k, (c, ls, mk, lab, off) in style.items():
    ax.plot(P.columns, P.loc[k].values + off, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
ax.set_xlabel("Bytes the attacker may change ($k$)")
ax.set_ylabel("Attacks evading detection")
ax.set_ylim(-0.05, 1.1); ax.set_xticks(range(9))
ax.yaxis.grid(True, color="#ddd", lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.3, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{a.out}/fig_evasion.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{a.out}/fig_evasion.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", f"{a.out}/fig_evasion.pdf")
