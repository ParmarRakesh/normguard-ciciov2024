# NormGuard: Label-free Detection of Unseen and Evasive Attacks in In-vehicle Networks

Code and results for the paper **"NormGuard: Label-free Detection of Unseen and Evasive Attacks in In-vehicle Networks"** (under review).

NormGuard is an intrusion detector for the in-vehicle CAN bus that is trained on **normal traffic only**:

- **Stage 1: self-configuring guard.** It learns a rule for every CAN ID and data byte (fixed value, allowed set or range) and relaxes these rules automatically until a chosen false-alarm budget is met.
  - An **unknown ID** raises an **alarm**.
  - A **byte that breaks its rule** raises a **warning**.
  - Every alert comes with a plain-language reason, e.g. `unknown ID 291`.
- **Stage 2: open-set namer.** It names an alert as a known attack type, or as `UNKNOWN attack` when it resembles no known attack.

It is evaluated on the public **CICIoV2024** dataset under three settings: seen attacks, unseen attacks (leave-one-attack-out), and an evasive attacker who resends attack payloads under legitimate IDs.

## Key results (CICIoV2024, decimal version)

| | Random forest | XGBoost | MLP | **NormGuard** |
|---|---|---|---|---|
| Recall, seen attacks (min over 5 types) | 0.33 | 1.00 | 0.33 | **1.00** |
| Recall, unseen attacks (min over 5 types) | 0.00 | 0.00 | 0.00 | **1.00** |
| Withheld attack named "UNKNOWN" | n/a (closed set: 0%) | n/a | n/a | **100%** |
| Evaded by moving payload to a legitimate ID (k = 0) | 100% | 100% | 100% | **0%** |
| Model size | 232 kB | 177 kB | 96 kB | **4.7 kB** |

Limitations:
- **Warnings on normal frames.** NormGuard raises warnings on 0.16–0.18% of normal frames (Tab. 7 and 8 of the paper).
- **Repetitive data.** Its strength on this dataset partly reflects how regular CICIoV2024's normal traffic is.

## Repository structure

```
src/
  normguard_experiments.py   experiments E1-E6 (detection, evasion, ablation, naming, cost)
  e7_aggregation.py          experiment E7 (alert aggregation)
  make_figures.py            re-creates the evasion figure from results/E3_evasion.csv
notebooks/                   the same experiments as Google Colab notebooks
results/                     all result files reported in the paper
figures/                     figure used in the paper
```

## Data

CICIoV2024 is published by the Canadian Institute for Cybersecurity, University of New Brunswick, and is **not** included here. Download it from <https://www.unb.ca/cic/datasets/iov-dataset-2024.html> and place the six **decimal** CSV files in `data/decimal/`:

```
decimal_benign.csv  decimal_DoS.csv  decimal_spoofing-GAS.csv
decimal_spoofing-RPM.csv  decimal_spoofing-SPEED.csv  decimal_spoofing-STEERING_WHEEL.csv
```

## How to run

### Locally (CPU is sufficient)

```bash
pip install -r requirements.txt
python src/normguard_experiments.py --data data/decimal --out results   # E1-E6, about 20-40 min
python src/e7_aggregation.py       --data data/decimal --out results    # E7, about 10-20 min
python src/make_figures.py         --results results --out figures
```

### Google Colab

1. Open a notebook from `notebooks/` in Colab.
2. Set `DATA_DIR` and `OUT_DIR` in the second cell.
3. Choose *Run all*.

All splits are in recorded (time) order and every random choice uses seed 42, so the runs are deterministic up to library versions.

## Where each result of the paper comes from

| Paper | File | Notes |
|---|---|---|
| Tab. 3, Tab. 4 (seen / unseen recall) | `results/E1_E2_seen_unseen.csv` | column `frames`: `all_frames` or `unique_frames` |
| Tab. 5 (naming) | `results/E5_open_set_naming.csv` | `radius = closed-set RF` marks the baseline |
| Fig. 2, Tab. 6 (evasion) | `results/E3_evasion.csv` | column `evaded_with_attack_content` |
| Tab. 7 (ablation, budget) | `results/E4_budget_ablation.csv` | |
| Tab. 8 (alert aggregation) | `results/E7_aggregation.csv` | |
| Tab. 9 (cost) | `results/E6_cost.csv` | |
| Tab. 10 (explanations) | `results/alert_explanations.csv` | |
| Rule counts (Section 4.3) | `results/guard_rules.json` | |

## Citation

The citation will be added after publication.

## License

The code is released under the MIT License (see `LICENSE`). CICIoV2024 remains subject to the terms of its publishers.
