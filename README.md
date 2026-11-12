# Freight load pricing

Predicting the agreed rate per mile for a freight load before it is posted, from historical load records.

**Status: week 1 — setup and data exploration.** No model has been fitted yet. This week describes the dataset and its
problems and fixes the validation split and the reference number that later work is measured against.

## Repository layout

```
notebooks/01_data_exploration.ipynb   Week 1 exploration, executed, with figures and conclusions
src/data.py                           Paths, column roles, loading, train/validation split, holdout seal
src/checks.py                         Data-quality audit and model-free association tests
src/plotstyle.py                      Shared matplotlib style
requirements.txt                      Pinned dependencies
```

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**The dataset is not in this repository.** Unzip the programme's `dataset.zip` at the repository root so the files sit at
`freight_load_pricing/data/freight_load_pricing_train.csv`, or point `FREIGHT_DATA_DIR` at the folder that holds the CSVs.

## Run

```bash
python -m src.checks                                   # plain-text data-quality audit
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
```

Everything is seeded (`SEED = 42` in `src/data.py`), so both commands reproduce the committed numbers exactly.

## Rules the code enforces

- **The holdout is sealed until week 4.** `load_holdout()` raises `HoldoutSealedError` unless it is deliberately
  unsealed. Nothing in this repository reads the holdout file.
- **No post-outcome inputs.** `finance_settlement_code` is written after settlement; `load_train()` drops it by default.
  The notebook loads it once, only to measure the leak.
- **One validation split.** `split_train_validation()` is an 80/20 split of the training file, stratified on pickup
  regime × target quintile, with a fixed seed. It is random because the export has no timestamp and no shipper or lane
  key to order by or group on.
- **Target-related exploration uses the train split only**, so the validation split stays an honest check.

## Week 1 findings

Details, figures and evidence are in the notebook. Training file: 2,400 loads, 12 columns.

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| 1 | Post-outcome column in the file | `finance_settlement_code` alone explains 58% of target variance | Dropped at load time, never a feature |
| 2 | Column whose timing contradicts its label | `carrier_bids_received` counts bids that arrive after posting; 33 loads have 0 bids and an agreed rate | Excluded until confirmed |
| 3 | Missing values throughout | 6–8% per feature; 44.4% of loads have at least one | Impute inside training folds, add indicators, never zero-fill |
| 4 | Target outside the documented range | 24 loads (1.0%); observed 1.00–10.14 vs documented 1.05–9.8 | Kept, not clipped |
| 5 | One level dominates each categorical column | Southeast / Monday / Enterprise each about 78%; 25 of 36 market pairs have under 10 loads | Origin and destination as separate effects; plan for unseen pairs |
| 6 | Distance inconsistent with market labels | Same-market lanes reach 2,849 miles, the same distribution as cross-market lanes | No geography-derived features; never join on distance |
| 7 | The regime that matters is almost absent | 15 loads under 2 hours to pickup (12 train / 3 validation) | Repeated cross-validation with intervals for this slice |
| 8 | No timestamp, shipper key or lane key | Only `load_id` order is available, and it shows no drift | Random stratified split; time drift cannot be tested |
| 9 | Very little signal in the legitimate features | Only `hours_to_pickup` (Spearman −0.06, p ≈ 0.02) is distinguishable from noise, one column at a time | Baseline first; treat any large gain as suspected leakage |

## Reference number

Metric: **mean absolute error in USD per mile** on the validation split.

Quoting every load at the train-split median (3.67 USD/mile) gives an MAE of **2.11 USD/mile** on 479 validation loads.
This is not a model; it is the number a model has to beat.

## Open questions

1. Is `carrier_bids_received` known at the moment a quote is made?
2. Is the documented target range out of date, or are the 24 out-of-range rates errors?
3. Are the 78% shares for Southeast, Monday and Enterprise real, or a default written upstream?
4. Can more loads with under two hours to pickup be provided?
5. Do the source systems hold a posting timestamp or a city-level lane key?
