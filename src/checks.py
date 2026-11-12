"""Data-quality checks and model-free association tests.

Nothing here fits a model. Run `python -m src.checks` for a plain-text audit of
the training file; the notebook uses the same functions and adds the plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    POST_OUTCOME_COLUMNS,
    TARGET,
    TARGET_DOCUMENTED_RANGE,
    URGENT_HOURS,
    load_train,
    split_train_validation,
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and share per column, worst first."""
    n = df.isna().sum()
    out = pd.DataFrame({"n_missing": n, "pct_missing": (100 * n / len(df)).round(2)})
    return out.sort_values("n_missing", ascending=False)


def target_outside_documented_range(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target falls outside the range the data dictionary states."""
    lo, hi = TARGET_DOCUMENTED_RANGE
    return df.loc[(df[TARGET] < lo) | (df[TARGET] > hi), [ID_COL, TARGET]].sort_values(TARGET)


def category_balance(df: pd.DataFrame) -> pd.DataFrame:
    """How concentrated each categorical column is (shares exclude missing values)."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts()
        rows.append(
            {
                "column": col,
                "n_levels": len(counts),
                "top_level": counts.index[0],
                "top_share_pct": round(100 * counts.iloc[0] / counts.sum(), 1),
                "smallest_level": counts.index[-1],
                "smallest_level_rows": int(counts.iloc[-1]),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def lane_distance_by_lane_type(df: pd.DataFrame) -> pd.DataFrame:
    """Distance summary for lanes that stay inside one market vs. lanes that cross markets."""
    known = df.dropna(subset=["origin_market", "destination_market", "lane_distance_miles"])
    lane_type = np.where(
        known["origin_market"] == known["destination_market"], "same market", "cross market"
    )
    return known.groupby(lane_type)["lane_distance_miles"].describe().round(0)


def correlation_ratio(codes: np.ndarray, y: np.ndarray) -> float:
    """Eta squared: share of the variance of y explained by group means (0..1)."""
    n = np.bincount(codes)
    group_mean = np.bincount(codes, weights=y) / n
    return float((n * (group_mean - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def association_with_target(
    df: pd.DataFrame, columns: list[str], n_perm: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Strength of association between each column and the target, with a permutation p-value.

    Categorical columns: eta squared (missing treated as its own level).
    Numeric columns: Spearman rank correlation on rows where the value is present.
    The p-value is the share of target shuffles whose statistic is at least as extreme.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            present = df[col].notna()
            x = df.loc[present, col].rank().to_numpy()
            y = df.loc[present, TARGET].rank().to_numpy()
            stat = float(np.corrcoef(x, y)[0, 1])
            null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
            measure, extreme = "spearman_rho", np.abs(null) >= abs(stat)
        else:
            codes = pd.factorize(df[col].fillna("<missing>"))[0]
            y = df[TARGET].to_numpy()
            stat = correlation_ratio(codes, y)
            null = np.array([correlation_ratio(codes, rng.permutation(y)) for _ in range(n_perm)])
            measure, extreme = "eta_squared", null >= stat
        rows.append(
            {
                "column": col,
                "measure": measure,
                "value": round(stat, 4),
                # +1 correction so a p-value is never reported as exactly zero
                "perm_p_value": round((extreme.sum() + 1) / (n_perm + 1), 4),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Headline data-quality facts about the training file, one row per check."""
    lo, hi = TARGET_DOCUMENTED_RANGE
    features = df[FEATURES]
    known_lane = df.dropna(subset=["origin_market", "destination_market"])
    same_market = known_lane[known_lane["origin_market"] == known_lane["destination_market"]]
    n_pairs_seen = known_lane.groupby(["origin_market", "destination_market"]).ngroups
    n_markets = df["origin_market"].nunique()
    checks = [
        ("rows", len(df)),
        ("duplicate load_id", int(df[ID_COL].duplicated().sum())),
        ("duplicate rows ignoring load_id", int(df.drop(columns=ID_COL).duplicated().sum())),
        ("rows with >= 1 missing feature", int(features.isna().any(axis=1).sum())),
        ("pct rows with >= 1 missing feature", round(100 * features.isna().any(axis=1).mean(), 1)),
        (f"target below documented min {lo}", int((df[TARGET] < lo).sum())),
        (f"target above documented max {hi}", int((df[TARGET] > hi).sum())),
        (f"urgent loads (hours_to_pickup < {URGENT_HOURS:g})", int((df["hours_to_pickup"] < URGENT_HOURS).sum())),
        ("loads with hours_to_pickup missing", int(df["hours_to_pickup"].isna().sum())),
        ("loads with zero carrier bids but an agreed rate", int((df["carrier_bids_received"] == 0).sum())),
        ("origin-destination market pairs seen", f"{n_pairs_seen} of {n_markets * n_markets}"),
        ("max distance on a same-market lane (miles)", int(same_market["lane_distance_miles"].max())),
    ]
    return pd.DataFrame(checks, columns=["check", "value"]).set_index("check")


def main() -> None:
    df = load_train(include_post_outcome=True)
    with pd.option_context("display.width", 120, "display.max_colwidth", 60):
        print("== Audit of the training file ==")
        print(audit(df).to_string(), "\n")
        print("== Missing values ==")
        print(missingness(df).to_string(), "\n")
        print("== Categorical balance ==")
        print(category_balance(df).to_string(), "\n")
        print("== Lane distance: same-market vs cross-market ==")
        print(lane_distance_by_lane_type(df).to_string(), "\n")
        # Anything that relates features to the target uses the train split only,
        # so the validation split stays an honest check on decisions made from it.
        train, _ = split_train_validation(df)
        print("== Association with the target (train split only, no model) ==")
        print(association_with_target(train, FEATURES + POST_OUTCOME_COLUMNS).to_string())


if __name__ == "__main__":
    main()
