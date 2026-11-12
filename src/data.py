"""Data access for the freight load pricing project.

This is the only module that knows where the files live and which columns a
model is allowed to see. Everything else imports from here, so the leakage rule
and the holdout seal are enforced in one place instead of by convention.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Unzip dataset.zip at the repo root, or point FREIGHT_DATA_DIR somewhere else.
DATA_DIR = Path(os.environ.get("FREIGHT_DATA_DIR", REPO_ROOT / "freight_load_pricing" / "data"))
TRAIN_FILE = "freight_load_pricing_train.csv"
HOLDOUT_FILE = "freight_load_pricing_holdout.csv"

ID_COL = "load_id"
TARGET = "agreed_rate_per_mile"
# Range stated in DATA_DICTIONARY.md. The data does not fully respect it.
TARGET_DOCUMENTED_RANGE = (1.05, 9.8)

NUMERIC_FEATURES = [
    "load_weight_lbs",
    "lane_distance_miles",
    "hours_to_pickup",
    "carrier_bids_received",
]
CATEGORICAL_FEATURES = [
    "origin_market",
    "destination_market",
    "equipment_type",
    "day_of_week_posted",
    "shipper_tier",
]

# Written by finance after the invoice is reconciled, i.e. after the rate we are
# trying to predict has been agreed. Never a model input.
POST_OUTCOME_COLUMNS = ["finance_settlement_code"]

# The dictionary marks this "At collection", but it counts bids received after
# the load was posted, while the question is to quote *before* posting. Kept out
# of the default feature set until the mentor confirms when it is known.
TIMING_SUSPECT_COLUMNS = ["carrier_bids_received"]

QUOTE_TIME_FEATURES = [
    c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in TIMING_SUSPECT_COLUMNS
]

# The ML engineer's note: loads posted less than ~2h before pickup behave like a
# separate regime, and that is where the model has to perform.
URGENT_HOURS = 2.0

SEED = 42


class HoldoutSealedError(RuntimeError):
    """Raised when something tries to read the holdout before week 4."""


def load_train(include_post_outcome: bool = False) -> pd.DataFrame:
    """Load the training file.

    Post-outcome columns are dropped unless explicitly requested. The only
    legitimate reason to request them is to document the leak.
    """
    df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    if not include_post_outcome:
        df = df.drop(columns=POST_OUTCOME_COLUMNS)
    return df


def load_holdout() -> pd.DataFrame:
    """Load the holdout. Sealed until week 4.

    Unsealing takes a deliberate act (setting FREIGHT_UNSEAL_HOLDOUT=week4), so
    it cannot happen by accident from a notebook cell.
    """
    if os.environ.get("FREIGHT_UNSEAL_HOLDOUT") != "week4":
        raise HoldoutSealedError(
            "The holdout is sealed until week 4. Use split_train_validation() "
            "on the training file for every decision before then."
        )
    return pd.read_csv(DATA_DIR / HOLDOUT_FILE).drop(columns=POST_OUTCOME_COLUMNS)


def pickup_regime(df: pd.DataFrame) -> pd.Series:
    """Label each load 'urgent' (< URGENT_HOURS), 'standard' or 'unknown' (missing)."""
    hours = df["hours_to_pickup"]
    regime = np.where(hours < URGENT_HOURS, "urgent", "standard")
    return pd.Series(np.where(hours.isna(), "unknown", regime), index=df.index, name="pickup_regime")


def split_train_validation(
    df: pd.DataFrame, valid_frac: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified train/validation split of the training file.

    Random rather than time- or group-based because the export has no timestamp
    and no shipper/lane key to group on, and contains no duplicate loads.
    Stratified on pickup regime x target quintile so the rare urgent loads and
    the long right tail of the target are represented on both sides.
    """
    df = df.sort_values(ID_COL).reset_index(drop=True)  # independent of file row order
    regime = pickup_regime(df)
    target_band = pd.qcut(df[TARGET], 5, labels=False).astype(str)
    # Urgent loads are too few to also band by target; keep them as one stratum.
    strata = (regime + "|" + target_band).where(regime != "urgent", "urgent")

    rng = np.random.default_rng(seed)
    is_valid = np.zeros(len(df), dtype=bool)
    for _, positions in sorted(df.groupby(strata).indices.items()):
        n_valid = int(round(len(positions) * valid_frac))
        is_valid[rng.choice(positions, size=n_valid, replace=False)] = True

    return df[~is_valid].reset_index(drop=True), df[is_valid].reset_index(drop=True)
