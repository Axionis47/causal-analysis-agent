"""Synthetic datasets for causal evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_simple_causal(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    confounder = rng.normal(0, 1, size=n)
    treatment = (confounder + rng.normal(0, 1, size=n) > 0).astype(int)
    outcome = 2.0 * treatment + 0.5 * confounder + rng.normal(0, 1, size=n)
    return pd.DataFrame({"confounder": confounder, "treatment": treatment, "outcome": outcome})


if __name__ == "__main__":
    df = generate_simple_causal()
    print(df.head())
