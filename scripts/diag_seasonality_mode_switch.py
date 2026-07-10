"""Diagnostic: how does desktop_forecast_model's seasonality_mode / growth
auto-switch resolve across legacy-desktop tiles?

The switch is:  (x.abs().corr(x.diff().abs()) or 0) > 0.0  -> multiplicative + linear
                else                                        -> additive + logistic

It fires per tile on the holiday-detrended series. This script characterizes the
distribution of that correlation across tiles so we can decide whether the switch
is a stable, data-driven decision or a near-zero coin-flip worth pinning for the
parameter sweep.

Usage:
    <daily-venv-python> scripts/diag_seasonality_mode_switch.py [parquet_path]
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

# NOTE: the model actually sees the holiday-DETRENDED series, but detrending only
# nudges a handful of holiday days, which barely moves corr(|x|, |diff x|). The raw
# series is a faithful proxy for characterizing the switch; borderline tiles could
# shift slightly under detrending.

DEFAULT_PARQUET = (
    "/Users/brendanwells/work/mozaic-daily/test_data/"
    "mozaic_parts.raw.legacy.desktop.DAU.parquet"
)


def load(path):
    """Read the dbdate parquet, coercing x to datetime and dropping pandas meta."""
    t = pq.read_table(path)
    i = t.schema.get_field_index("x")
    t = t.set_column(i, "x", pc.strftime(t["x"], "%Y-%m-%d"))
    t = t.replace_schema_metadata(None)
    df = t.to_pandas()
    df["x"] = pd.to_datetime(df["x"])
    return df


def switch_corr(y):
    """Return the exact quantity the auto-switch tests: corr(|x|, |diff x|).

    Mirrors desktop_forecast_model, which feeds the model the series with 0->NaN.
    """
    x = pd.Series(y).astype(float).replace({0: np.nan})
    return x.abs().corr(x.diff().abs())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET
    df = load(path)
    print(f"loaded {len(df):,} rows | {df.x.min().date()} -> {df.x.max().date()}")

    keys = ["country", "win10", "win11", "winX"]
    rows = []
    for key, g in df.groupby(keys):
        g = g.sort_values("x")
        rows.append((key, g["y"].sum(), switch_corr(g["y"])))

    res = pd.DataFrame(rows, columns=["tile", "total_dau", "corr"])
    res["mode"] = np.where(
        res["corr"].fillna(0) > 0, "mult+linear", "add+logistic"
    )

    print(f"\n{len(res)} tiles total\n")

    print("=== auto-switch outcome (tile counts) ===")
    print(res["mode"].value_counts().to_string())
    w = res.groupby("mode")["total_dau"].sum()
    print("\nDAU-weighted share by mode:")
    print((w / w.sum()).round(3).to_string())

    print("\n=== how many tiles sit near the zero boundary (fragile) ===")
    for thr in (0.02, 0.05, 0.10):
        n = (res["corr"].abs() < thr).sum()
        print(f"  |corr| < {thr:.2f}: {n} tiles ({n / len(res):.0%})")

    print("\n=== corr distribution ===")
    print(res["corr"].describe().round(4).to_string())

    print("\n=== top-15 tiles by DAU (these dominate the KPI) ===")
    top = res.sort_values("total_dau", ascending=False).head(15)
    print(top[["tile", "corr", "mode"]].to_string(index=False))


if __name__ == "__main__":
    main()
