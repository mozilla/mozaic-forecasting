"""Batch-fit every legacy_desktop tile under each seasonality_regime and report
failures. Answers: are the FORCED regime arms safe to batch-run, or do they blow
up on the specific tiles they push into a never-before-run mode x growth quadrant?

Faithful to production: builds tiles exactly like populate_tiles (same per-country
pivot, same >30-obs filter, same detrend + holiday calendar), then fits each via
the real Tile.__post_init__. Per tile we record the fitted mode/growth, and flag:
  - EXCEPTION   : Stan / Prophet raised
  - NONFINITE   : forecast or yhat contains NaN/inf
  - DEGEN_CAP   : logistic tile with cap <= floor (or non-finite cap/floor)
  - BLOWUP      : median forecast max > 5x historical max (runaway extrapolation)

NOTE: runs the RAW series (no Iran fill splice) -- the regime-robustness question
is about per-tile mode/growth/cap shape, which the fill (IR only) does not change.

Usage:
    <daily-venv-python> scripts/robustness_regime_tiles.py [parquet] [out.json]
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mozaic import Tile
from mozaic.models import DesktopModelConfig, make_desktop_model
from mozaic.utils import _pivot_populations

DEFAULT_PARQUET = (
    "/Users/brendanwells/work/mozaic-daily/test_data/"
    "mozaic_parts.raw.legacy.desktop.DAU.parquet"
)
FORECAST_START = "2026-03-03"   # day after the parquet's last date
FORECAST_END = "2026-10-01"     # covers the ~2026-08-22 summer-trough KPI
REGIMES = ["auto", "additive", "multiplicative"]


def load(path):
    t = pq.read_table(path)
    i = t.schema.get_field_index("x")
    t = t.set_column(i, "x", pc.strftime(t["x"], "%Y-%m-%d"))
    t = t.replace_schema_metadata(None)
    df = t.to_pandas()
    df["x"] = pd.to_datetime(df["x"])
    return df


def build_tile_inputs(df):
    """Replicate populate_tiles: (country, population, dates, series) per tile."""
    tiles = []
    for country in sorted(df.country.unique()):
        piv = _pivot_populations(df, country)
        pops = [c for c in piv.columns if c not in ("x", "country")]
        for pop in pops:
            if len(piv[pop].dropna()) > 30:
                tiles.append((country, pop, piv["x"], piv[pop]))
    return tiles


def logistic_cap_floor(detrended, dates, start):
    """Recompute the desktop logistic cap/floor to flag degenerate caps."""
    mask = pd.to_datetime(dates) < pd.Timestamp(start)
    obs = pd.Series(detrended)[mask.values].replace({0: np.nan}).dropna()
    if len(obs) == 0:
        return np.nan, np.nan
    cap = obs.tail(426).max() * 1.05
    floor = obs.tail(426).min() * (1 if cap > 100e6 else 0.92)
    return cap, floor


def assess(tile, hist_series):
    """Return (mode, growth, list_of_flags) for a fitted tile."""
    m = tile._prophet_model
    mode = getattr(m, "seasonality_mode", "?")
    growth = getattr(m, "growth", "?")
    flags = []

    # Strict: scan the FULL predictive-sample matrix, not just the median (which
    # skips NaNs), plus all yhat bounds -- catches partial corruption from the
    # overflow/invalid-value warnings Prophet emits in the seasonal matmul.
    samples = tile.forecast.to_numpy()
    yb = tile._prophet_forecast[["yhat", "yhat_lower", "yhat_upper"]].to_numpy()
    n_bad_samples = int((~np.isfinite(samples)).sum())
    n_bad_yhat = int((~np.isfinite(yb)).sum())
    if n_bad_samples or n_bad_yhat:
        flags.append(f"NONFINITE(samples={n_bad_samples},yhat={n_bad_yhat})")

    med = tile.forecast.quantile(0.5, axis=1)

    if growth == "logistic":
        cap, floor = logistic_cap_floor(
            tile.holiday_detrended_historical_data, tile.historical_dates,
            FORECAST_START,
        )
        if not (np.isfinite(cap) and np.isfinite(floor) and cap > floor):
            flags.append("DEGEN_CAP")

    hist_max = pd.Series(hist_series).max()
    if np.isfinite(med.max()) and hist_max > 0 and med.max() > 5 * hist_max:
        flags.append("BLOWUP")

    return mode, growth, flags


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET
    out = sys.argv[2] if len(sys.argv) > 2 else "./tmp/regime_robustness.json"

    df = load(path)
    tiles = build_tile_inputs(df)
    print(f"{len(tiles)} tiles x {len(REGIMES)} regimes = "
          f"{len(tiles) * len(REGIMES)} fits\n", flush=True)

    results = []
    t0 = time.time()
    n_done = 0
    total = len(tiles) * len(REGIMES)
    for regime in REGIMES:
        model = make_desktop_model(DesktopModelConfig(seasonality_regime=regime))
        for country, pop, dates, series in tiles:
            rec = {"regime": regime, "country": country, "population": pop}
            try:
                tile = Tile(
                    metric="DAU", country=country, population=pop,
                    forecast_start_date=FORECAST_START,
                    forecast_end_date=FORECAST_END,
                    forecast_model=model,
                    historical_dates=dates, raw_historical_data=series,
                    threshold=-0.032, max_radius=5, min_radius=3,
                )
                mode, growth, flags = assess(tile, series)
                rec.update(mode=mode, growth=growth,
                           status="FAIL" if flags else "OK", flags=flags)
            except Exception as e:  # noqa: BLE001 -- diagnostic harness
                rec.update(mode=None, growth=None, status="EXCEPTION",
                           flags=["EXCEPTION"], error=f"{type(e).__name__}: {e}"[:200])
            results.append(rec)

            n_done += 1
            if n_done % 20 == 0 or n_done == total:
                el = time.time() - t0
                eta = el / n_done * (total - n_done)
                sys.stdout.write(
                    f"  [{n_done}/{total}] {regime} | {el:.0f}s elapsed | "
                    f"ETA {eta:.0f}s\n")
                sys.stdout.flush()

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    res = pd.DataFrame(results)
    print("\n=== status by regime ===")
    print(pd.crosstab(res["regime"], res["status"]).to_string())

    print("\n=== fitted quadrant (mode x growth) by regime ===")
    ok = res[res["status"] != "EXCEPTION"].copy()
    ok["quadrant"] = ok["mode"].astype(str) + "+" + ok["growth"].astype(str)
    print(pd.crosstab(ok["regime"], ok["quadrant"]).to_string())

    print("\n=== tiles pushed into a NEW quadrant vs auto ===")
    auto_q = ok[ok.regime == "auto"].set_index(["country", "population"])
    auto_q = (auto_q["mode"] + "+" + auto_q["growth"]).to_dict()
    for regime in ("additive", "multiplicative"):
        sub = ok[ok.regime == regime]
        pushed = sum(
            1 for _, r in sub.iterrows()
            if auto_q.get((r["country"], r["population"]))
            != f"{r['mode']}+{r['growth']}"
        )
        print(f"  {regime}: {pushed}/{len(sub)} tiles differ from their auto quadrant")

    print("\n=== all non-OK tiles ===")
    bad = res[res["status"] != "OK"]
    if len(bad) == 0:
        print("  (none)")
    else:
        for _, r in bad.iterrows():
            extra = r.get("error", "") or ",".join(r["flags"])
            print(f"  [{r['regime']}] {r['country']}|{r['population']} "
                  f"({r['mode']}+{r['growth']}): {extra}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
