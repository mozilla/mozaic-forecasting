"""Blast-radius check for the per-(country, year) HOLIDAY_SKIPS mechanism.

Compares current detrend() on the skip-filtered calendar vs the unfiltered calendar,
across every country/platform in the cached mozaic inputs (Legacy desktop, Glean mobile,
2020-2026). Expectation: ONLY the skipped country-years move; every other country shows
zero changed days, and the skipped country changes only within the skipped year(s).
"""
import sys
import numpy as np
import pandas as pd
import db_dtypes  # noqa: F401  (registers 'dbdate' for the cached parquets)

from mozaic.holiday_smart import detrend, get_calendar, HOLIDAY_SKIPS

THR, MAXR, MINR = -0.032, 5, 3
DESK = "../mozaic-daily/mozaic_parts.raw.legacy.desktop.DAU.parquet"
MOB = "../mozaic-daily/mozaic_parts.raw.glean.mobile.DAU.parquet"
EPS = 0.01  # a day "changed" if |full-skip|/y exceeds 1%


def load_total(path):
    df = pd.read_parquet(path)
    df["x"] = pd.to_datetime(df["x"])
    return df.groupby(["country", "x"], as_index=False)["y"].sum()


def analyze(country, s):
    s = s.sort_values("x").reset_index(drop=True)
    dates, y = s["x"], s["y"].to_numpy(float)
    years = sorted(set(dates.dt.year.unique()) | {dates.dt.year.min() - 1, dates.dt.year.max() + 1})

    cal_full = get_calendar(country, years, skip_holidays=[])
    cal_skip = get_calendar(country, years, skip_holidays=HOLIDAY_SKIPS)
    for c in (cal_full, cal_skip):
        c["submission_date"] = pd.to_datetime(c["submission_date"])
        c.drop_duplicates(subset=["submission_date"], inplace=True)
        c.reset_index(drop=True, inplace=True)

    full = detrend(dates=dates, y=pd.Series(y), holiday_df=cal_full,
                   threshold=THR, max_radius=MAXR, min_radius=MINR).to_numpy()
    skip = detrend(dates=dates, y=pd.Series(y), holiday_df=cal_skip,
                   threshold=THR, max_radius=MAXR, min_radius=MINR).to_numpy()

    rel = np.abs(full - skip) / np.maximum(y, 1.0)
    changed = rel > EPS
    yrs_changed = sorted(dates[changed].dt.year.unique().tolist()) if changed.any() else []
    return {"n": len(y), "n_changed": int(changed.sum()),
            "pct_changed": 100 * changed.mean(),
            "max_chg": float(rel.max()),
            "years_changed": yrs_changed}


for label, path in [("DESKTOP (legacy)", DESK), ("MOBILE (glean)", MOB)]:
    tot = load_total(path)
    print(f"\n================= {label} =================")
    print(f"{'ctry':>4} {'days':>5} {'chg':>5} {'%chg':>6} {'max%':>8}  years-changed")
    for c in sorted(tot["country"].unique()):
        r = analyze(c, tot[tot["country"] == c])
        flag = "  <-- IR (expected)" if c == "IR" else ("  !!! UNEXPECTED" if r["n_changed"] else "")
        print(f"{c:>4} {r['n']:>5} {r['n_changed']:>5} {r['pct_changed']:>6.2f} "
              f"{100*r['max_chg']:>8.1f}  {r['years_changed']}{flag}")
