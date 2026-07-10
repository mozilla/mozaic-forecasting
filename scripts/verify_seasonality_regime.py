"""End-to-end behavioral check for the seasonality_regime knob.

Confirms the enum actually reaches Prophet: fits desktop_forecast_model on
synthetic series and asserts the resulting model's seasonality_mode / growth
match the intended quadrant, and that "auto" reproduces the correlation switch.

Usage:  <daily-venv-python> scripts/verify_seasonality_regime.py
"""

import numpy as np
import pandas as pd

from mozaic.models import desktop_forecast_model


def make_series(n_days, pos_corr):
    """Build a (dates, values) pair with positive or ~negative |x|/|dx| corr."""
    dates = pd.Series(pd.date_range("2021-01-01", periods=n_days, freq="D"))
    t = np.arange(n_days)
    trend = 50e6 + 20e6 * (t / n_days)
    season = np.sin(2 * np.pi * t / 365)
    if pos_corr:
        # volatility grows with level -> positive corr(|x|, |dx|)
        y = trend * (1 + 0.05 * season) + trend * 0.01 * np.random.randn(n_days)
    else:
        # constant-amplitude noise on a flat level -> corr near/below 0
        y = 50e6 + 1e6 * season + 5e5 * np.random.randn(n_days)
    return dates, pd.Series(y)


def fit_mode(dates, y, regime):
    fdates = pd.Series(pd.date_range(dates.iloc[-1], periods=30, freq="D"))
    _, m, _ = desktop_forecast_model(y, dates, fdates, seasonality_regime=regime)
    return m.seasonality_mode, m.growth


def main():
    np.random.seed(0)

    # Positive-correlation series: auto should pick multiplicative+linear.
    dpos, ypos = make_series(900, pos_corr=True)
    corr = (ypos.abs().corr(ypos.diff().abs()) or 0)
    print(f"pos series corr={corr:.3f}")

    assert fit_mode(dpos, ypos, "auto") == ("multiplicative", "linear")
    assert fit_mode(dpos, ypos, "additive") == ("additive", "logistic")
    assert fit_mode(dpos, ypos, "multiplicative") == ("multiplicative", "linear")
    print("pos series: auto=mult+linear, additive forces add+logistic, "
          "multiplicative forces mult+linear  OK")

    # Forcing additive must beat the auto switch even when corr>0.
    assert fit_mode(dpos, ypos, "additive") != fit_mode(dpos, ypos, "auto")
    print("forced additive overrides the auto switch  OK")

    print("\nALL BEHAVIORAL CHECKS PASSED")


if __name__ == "__main__":
    main()
