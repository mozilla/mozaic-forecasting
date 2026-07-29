import dataclasses
import logging
import numpy as np
import pandas as pd
import prophet

from dataclasses import dataclass

logging.getLogger("cmdstanpy").disabled = True


@dataclass
class ModelConfig:
    prophet_recent_weeks: int = 13
    # changepoint_range and n_changepoints control how Prophet places its trend
    # changepoints. Defaults match the values hardcoded in the desktop/mobile
    # forecast functions prior to this knob being exposed (changepoint_range=0.7,
    # n_changepoints=25). Subclasses may override changepoint_range to match the
    # platform-specific default (e.g. mobile uses 0.82).
    prophet_changepoint_range: float = 0.7
    prophet_n_changepoints: int = 25
    holiday_threshold: float = -0.032
    holiday_max_radius: int = 5
    holiday_min_radius: int = 3
    holiday_effect_floor: float = -0.6
    # Composite seasonality/growth regime. "auto" keeps each platform's
    # data-driven switch (desktop: level/volatility correlation; mobile: volume
    # thresholds). "additive"/"multiplicative" force that seasonality_mode. On
    # desktop the regime is coupled to growth (additive->logistic,
    # multiplicative->linear) to stay in the two quadrants the model has run; on
    # mobile growth stays volume-driven and the regime only sets seasonality_mode.
    seasonality_regime: str = "auto"
    # Threshold on the desktop "auto" switch statistic: a tile goes multiplicative
    # (and linear-growth) when corr(|y|, |dy|) > this value. 0.0 is the historical
    # hardcoded behaviour. Because the switch is per tile, this is a *continuous*
    # dial between all-additive (threshold above every tile's corr, ~+0.5) and
    # all-multiplicative (below every tile's corr, ~-0.6), passing through the
    # legacy split at 0.0 -- which "additive"/"multiplicative" cannot express.
    # Ignored unless seasonality_regime == "auto", and desktop-only (mobile's
    # switch is volume-driven, not correlation-driven).
    seasonality_corr_threshold: float = 0.0

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_slug(self):
        cps = self.prophet_changepoint_prior_scale
        thresh = f"{abs(self.holiday_threshold) * 1000:03.0f}"
        slug = (
            f"cps{cps}"
            f"_thresh{thresh}"
            f"_recent{self.prophet_recent_weeks}"
            f"_cpr{self.prophet_changepoint_range}"
            f"_ncp{self.prophet_n_changepoints}"
            f"_clip{abs(self.holiday_effect_floor)}"
            f"_sps{self.prophet_seasonality_prior_scale}"
        )
        if self.seasonality_regime != "auto":
            slug += f"_regime{self.seasonality_regime}"
        if self.seasonality_corr_threshold != 0.0:
            slug += f"_corr{self.seasonality_corr_threshold}"
        return slug


@dataclass
class DesktopModelConfig(ModelConfig):
    prophet_changepoint_prior_scale: float = 0.15983
    # Prophet seasonality_prior_scale. Default matches the value hardcoded in
    # desktop_forecast_model prior to this knob being exposed (0.00825).
    prophet_seasonality_prior_scale: float = 0.00825


@dataclass
class MobileModelConfig(ModelConfig):
    # Mobile previously hardcoded changepoint_range=0.82; preserve that default.
    prophet_changepoint_range: float = 0.82
    prophet_changepoint_prior_scale: float = 0.02
    # Prophet seasonality_prior_scale. Default matches the value hardcoded in
    # mobile_forecast_model's high-volume branch (0.1). Like today, this only
    # takes effect when historical_data.max() >= 1e6.
    prophet_seasonality_prior_scale: float = 0.1

    def __post_init__(self):
        # Fail loudly rather than silently ignoring it: mobile's regime switch is
        # volume-driven, so there is no correlation cutoff for this to move.
        if self.seasonality_corr_threshold != 0.0:
            raise ValueError(
                "seasonality_corr_threshold is desktop-only -- mobile's regime "
                "switch is volume-driven, not correlation-driven. Got "
                f"{self.seasonality_corr_threshold!r}; leave it at 0.0 for mobile."
            )


def make_desktop_model(config: DesktopModelConfig = None):
    if config is None:
        config = DesktopModelConfig()

    def model(historical_data, historical_dates, forecast_dates):
        return desktop_forecast_model(
            historical_data,
            historical_dates,
            forecast_dates,
            recent_weeks=config.prophet_recent_weeks,
            changepoint_prior_scale=config.prophet_changepoint_prior_scale,
            changepoint_range=config.prophet_changepoint_range,
            n_changepoints=config.prophet_n_changepoints,
            seasonality_prior_scale=config.prophet_seasonality_prior_scale,
            seasonality_regime=config.seasonality_regime,
            seasonality_corr_threshold=config.seasonality_corr_threshold,
        )

    return model


def make_mobile_model(config: MobileModelConfig = None):
    if config is None:
        config = MobileModelConfig()

    def model(historical_data, historical_dates, forecast_dates):
        return mobile_forecast_model(
            historical_data,
            historical_dates,
            forecast_dates,
            recent_weeks=config.prophet_recent_weeks,
            changepoint_prior_scale=config.prophet_changepoint_prior_scale,
            changepoint_range=config.prophet_changepoint_range,
            n_changepoints=config.prophet_n_changepoints,
            seasonality_prior_scale=config.prophet_seasonality_prior_scale,
            seasonality_regime=config.seasonality_regime,
        )

    return model


def _add_conditional_weekly_seasonality(
    m, observed, future, forecast_start, recent_weeks=13, fourier_order=3
):
    """
    Replace default weekly seasonality with two conditional seasonalities:
    - weekly_historical: active for training data before the recent window
    - weekly_recent: active for the recent window and all future dates

    Only weekly_recent is propagated into the forecast horizon.
    """
    recent_cutoff = forecast_start - pd.Timedelta(weeks=recent_weeks)

    observed = observed.copy()
    future = future.copy()

    if observed["ds"].min() >= recent_cutoff:
        observed["is_historical"] = False
        observed["is_recent"] = True
        future["is_historical"] = False
        future["is_recent"] = True

        m.add_seasonality(
            name="weekly_recent",
            period=7,
            fourier_order=fourier_order,
            condition_name="is_recent",
        )
    else:
        observed["is_historical"] = observed["ds"] < recent_cutoff
        observed["is_recent"] = observed["ds"] >= recent_cutoff
        future["is_historical"] = False
        future["is_recent"] = True

        m.add_seasonality(
            name="weekly_historical",
            period=7,
            fourier_order=fourier_order,
            condition_name="is_historical",
        )
        m.add_seasonality(
            name="weekly_recent",
            period=7,
            fourier_order=fourier_order,
            condition_name="is_recent",
        )

    return observed, future


def desktop_forecast_model(
    historical_data,
    historical_dates,
    forecast_dates,
    recent_weeks=13,
    changepoint_prior_scale=0.15983,
    changepoint_range=0.7,
    n_changepoints=25,
    seasonality_prior_scale=0.00825,
    seasonality_regime="auto",
    seasonality_corr_threshold=0.0,
):
    assert seasonality_regime in ("auto", "additive", "multiplicative"), (
        f"seasonality_regime must be auto/additive/multiplicative, "
        f"got {seasonality_regime!r}"
    )
    assert -1.0 <= seasonality_corr_threshold <= 1.0, (
        f"seasonality_corr_threshold is a correlation cutoff and must lie in "
        f"[-1, 1], got {seasonality_corr_threshold!r}"
    )
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": False,
        "yearly_seasonality": True,
        "uncertainty_samples": 1000,
        "changepoint_range": changepoint_range,
        "n_changepoints": n_changepoints,
        "seasonality_prior_scale": seasonality_prior_scale,
        "changepoint_prior_scale": changepoint_prior_scale,
        "growth": "logistic",
    }

    x = historical_data

    # "auto" keeps the historical level/volatility correlation switch; forcing a
    # regime pins mode+growth to the matching tested quadrant.
    # Under "auto", seasonality_corr_threshold moves the cutoff. Tiles are decided
    # independently, so sweeping it interpolates the *fraction* of tiles that run
    # multiplicative -- the interior between the two forced regimes. Note the tile
    # mix is heavily weight-skewed: on 2026-08 desktop, the legacy 0.0 cutoff puts
    # 37.5% of tiles but only 7.6% of DAU on the multiplicative side.
    corr = x.abs().corr(x.diff().abs()) or 0
    use_mult = seasonality_regime == "multiplicative" or (
        seasonality_regime == "auto" and corr > seasonality_corr_threshold
    )
    if use_mult:
        params["seasonality_mode"] = "multiplicative"
        params["growth"] = "linear"

    if (len(x.dropna()) > (365 * 2)) and (
        np.quantile(x.dropna(), 0.5) / (np.quantile(x.dropna(), 0.1) + 1e-8) < 5
    ):
        params["yearly_seasonality"] = True

    historical_mask = historical_dates < forecast_dates[0]
    observed = (
        pd.DataFrame(
            {
                "ds": historical_dates[historical_mask],
                "y": historical_data[historical_mask],
            }
        )
        .dropna()
        .reset_index(drop=True)
        .copy(deep=True)
    )
    future = pd.DataFrame({"ds": forecast_dates})

    if params["growth"] == "logistic":
        cap = observed["y"].tail(426).max() * 1.05
        if cap > 100e6:
            floor = observed["y"].tail(426).min() * 1
        else:
            floor = observed["y"].tail(426).min() * 0.92
        
        observed["cap"] = cap
        observed["floor"] = floor
        future["cap"] = cap
        future["floor"] = floor
    else:
        with np.errstate(invalid="ignore"):
            observed["y"] = np.log(observed["y"] + 1.0)

    np.random.seed(42)
    m = prophet.Prophet(**params)
    observed, future = _add_conditional_weekly_seasonality(
        m, observed, future, forecast_dates[0], recent_weeks=recent_weeks
    )
    m.fit(observed)

    prophet_forecast = m.predict(future)
    predictive_samples = pd.DataFrame(m.predictive_samples(future)["yhat"])

    if params["growth"] == "linear":
        predictive_samples = np.exp(predictive_samples) - 1

    predictive_samples[predictive_samples < 0] = 0
    prophet_forecast = prophet_forecast.drop(columns=["is_historical", "is_recent"], errors="ignore")
    return predictive_samples, m, prophet_forecast


def mobile_forecast_model(
    historical_data,
    historical_dates,
    forecast_dates,
    recent_weeks=13,
    changepoint_prior_scale=0.02,
    changepoint_range=0.82,
    n_changepoints=25,
    seasonality_prior_scale=0.1,
    seasonality_regime="auto",
):
    assert seasonality_regime in ("auto", "additive", "multiplicative"), (
        f"seasonality_regime must be auto/additive/multiplicative, "
        f"got {seasonality_regime!r}"
    )
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": False,
        "yearly_seasonality": len(historical_data.dropna()) > (365 * 2),
        "uncertainty_samples": 1000,
        "changepoint_range": changepoint_range,
        "n_changepoints": n_changepoints,
        "growth": "logistic",
    }

    if historical_data.max() >= 1e6:
        params["seasonality_prior_scale"] = seasonality_prior_scale
        params["changepoint_prior_scale"] = changepoint_prior_scale
        params["growth"] = "linear"

    # "auto" keeps mobile's volume-threshold mode switch; forcing a regime only
    # sets seasonality_mode (growth stays volume-driven, unlike desktop).
    if seasonality_regime == "multiplicative" or (
        seasonality_regime == "auto" and historical_data.max() <= 2e6
    ):
        params["seasonality_mode"] = "multiplicative"

    np.random.seed(42)
    m = prophet.Prophet(**params)

    historical_mask = historical_dates < forecast_dates[0]
    observed = pd.DataFrame(
        {"ds": historical_dates[historical_mask], "y": historical_data[historical_mask]}
    ).copy(deep=True)
    future = pd.DataFrame({"ds": forecast_dates})

    if "growth" in params:
        if historical_data.max() >= 10e6:
            cap = observed["y"].tail(426).max() * 1.10
            floor = observed["y"].tail(426).min() * 1.05
            observed["cap"] = cap
            observed["floor"] = floor
            future["cap"] = cap
            future["floor"] = floor
        else:
            cap = historical_data.max() * 1.1
            floor = 0.0
            observed["cap"] = cap
            observed["floor"] = floor
            future["cap"] = cap
            future["floor"] = floor

    observed, future = _add_conditional_weekly_seasonality(
        m, observed, future, forecast_dates[0], recent_weeks=recent_weeks
    )
    m.fit(observed)
    prophet_forecast = m.predict(future)
    predictive_samples = pd.DataFrame(m.predictive_samples(future)["yhat"])
    predictive_samples[predictive_samples < 0] = 0
    prophet_forecast = prophet_forecast.drop(columns=["is_historical", "is_recent"], errors="ignore")
    return predictive_samples, m, prophet_forecast
