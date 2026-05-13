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
    holiday_threshold: float = -0.032
    holiday_max_radius: int = 5
    holiday_min_radius: int = 3
    holiday_effect_floor: float = -0.6

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_slug(self):
        cps = self.prophet_changepoint_prior_scale
        thresh = f"{abs(self.holiday_threshold) * 1000:03.0f}"
        return (
            f"cps{cps}"
            f"_thresh{thresh}"
            f"_recent{self.prophet_recent_weeks}"
            f"_clip{abs(self.holiday_effect_floor)}"
        )


@dataclass
class DesktopModelConfig(ModelConfig):
    prophet_changepoint_prior_scale: float = 0.15983


@dataclass
class MobileModelConfig(ModelConfig):
    prophet_changepoint_prior_scale: float = 0.02


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


def desktop_forecast_model(historical_data, historical_dates, forecast_dates, recent_weeks=13, changepoint_prior_scale=0.15983):
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": False,
        "yearly_seasonality": True,
        "uncertainty_samples": 1000,
        "changepoint_range": 0.7,
        "seasonality_prior_scale": 0.00825,
        "changepoint_prior_scale": changepoint_prior_scale,
        "growth": "logistic",
    }

    x = historical_data

    if (x.abs().corr(x.diff().abs()) or 0) > 0.0:
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
        cap = observed["y"].tail(366).max() * 1.05
        if cap > 100e6:
            floor = observed["y"].tail(366).min() * 1
        else:
            floor = observed["y"].tail(366).min() * 0.92
        
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


def mobile_forecast_model(historical_data, historical_dates, forecast_dates, recent_weeks=13, changepoint_prior_scale=0.02):
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": False,
        "yearly_seasonality": len(historical_data.dropna()) > (365 * 2),
        "uncertainty_samples": 1000,
        "changepoint_range": 0.82,
        "growth": "logistic",
    }

    if historical_data.max() >= 1e6:
        params["seasonality_prior_scale"] = 0.1
        params["changepoint_prior_scale"] = changepoint_prior_scale
        params["growth"] = "linear"

    if historical_data.max() <= 2e6:
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
            cap = observed["y"].tail(366).max() * 1.10
            floor = observed["y"].tail(366).min() * 1.05
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
