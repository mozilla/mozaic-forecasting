import logging
import numpy as np
import pandas as pd
import prophet

logging.getLogger("cmdstanpy").disabled = True


def desktop_forecast_model(historical_data, historical_dates, forecast_dates):
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": False,
        "uncertainty_samples": 1000,
        "changepoint_range": 0.8,
        "seasonality_prior_scale": 0.00825,
        "changepoint_prior_scale": 0.15983,
        "growth": "logistic",
    }

    x = historical_data

    # if x.max() >= 10e6:
    #     params["growth"] = "logistic"

    if (x.abs().corr(x.diff().abs()) or 0) > 0.0:
        params["seasonality_mode"] = "multiplicative"
        params["growth"] = "linear"
        # params["seasonality_prior_scale"] = 20
        # params["changepoint_prior_scale"] = 0.05
        # params["growth"] = "logistic"

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

    # if "growth" in params:
    #     if historical_data.max() >= 10e6:
    #         cap =  observed["y"].max() * 1.2
    #         floor = observed["y"].min() * 0.8
    #         observed["cap"] = cap
    #         observed["floor"] = floor
    #         future["cap"] = cap
    #         future["floor"] = floor
    #     else:
    #         cap = observed["y"].max() * 1.2
    #         floor = 0.0
    #         observed["cap"] = cap
    #         observed["floor"] = floor
    #         future["cap"] = cap
    #         future["floor"] = floor

    if params["growth"] == "logistic":
        cap = observed["y"].max() * 1.2
        floor = observed["y"].min() * 0.8
        observed["cap"] = cap
        observed["floor"] = floor
        future["cap"] = cap
        future["floor"] = floor
    else:
        with np.errstate(invalid="ignore"):
            observed["y"] = np.log(observed["y"] + 1.0)

    np.random.seed(42)
    m = prophet.Prophet(**params)
    m.fit(observed)

    prophet_forecast = m.predict(future)
    predictive_samples = pd.DataFrame(m.predictive_samples(future)["yhat"])

    if params["growth"] == "linear":
        predictive_samples = np.exp(predictive_samples) - 1

    predictive_samples[predictive_samples < 0] = 0
    return predictive_samples, m, prophet_forecast


def mobile_forecast_model(historical_data, historical_dates, forecast_dates):
    params = {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": len(historical_data.dropna()) > (365 * 2),
        "uncertainty_samples": 1000,
        "changepoint_range": 0.8,
        "growth": "logistic",
    }

    if historical_data.max() >= 1e6:
        params["seasonality_prior_scale"] = 0.1
        params["changepoint_prior_scale"] = 0.1

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
            cap = historical_data.max() * 2.0
            floor = historical_data.min() * 0.8
            observed["cap"] = cap
            observed["floor"] = floor
            future["cap"] = cap
            future["floor"] = floor
        else:
            cap = historical_data.max() * 2.0
            floor = 0.0
            observed["cap"] = cap
            observed["floor"] = floor
            future["cap"] = cap
            future["floor"] = floor

    m.fit(observed)
    prophet_forecast = m.predict(future)
    predictive_samples = pd.DataFrame(m.predictive_samples(future)["yhat"])
    predictive_samples[predictive_samples < 0] = 0
    return predictive_samples, m, prophet_forecast
