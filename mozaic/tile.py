import holiday_smart
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, List, Type

from mozaic.holiday_smart import get_calendar, detrend


@dataclass
class Tile:
    metric: str
    country: str
    population: str
    forecast_start_date: str
    forecast_end_date: str
    forecast_model: Any

    historical_dates: pd.Series
    raw_historical_data: pd.Series

    additional_holidays: List[Type[holidays.HolidayBase]] = field(default_factory=list)
    threshold: float = -0.05
    max_radius: int = 5
    min_radius: int = 3

    def __post_init__(self):
        self.mozaic = None
        self.name = f"{self.metric} | {self.country} | {self.population}"

        self.historical_dates = pd.to_datetime(self.historical_dates)
        self.forecast_dates = pd.date_range(
            self.forecast_start_date, self.forecast_end_date
        )

        self._set_holiday_calendar()
        self._detrend_holidays()
        self._run_forecast()

    def _set_holiday_calendar(self):
        self.holiday_calendar = get_calendar(
            country=self.country,
            holiday_years=self.historical_dates.dt.year.unique(),
            split_concurrent_holidays=False,
            additional_holidays=self.additional_holidays,
        )

    def _detrend_holidays(self):
        self.holiday_detrended_historical_data = detrend(
            dates=self.historical_dates,
            y=self.raw_historical_data,
            holiday_df=self.holiday_calendar,
            threshold=self.threshold,
            max_radius=self.max_radius,
            min_radius=self.min_radius,
        )

    def _run_forecast(self):
        self.forecast, self._prophet_model, self._prophet_forecast = (
            self.forecast_model(
                self.holiday_detrended_historical_data.replace({0: np.nan}),
                self.historical_dates,
                self.forecast_dates,
            )
        )
        self.trend = self._prophet_forecast.trend.copy(deep=True)
        self.forecast_reconciled = self.forecast.copy(deep=True)

    def to_df(self, quantile=0.5):
        actuals_df = pd.DataFrame(
            {
                "submission_date": self.historical_dates.values,
                "actuals": self.raw_historical_data,
                "actuals_detrended": self.holiday_detrended_historical_data,
            }
        )

        forecast_df = pd.DataFrame(
            {
                "submission_date": self.forecast_dates.values,
                "forecast_detrended_raw": self.forecast.quantile(quantile, axis=1),
            }
        )

        if hasattr(self, "forecasted_holiday_impacts"):
            forecast_df["forecast_raw"] = (
                self.forecast + self.forecasted_holiday_impacts
            ).quantile(quantile, axis=1)

        if hasattr(self, "forecast_reconciled"):
            forecast_df["forecast_detrended"] = self.forecast_reconciled.quantile(
                quantile, axis=1
            )

        if hasattr(self, "forecasted_holiday_impacts") and hasattr(
            self, "forecast_reconciled"
        ):
            forecast_df["forecast"] = (
                self.forecast_reconciled + self.forecasted_holiday_impacts
            ).quantile(quantile, axis=1)

        df = actuals_df.merge(forecast_df, on="submission_date", how="outer")

        for i in df.columns:
            if "forecast" in i:
                a = "actuals_detrended" if "detrended" in i else "actuals"
                mask = df[i].isna()
                df[f"{i}_28ma"] = (df[i].fillna(df[a])).rolling(28).mean().mask(mask)
        for i in df.columns:
            if "actuals" in i:
                df[f"{i}_28ma"] = df[i].rolling(28).mean()

        return df
