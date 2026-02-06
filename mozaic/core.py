import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, List
from scipy.optimize import minimize_scalar

from mozaic import Tile


@dataclass
class Mozaic:
    tiles: List[Tile]
    forecast_model: Any
    is_country_level: bool = False

    def __post_init__(self):

        tile_0 = self.tiles[0]

        # check that tiles are compatible
        assert all(
            type(i) == type(tile_0) for i in self.tiles
        ), "type differs across tiles"
        assert all(
            i.forecast_start_date == tile_0.forecast_start_date for i in self.tiles
        ), "forecast_start_date differs across tiles"
        assert all(
            i.forecast_end_date == tile_0.forecast_end_date for i in self.tiles
        ), "forecast_end_date differs across tiles"
        assert all(
            i.historical_dates.equals(tile_0.historical_dates) for i in self.tiles
        ), "historical_dates differ across tiles"
        assert all(
            i.forecast_dates.equals(tile_0.forecast_dates) for i in self.tiles
        ), "forecast_dates differ across tiles"

        # set direct copy attributes using tile_0
        for i in [
            "forecast_start_date",
            "forecast_end_date",
            "historical_dates",
            "forecast_dates",
        ]:
            setattr(self, i, getattr(tile_0, i))

        # set collection attributes based on tile values
        for i in ["metric", "country", "population"]:
            values = []
            for tile in self.tiles:
                x = getattr(tile, i)
                if isinstance(x, list):
                    values.extend(x)
                else:
                    values.append(x)
            setattr(self, i, np.unique(values).tolist())

        self.holiday_calendar = (
            pd.concat(i.holiday_calendar for i in self.tiles)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # set aggregate attribues
        for i in ["raw_historical_data", "holiday_detrended_historical_data"]:
            y = np.sum(
                [getattr(tile, i).astype(float).fillna(0.0) for tile in self.tiles],
                axis=0,
            )
            setattr(self, i, pd.Series(y).replace({np.nan: 0.0}))

        # get mozaic-level forcast
        self.forecast, self._prophet_model, self._prophet_forecast = (
            self.forecast_model(
                self.holiday_detrended_historical_data.replace({0: np.nan}),
                self.historical_dates,
                self.forecast_dates,
            )
        )
        self.trend = self._prophet_forecast.trend
        self.forecast_reconciled = self.forecast.copy(deep=True)

        # set this mozaic as the mozaic for all tiles
        for i in self.tiles[1:]:
            i.mozaic = self

    def reset_reconciliation(self):
        self.forecast_reconciled = self.forecast.copy(deep=True)
        for tile in self.tiles:
            tile.forecast_reconciled = tile.forecast.copy(deep=True)

    def reconcile_bottom_up(self):
        for tile in self.tiles:
            if isinstance(tile, Mozaic):
                tile.reconcile_bottom_up()

        # Rebuild this node's forecast as sum of reconciled children
        self.forecast_reconciled = sum(tile.forecast_reconciled for tile in self.tiles)

    def _reconcile_top_down_by_quantiles(self, bounds=(0.40, 0.60)):
        def mae(q):
            topline_target = self.forecast_reconciled.median(axis=1).rolling(28).mean()
            component_sums = (
                sum(tile.forecast_reconciled.quantile(q, axis=1) for tile in self.tiles)
                .rolling(28)
                .mean()
            )
            return (topline_target - component_sums).abs().mean()

        q = minimize_scalar(
            mae, bounds=bounds, method="bounded", options={"xatol": 1e-6}
        ).x

        # apply the rescaling factor down-ladder
        for tile in self.tiles:
            diff = tile.forecast_reconciled.median(
                axis=1
            ) - tile.forecast_reconciled.quantile(q, axis=1)
            tile.forecast_reconciled = tile.forecast_reconciled.add(diff, axis=0)

            if isinstance(tile, Mozaic):
                tile.reconcile_top_down()

    def _reconcile_top_down_by_rescaling(self, use_holidays=False):
        def get_weight(tile):
            # note: var works better than std
            return tile.var(axis=1) / tile.mean(axis=1) + tile.mean(axis=1)

        if use_holidays:
            _topline = self.forecast_reconciled + self.forecasted_holiday_impacts
            _tiles = [
                tile.forecast_reconciled + tile.forecasted_holiday_impacts
                for tile in self.tiles
            ]
        else:
            _topline = self.forecast_reconciled
            _tiles = [tile.forecast_reconciled for tile in self.tiles]

        # calculate the rescaling factor
        diff = _topline.median(axis=1) - sum(tile.median(axis=1) for tile in _tiles)
        weights = [get_weight(tile) for tile in _tiles]
        weight_sum = sum(weights)

        # apply the rescaling factor down-ladder
        for tile, weight in zip(self.tiles, weights):
            delta = diff.mul(weight / weight_sum, axis=0)
            tile.forecast_reconciled = tile.forecast_reconciled.add(delta, axis=0)

            if isinstance(tile, Mozaic):
                tile.reconcile_top_down()

    def reconcile_top_down(self, use_holidays=False):
        self._reconcile_top_down_by_quantiles()
        self._reconcile_top_down_by_rescaling(use_holidays)

    def _fit_holiday_effects(self, window: int = 7):
        """
        Estimate proportional holiday effects using this mozaic's observed and expected historical data.

        Stores a flat DataFrame with columns:
            holiday, date_diff, average_effect, n_years
        """

        # TODO: check if this is the country-level anchor; if not self.observed_holiday_effects will be set on another level

        df = pd.DataFrame(
            {
                "submission_date": self.historical_dates,
                "observed": self.raw_historical_data,
                "expected": self.holiday_detrended_historical_data,
            }
        )

        # Filter out future data
        df = df[df["submission_date"] < pd.to_datetime(self.forecast_start_date)].copy()

        # Cross-join with holiday dates
        merged = df.merge(
            self.holiday_calendar, how="cross", suffixes=("_dau", "_holiday")
        )

        # Calculate offset from each holiday
        merged["date_diff"] = (
            merged["submission_date_dau"] - merged["submission_date_holiday"]
        ).dt.days
        merged = merged[merged["date_diff"].between(-window, window)].copy()

        # Exclude system holidays like "Data Loss"
        merged = merged[~merged["holiday"].str.contains("Data Loss", na=False)].copy()

        # Compute impact (observed - expected) / expected
        merged["proportional_effect"] = (
            merged["observed"] - merged["expected"]
        ) / merged["expected"]

        # Inverse distance weighting
        merged["weight"] = 1 / (1 + merged["date_diff"].abs())
        merged["scale"] = merged["weight"] / merged.groupby("submission_date_dau")[
            "weight"
        ].transform("sum")
        merged["weighted_effect"] = merged["proportional_effect"] * merged["scale"]

        # Expand semicolon-delimited holidays if needed
        merged = merged.assign(holiday=merged["holiday"].str.split("; ")).explode(
            "holiday"
        )

        # Aggregate
        self.observed_holiday_effects = (
            merged.groupby(["holiday", "date_diff"])
            .agg(
                average_effect=("weighted_effect", "mean"),  # correct averaging
                all_effects=("weighted_effect", list),  # helpful for diagnostics
                n_years=("submission_date_holiday", lambda x: x.dt.year.nunique()),
            )
            .reset_index()
        )

    def _predict_holiday_effects(self):
        """
        Compute proportional holiday effect per forecast date (unitless).
        These are stored as a time-indexed Series and not yet applied to forecasts.
        """

        # Cross-join forecast dates and holiday dates
        df = pd.merge(
            pd.DataFrame({"date": self.forecast_dates}),
            self.holiday_calendar.rename(columns={"submission_date": "holiday_date"}),
            how="cross",
        )

        df["date_diff"] = (df["date"] - df["holiday_date"]).dt.days
        df = df[df["date_diff"].between(-7, 7)]
        df = df.assign(holiday=df["holiday"].str.split("; ")).explode("holiday")

        df = df.merge(
            self.observed_holiday_effects, how="left", on=["holiday", "date_diff"]
        )

        # Exclude system holidays like "Data Loss"
        df = df[~df["holiday"].str.contains("Data Loss", na=False)].copy()
        unmatched = df[df["average_effect"].isna()]["holiday"].unique()

        # --- simple inline DOW scaling computed from history ---
        # build a tiny historical merged table like _fit_holiday_effects but local
        hist = pd.DataFrame(
            {
                "submission_date": self.historical_dates,
                "observed": self.raw_historical_data,
                "expected": self.holiday_detrended_historical_data,
            }
        )
        hist = hist[
            hist["submission_date"] < pd.to_datetime(self.forecast_start_date)
        ].copy()

        hist_merged = hist.merge(
            self.holiday_calendar, how="cross", suffixes=("_dau", "_holiday")
        )
        hist_merged["date_diff"] = (
            hist_merged["submission_date_dau"] - hist_merged["submission_date_holiday"]
        ).dt.days
        # same +/-7 window as your pipeline
        hist_merged = hist_merged[hist_merged["date_diff"].between(-7, 7)].copy()
        hist_merged = hist_merged.assign(
            holiday=hist_merged["holiday"].str.split("; ")
        ).explode("holiday")
        hist_merged = hist_merged[
            ~hist_merged["holiday"].str.contains("Data Loss", na=False)
        ].copy()

        # proportional residual and the same inverse-distance weighting
        hist_merged["proportional_effect"] = (
            hist_merged["observed"] - hist_merged["expected"]
        ) / hist_merged["expected"]
        hist_merged["weight"] = 1 / (1 + hist_merged["date_diff"].abs())
        hist_merged["scale"] = hist_merged["weight"] / hist_merged.groupby(
            "submission_date_dau"
        )["weight"].transform("sum")
        hist_merged["weighted_effect"] = (
            hist_merged["proportional_effect"] * hist_merged["scale"]
        )

        # aggregate by dow and shrink with a tiny pseudocount
        hist_merged["dow"] = hist_merged["submission_date_dau"].dt.dayofweek
        dow_agg = (
            hist_merged.groupby("dow")
            .agg(
                weighted_sum=("weighted_effect", "sum"),
                weight_total=("scale", "sum"),
            )
            .reindex(range(7), fill_value=0)
        )

        prior = 5.0  # tiny pseudocount to avoid extreme values with sparse data
        shrunk_mean = dow_agg["weighted_sum"] / (dow_agg["weight_total"] + prior)

        # final multiplier per dow: 1.0 means no change
        dow_scale = pd.Series(1.0 + shrunk_mean.values, index=range(7))

        # apply multiplier to the forecast rows
        df["dow"] = df["date"].dt.dayofweek
        df["average_effect"] = df["average_effect"].astype(float).fillna(0.0)
        df["average_effect"] = df["average_effect"] * df["dow"].map(dow_scale)

        # then aggregate as before
        self.proportional_holiday_effects = (
            df.groupby("date")["average_effect"]
            .sum()
            .reindex(self.forecast_dates, fill_value=0)
        )

        return unmatched.tolist() or None

    def assign_holiday_effects(self):
        # If this mozaic is responsible for calculating holiday effects, do so
        if self.is_country_level:
            self._fit_holiday_effects()
            unmatched = self._predict_holiday_effects()

        for tile in self.tiles:
            # If this mozaic has proportional effects, pass them to child tiles
            if hasattr(self, "proportional_holiday_effects"):
                tile.proportional_holiday_effects = self.proportional_holiday_effects

            # if the child is a mozaic, run again on the children
            if isinstance(tile, Mozaic):
                tile.assign_holiday_effects()

        return unmatched

    def aggregate_holiday_impacts_upward(self, use_reconciled=True):
        impact_series = None
        if use_reconciled:
            attr = "forecast_reconciled"
        else:
            attr = "forecast"

        for tile in self.tiles:
            if isinstance(tile, Mozaic):
                tile._aggregate_holiday_impacts_upward()
                child_impact = tile.forecasted_holiday_impacts
            else:
                if not hasattr(tile, "forecasted_holiday_impacts"):
                    tile.forecasted_holiday_impacts = getattr(tile, attr).multiply(
                        tile.proportional_holiday_effects.reset_index(drop=True), axis=0
                    )
                child_impact = tile.forecasted_holiday_impacts

            if impact_series is None:
                impact_series = child_impact.copy(deep=True)
            else:
                impact_series += child_impact

        self.forecasted_holiday_impacts = impact_series

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
