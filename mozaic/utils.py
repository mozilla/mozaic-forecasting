import holidays
import numpy as np
import pandas as pd

from dataclasses import field
from typing import List, Type

from mozaic import Mozaic, Tile


def mozaic_divide(numerator, denominator):
    a = numerator.to_df()
    b = denominator.to_df()

    df = pd.DataFrame({"submission_date": a.submission_date})

    for i in a.columns:
        if ("_28ma" not in i) and (i != "submission_date"):
            df[i] = a[i] / b[i]

    for i in df.columns:
        if "forecast" in i:
            a = "actuals_detrended" if "detrended" in i else "actuals"
            mask = df[i].isna()
            df[f"{i}_28ma"] = (df[i].fillna(df[a])).rolling(28).mean().mask(mask)

    for i in df.columns:
        if "actuals" in i:
            df[f"{i}_28ma"] = df[i].rolling(28).mean()

    return df


def populate_tiles(
    datasets,
    tileset,
    forecast_model,
    forecast_start_date,
    forecast_end_date,
    additional_holidays: List[Type[holidays.HolidayBase]] = [],
):
    for metric, dataset in datasets.items():
        print("\n" + metric)

        for country in dataset.country.unique():
            print("\n" + country.rjust(3), end=": ")

            df = dataset[dataset["country"] == country].copy(deep=True)
            cols = list(set(df.columns) - {"x", "y", "country"})
            df["population"] = (
                df[cols]
                .apply(lambda row: "_".join(col for col in cols if row[col]), axis=1)
                .replace("", "other")  # Replace empty strings with "other"
            )
            populations = np.unique(df["population"])

            df = (
                df.pivot_table(
                    index=["x", "country"],
                    columns="population",
                    values="y",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
                .rename_axis(columns=None)
                .replace({0: np.nan})
            )

            for population in populations:
                if len(df[population].dropna()) > 30:
                    print(population, end=", ")
                    tileset.add(
                        Tile(
                            metric=metric,
                            country=country,
                            population=population,
                            forecast_start_date=forecast_start_date,
                            forecast_end_date=forecast_end_date,
                            forecast_model=forecast_model,
                            historical_dates=df["x"],
                            raw_historical_data=df[population],
                            additional_holidays=additional_holidays,
                        )
                    )


def curate_mozaics(
    datasets,
    tileset,
    forecast_model,
    metric_mozaics,
    country_mozaics,
    population_mozaics,
):
    for m in datasets.keys():
        print(m)
        print("   countries: ", end="")
        all_unmatched_holidays = []
        for c in tileset.levels(metric=m).countries:
            print(c, end=", ")
            country_mozaics[m][c] = Mozaic(
                tileset.fetch(metric=m, country=c),
                forecast_model=forecast_model,
                is_country_level=True,
            )
            unmatched = country_mozaics[m][c].assign_holiday_effects()
            if unmatched:
                all_unmatched_holidays.extend(unmatched)

        print("\n   populations: ", end="")
        for p in tileset.levels(metric=m, country=c).populations:
            print(p, end=", ")
            population_mozaics[m][p] = Mozaic(
                tileset.fetch(metric=m, population=p),
                forecast_model=forecast_model,
            )

        print("\n   reconciling...")
        metric_mozaics[m] = Mozaic(
            tileset.fetch(metric=m),
            forecast_model=forecast_model,
        )
        metric_mozaics[m].reset_reconciliation()
        metric_mozaics[m].aggregate_holiday_impacts_upward(use_reconciled=True)
        metric_mozaics[m].reconcile_top_down(use_holidays=True)

        for c in tileset.levels(metric=m).countries:
            country_mozaics[m][c].aggregate_holiday_impacts_upward(use_reconciled=True)
            country_mozaics[m][c].reconcile_bottom_up()

        for p in tileset.levels(metric=m, country=c).populations:
            population_mozaics[m][p].aggregate_holiday_impacts_upward(
                use_reconciled=True
            )
            population_mozaics[m][p].reconcile_bottom_up()

        if len(all_unmatched_holidays):
            print(
                "\n⚠️ New holidays in forecasted dates:\n - "
                + "\n - ".join(sorted(all_unmatched_holidays))
            )
    print("\ndone.")
