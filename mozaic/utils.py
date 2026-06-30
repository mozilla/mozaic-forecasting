import holidays
import numpy as np
import pandas as pd

from dataclasses import field
from typing import List, Optional, Type

from mozaic import Mozaic, Tile, fills


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


def splice_fill(datasets: dict, fill: pd.DataFrame, country: str = None) -> dict:
    """
    Replace a country's in-window rows with counterfactual fill rows, per metric.

    Parameters:
        datasets (dict): metric -> dataframe (columns x, country, segment bools, y).
        fill (pd.DataFrame): long-by-metric fill frame; the same columns plus "metric".
            Each metric's window is taken from its own fill date coverage.
        country (str): country to replace; inferred from the fill when not given.

    Returns:
        dict: a new metric -> dataframe mapping; inputs are not mutated. A metric absent
        from the fill -- or from which the country is entirely absent -- passes through
        unchanged: the fill replaces existing rows, it never injects a missing country.
    """
    if "metric" not in fill.columns:
        raise ValueError("fill is missing a 'metric' column")
    if country is None:
        countries = fill["country"].unique()
        if len(countries) != 1:
            raise ValueError(f"fill spans countries {sorted(countries)}; pass country=")
        country = countries[0]
    elif not (fill["country"] == country).all():
        raise ValueError(f"fill contains non-{country} rows")

    fill_metrics = set(fill["metric"].unique())
    out = {}
    for metric, dataset in datasets.items():
        df = dataset.copy(deep=True)
        # Replace existing rows only; never inject a country the dataset lacks
        if metric not in fill_metrics or not (df["country"] == country).any():
            out[metric] = df
            continue

        fm = fill[fill["metric"] == metric].drop(columns="metric").copy(deep=True)
        if set(fm.columns) != set(df.columns):
            raise ValueError(
                f"{metric}: fill columns {sorted(fm.columns)} "
                f"!= dataset columns {sorted(df.columns)}"
            )

        # Window is the fill's own coverage (the fill carries no out-of-window rows)
        fm["x"] = pd.to_datetime(fm["x"])
        lo, hi = fm["x"].min(), fm["x"].max()

        df["x"] = pd.to_datetime(df["x"])
        is_country = df["country"] == country
        in_window = is_country & df["x"].between(lo, hi)
        if not (is_country & (df["x"] > hi)).any():
            print(f"⚠️  {metric}: no real {country} rows after {hi.date()} to anchor the seam")

        # Drop the real in-window rows, append the fill (aligned column order)
        out[metric] = pd.concat([df[~in_window], fm[df.columns]], ignore_index=True)

    return out


def _pivot_populations(dataset, country):
    df = dataset[dataset["country"] == country].copy(deep=True)
    cols = list(set(df.columns) - {"x", "y", "country"})
    df["population"] = (
        df[cols]
        .apply(lambda row: "_".join(col for col in cols if row[col]), axis=1)
        .replace("", "other")  # Replace empty strings with "other"
    )
    return (
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


def populate_tiles(
    datasets,
    tileset,
    forecast_model,
    forecast_start_date,
    forecast_end_date,
    additional_holidays: List[Type[holidays.HolidayBase]] = [],
    holiday_threshold: float = -0.032,
    holiday_max_radius: int = 5,
    holiday_min_radius: int = 3,
    data_source: Optional[str] = None,
    additional_fills: Optional[List[pd.DataFrame]] = None,
):
    # Built-in counterfactual gap fills (e.g. the Iran 2026 outage) apply automatically,
    # like a country's holiday calendar; additional_fills adds caller-supplied ones.
    # Only consult the registry when a country it covers is actually present, so ordinary
    # runs stay silent (no work, no ambiguity warning).
    gap_fills = list(additional_fills or [])
    registered = fills.registered_fills()
    present = set().union(*(set(d["country"]) for d in datasets.values()))
    covered = {c for _, f in registered for c in f["country"].unique()}
    if present & covered:
        seg = set(next(iter(datasets.values())).columns) - {"x", "y", "country"}
        gap_fills = fills.fills_for(seg, data_source=data_source) + gap_fills

    synthetic_datasets = datasets
    for fill in gap_fills:
        synthetic_datasets = splice_fill(synthetic_datasets, fill)
    has_fill = synthetic_datasets is not datasets

    for metric, dataset in datasets.items():
        print("\n" + metric)

        for country in dataset.country.unique():
            print("\n" + country.rjust(3), end=": ")

            df = _pivot_populations(dataset, country)
            populations = [c for c in df.columns if c not in ("x", "country")]

            synth = None
            if has_fill:
                synth = _pivot_populations(synthetic_datasets[metric], country)

            for population in populations:
                if len(df[population].dropna()) > 30:
                    print(population, end=", ")

                    # Counterfactual series for this tile, only where it differs from real
                    synthetic = None
                    if synth is not None and population in synth.columns:
                        s = synth.set_index("x")[population].reindex(df["x"])
                        s = s.reset_index(drop=True)
                        if not s.equals(df[population].reset_index(drop=True)):
                            synthetic = s

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
                            synthetic_historical_data=synthetic,
                            additional_holidays=additional_holidays,
                            threshold=holiday_threshold,
                            max_radius=holiday_max_radius,
                            min_radius=holiday_min_radius,
                        )
                    )
        print()


def curate_mozaics(
    datasets,
    tileset,
    forecast_model,
    metric_mozaics,
    country_mozaics,
    population_mozaics,
    holiday_effect_floor: float = -0.6,
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
                holiday_effect_floor=holiday_effect_floor,
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
                holiday_effect_floor=holiday_effect_floor,
            )

        print("\n   reconciling...")
        metric_mozaics[m] = Mozaic(
            tileset.fetch(metric=m),
            forecast_model=forecast_model,
            holiday_effect_floor=holiday_effect_floor,
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
