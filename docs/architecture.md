# Mozaic Forecasting — Architecture Overview

## Core Concepts

**Tile** — the atomic forecasting unit. Each tile represents a single `(metric, country, population)` combination (e.g. `desktop_dau | US | new_profile`). On construction, a tile:
1. Builds a holiday calendar for its country
2. Detrends holidays from the raw historical data (a physics-inspired kinematic smoother that detects and removes anomalies near holiday windows)
3. Fits a Prophet model on the detrended data and generates a distribution of predictive samples

**Mozaic** — a grouping of tiles that fits a second Prophet model on the aggregated (summed) series. Mozaics come in three flavors, all of which hold flat lists of shared `Tile` objects:

| Mozaic type | Groups tiles by | Example |
|---|---|---|
| Country | metric + country | all populations for desktop DAU in the US |
| Population | metric + population | all countries for desktop DAU, new profiles only |
| Metric | metric only | every tile for desktop DAU |

## Object Structure

Tiles are shared by reference across mozaics — the same Python object appears in the country mozaic, the population mozaic, and the metric mozaic simultaneously. There is no nested tree; all mozaics hold leaf tiles directly.

```
metric_mozaic["desktop_dau"]          ← Prophet fit on global sum
  tiles: [all Tile objects]

country_mozaic["desktop_dau"]["US"]   ← Prophet fit on US sum
  tiles: [Tile(US, new_profile), Tile(US, other), ...]

country_mozaic["desktop_dau"]["DE"]   ← Prophet fit on DE sum
  tiles: [Tile(DE, new_profile), Tile(DE, other), ...]

population_mozaic["desktop_dau"]["new_profile"]   ← Prophet fit on new_profile sum
  tiles: [Tile(US, new_profile), Tile(DE, new_profile), ...]
```

## Prophet Model Fitting

Each tile and each mozaic holds its own fitted Prophet model at `._prophet_model`. A global forecast therefore involves many independent Prophet fits:

- One per tile (`N_countries × N_populations` fits)
- One per country mozaic (`N_countries` fits)
- One per population mozaic (`N_populations` fits)
- One for the metric mozaic (1 fit)

The model uses logistic growth (with cap/floor derived from recent data) or log-linear growth depending on data characteristics. Yearly seasonality is disabled for series with fewer than two years of history.

After fitting, each model immediately generates predictive samples (`m.predictive_samples()`), which become the forecast distribution. The fitted model object is retained in memory but not serialized or reused for further prediction.

## Holiday Effects

Holiday effects are estimated and applied at the **country level**, then propagated downward to tiles. The process:

1. For each country mozaic, compare raw vs. detrended historical data near each holiday to estimate proportional effects (observed − expected) / expected, weighted by inverse distance to the holiday date
2. Effects are scaled by day-of-week and clipped at −60% (floor) and 0% (ceiling)
3. The fitted proportional effects are shared to all child tiles via `proportional_holiday_effects`
4. `forecasted_holiday_impacts` (absolute values) are computed per tile and summed upward to mozaics

## Reconciliation

Because tiles are shared objects, reconciliation modifies `tile.forecast_reconciled` in place and those changes are immediately visible to all mozaics that reference the same tile. The reconciliation sequence for each metric is:

1. **Metric-level top-down** — finds the quantile that minimizes MAE between the global Prophet forecast and the sum of tile forecasts, shifts tile distributions accordingly; then allocates any remaining residual to tiles weighted by `variance/mean + mean` (zero-mean tiles receive zero weight)
2. **Country-level bottom-up** — each country mozaic sums its tiles' (now-adjusted) `forecast_reconciled` values to ensure internal consistency
3. **Population-level bottom-up** — same, across the population slice

The final output per tile is `forecast_reconciled + forecasted_holiday_impacts`, clipped to zero.

## Data Filters

Tiles with fewer than 30 non-null historical observations are excluded. When aggregating tiles into a mozaic, NaNs are treated as zeros so that sparse tiles do not create gaps in the country- or metric-level series.

## Configurable Parameters

All previously-hardcoded parameters are now exposed via `ModelConfig` dataclasses in `models.py`.

```
ModelConfig
├── prophet_recent_weeks       (int, default 13)
├── holiday_threshold          (float, default -0.032)
├── holiday_max_radius         (int, default 5)
├── holiday_min_radius         (int, default 3)
└── holiday_effect_floor       (float, default -0.6)

DesktopModelConfig(ModelConfig)
└── prophet_changepoint_prior_scale  (float, default 0.15983)

MobileModelConfig(ModelConfig)
└── prophet_changepoint_prior_scale  (float, default 0.02)
```

`make_desktop_model(config)` and `make_mobile_model(config)` return closures that satisfy the `forecast_model` callable contract while baking in the config values. `config.to_slug()` produces a compact string label for output directories (e.g. `cps0.02_thresh032_recent13_clip0.6`).

**Holiday effect floor** (`holiday_effect_floor`) lives on the `Mozaic` dataclass and is used in `_predict_holiday_effects` to clip the proportional holiday effects before they are applied to forecasts. A warning is emitted if any effect would be clipped.

**Holiday detrend params** (`holiday_threshold`, `holiday_max_radius`, `holiday_min_radius`) live on `Tile` (as `threshold`, `max_radius`, `min_radius`) and are forwarded from `populate_tiles` kwargs.

**Prophet CPS** (`prophet_changepoint_prior_scale`) is forwarded from the factory closure into the `params` dict inside each model function.
