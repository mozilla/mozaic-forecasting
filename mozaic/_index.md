# mozaic package — module index

Core forecasting library. Each module has a single responsibility; see below for where to find and add code.

## Modules

| Module | What's in it | What isn't |
|---|---|---|
| `tile.py` | `Tile` dataclass — one (metric, country, population) series. Holiday calendar build, holiday detrend, per-tile Prophet fit | Aggregation, reconciliation |
| `core.py` | `Mozaic` dataclass — groups tiles, fits aggregate Prophet model, estimates/applies holiday effects, reconciles forecasts | Data loading, tile construction |
| `models.py` | `ModelConfig` / `DesktopModelConfig` / `MobileModelConfig` dataclasses; `make_desktop_model` / `make_mobile_model` factory fns; `desktop_forecast_model` / `mobile_forecast_model` raw callables | Application-level orchestration |
| `utils.py` | `populate_tiles` (builds TileSet from datasets), `curate_mozaics` (orchestrates country/population/metric mozaics), `mozaic_divide` | Core forecasting logic |
| `tile_set.py` | `TileSet` — indexed container for tiles, supports fetch by metric/country/population | |
| `holiday_smart.py` | `get_calendar` (builds country holiday calendar), `detrend` (kinematic smoother that removes holiday anomalies from historical data) | |
| `plotting.py` | Visualization helpers | |
| `__init__.py` | Public surface: `Tile`, `Mozaic`, `TileSet`, `ModelConfig`, `DesktopModelConfig`, `MobileModelConfig`, `make_desktop_model`, `make_mobile_model`, `populate_tiles`, `curate_mozaics`, `mozaic_divide` | |

## Where new code goes

- **New Prophet model variant**: add to `models.py` alongside `desktop_forecast_model`
- **New config param**: add field to `ModelConfig` (or a subclass), thread through `make_*_model` closures, forward in `populate_tiles` / `curate_mozaics`
- **New holiday logic**: `holiday_smart.py` (detrend algorithm) or `core.py` (effect estimation/application)
- **New reconciliation strategy**: `core.py` as a `_reconcile_*` method on `Mozaic`
- **New output format**: `core.py` `to_df` / `to_granular_forecast_df` or a new method

## Model callable contract

The `forecast_model` argument throughout the package must be a callable with signature:

```python
f(historical_data: pd.Series, historical_dates: pd.Series, forecast_dates: pd.DatetimeIndex)
    -> (predictive_samples: pd.DataFrame, prophet_model, prophet_forecast: pd.DataFrame)
```

`make_desktop_model(config)` and `make_mobile_model(config)` return closures that satisfy this contract. The raw `desktop_forecast_model` and `mobile_forecast_model` also satisfy it (with default params). Do not break this contract — package users pass custom callables here.
