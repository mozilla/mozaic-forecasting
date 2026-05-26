# mozaic-forecasting

Prophet-based hierarchical forecasting for Mozilla Firefox metrics.

## Concepts

**Tile** — atomic forecasting unit for one `(metric, country, population)` series. On construction: builds holiday calendar → detrends holidays → fits Prophet → generates predictive samples.

**Mozaic** — groups tiles and fits a second Prophet model on the aggregated series. Three flavors: country-level (metric+country), population-level (metric+population), metric-level (all tiles for one metric).

Tiles are shared by reference across mozaics. Reconciliation modifies `tile.forecast_reconciled` in-place and those changes are immediately visible to every mozaic holding that tile.

## Quick start

```python
import mozaic
from mozaic import DesktopModelConfig, make_desktop_model

# default run — identical to hardcoded behaviour
tileset = mozaic.TileSet()
mozaic.populate_tiles(datasets, tileset, mozaic.desktop_forecast_model, start, end)

metric_mozaics, country_mozaics, population_mozaics = {}, {}, {}
mozaic.utils.curate_mozaics(datasets, tileset, mozaic.desktop_forecast_model,
                             metric_mozaics, country_mozaics, population_mozaics)
```

## Configurable parameters

Use `ModelConfig` subclasses to vary parameters without editing source code:

```python
from mozaic import DesktopModelConfig, MobileModelConfig, make_desktop_model, make_mobile_model

c = DesktopModelConfig(
    prophet_changepoint_prior_scale=0.05,   # default 0.15983
    prophet_recent_weeks=8,                  # default 13
    holiday_threshold=-0.025,                # default -0.032
    holiday_max_radius=4,                    # default 5
    holiday_min_radius=2,                    # default 3
    holiday_effect_floor=-0.5,               # default -0.6
)

# slug for labelling output directories / files
c.to_slug()   # "cps0.05_thresh025_recent8_clip0.5"
c.to_dict()   # plain dict for serialization

# build a callable that satisfies the forecast_model contract
model = make_desktop_model(c)

# pass to populate_tiles / curate_mozaics
mozaic.populate_tiles(
    datasets, tileset, model, start, end,
    holiday_threshold=c.holiday_threshold,
    holiday_max_radius=c.holiday_max_radius,
    holiday_min_radius=c.holiday_min_radius,
)
mozaic.utils.curate_mozaics(
    datasets, tileset, model,
    metric_mozaics, country_mozaics, population_mozaics,
    holiday_effect_floor=c.holiday_effect_floor,
)
```

From mozaic-daily, pass `config=` directly to the wrappers:

```python
from mozaic_daily.forecast import get_desktop_forecast_dfs
result = get_desktop_forecast_dfs(metric_data, start, end, config=c)
result.config  # config is stored on the result
```

## Forecast model contract

Any `forecast_model` callable passed to `Tile`, `Mozaic`, `populate_tiles`, or `curate_mozaics` must have this signature:

```python
f(historical_data, historical_dates, forecast_dates) -> (samples, prophet_model, prophet_forecast)
```

`make_desktop_model(config)` and `make_mobile_model(config)` return closures satisfying this contract. The raw `desktop_forecast_model` and `mobile_forecast_model` also satisfy it.

## Development

Tests are written in `tmp/` (not committed) and run against the mozaic-daily venv:

```bash
source /Users/brendanwells/work/mozaic-daily/.venv/bin/activate
python -m pytest tmp/test_config.py tmp/test_config_daily.py -v
```

See `docs/architecture.md` for a detailed walkthrough of the reconciliation and holiday effect pipeline.
