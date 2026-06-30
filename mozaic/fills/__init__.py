import importlib.resources as resources

import pandas as pd

from functools import lru_cache

# Built-in counterfactual gap fills shipped with the package and applied automatically
# during populate_tiles (like a country's holiday calendar). Each entry replaces a
# country's collapsed telemetry over a known outage window with a synthetic "what would
# have happened" series, per data source. Registering a new gap is deliberate, not
# routine -- add a folder under fills/ and a registry entry here.
_REGISTRY = {
    # gap id -> {data_source -> packaged parquet relative path}
    "iran_2026": {
        "glean_desktop": "iran_2026/glean_desktop.parquet",
        "legacy_desktop": "iran_2026/legacy_desktop.parquet",
        "glean_mobile": "iran_2026/glean_mobile.parquet",
    },
}


@lru_cache(maxsize=None)
def _load(rel):
    source = resources.files("mozaic.fills").joinpath(rel)
    with resources.as_file(source) as path:
        return pd.read_parquet(path)


def registered_fills():
    """All built-in (data_source, fill_frame) pairs shipped with the package."""
    return [
        (data_source, _load(rel))
        for gap in _REGISTRY.values()
        for data_source, rel in gap.items()
    ]


def fills_for(segment_columns, data_source=None):
    """Built-in fill frames to apply to a dataset with the given segment columns.

    With data_source, returns the frames registered for it. Without, returns frames
    whose segment schema uniquely matches the dataset; an ambiguous match (e.g. the
    two desktop sources share a schema) is skipped with a warning -- pass data_source
    to disambiguate.
    """
    pairs = registered_fills()
    if data_source is not None:
        return [f for s, f in pairs if s == data_source]

    seg = set(segment_columns)
    matched = [
        (s, f)
        for s, f in pairs
        if set(f.columns) - {"metric", "x", "y", "country"} == seg
    ]
    sources = {s for s, _ in matched}
    if len(sources) > 1:
        print(
            f"⚠️  ambiguous built-in gap fill for sources {sorted(sources)}; "
            "pass data_source= to select one"
        )
        return []
    return [f for _, f in matched]
