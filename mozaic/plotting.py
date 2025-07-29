import pandas as pd
import plotly.graph_objects as go

from dataclasses import dataclass

from mozaic import Mozaic, Tile


@dataclass
class Line:
    column: str
    color: str
    name: str
    opacity: float = 0.8
    width: int = 1


def plot(df, lines, title="", y_title=""):
    fig = go.Figure()

    for line in lines:
        fig.add_trace(
            go.Scatter(
                x=df["submission_date"],
                y=df[line.column],
                mode="lines",
                line=dict(color=line.color, width=line.width),
                opacity=line.opacity,
                name=line.name,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="Submission Date",
        template="plotly_white",
    )

    fig.show()


def moving_average(series, window=28):
    return series.rolling(window=window).mean()


def year_over_year(df, column, date_column="submission_date"):
    # Assume df has columns: "submission_date" (datetime), "x"
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Create a copy with submission_date shifted 1 year
    df_shift = df.copy()
    df_shift[date_column] = df_shift[date_column] + pd.DateOffset(years=1)
    df_shift.rename(columns={column: f"{column}_last_year"}, inplace=True)

    # Merge original with shifted version
    df_merged = df.merge(df_shift, on=date_column, how="left")

    # Calculate YoY % change
    return (df_merged[column] / df_merged[f"{column}_last_year"]) - 1


def mozaic_plot(
    x,
    use_moving_average=True,
    title=None,
    show_detrended=True,
    zoom_after=None,
    quantile=0.5,
):

    if isinstance(x, Mozaic):
        title = (
            title
            or f"{x.metric[0]} | countries: {', '.join(x.country)} | populations: {', '.join(x.population)}"
        )
        df = x.to_df(quantile)
    elif isinstance(x, Tile):
        title = title or x.name
        df = x.to_df(quantile)
    elif isinstance(x, pd.DataFrame):
        title = title or ""
        df = x.copy(deep=True)
    else:
        raise ValueError("x must be either mozaic, Tile, or DataFrame")

    if zoom_after is not None:
        df = df[df["submission_date"] >= zoom_after]

    colors_daily = {
        "actuals_detrended": "#CF592A",
        "forecast_detrended": "#653EAD",
        "forecast_detrended_raw": "gray",
        "actuals": "#FF7139",
        "forecast": "#9059FF",
        "forecast_raw": "black",
    }

    if not show_detrended:
        colors_daily = {k: v for k, v in colors_daily.items() if "detrended" not in k}

    colors_28ma = {f"{k}_28ma": v for k, v in colors_daily.items()}

    if use_moving_average:
        colors = colors_28ma
    else:
        colors = colors_daily

    plot(
        df=df,
        lines=[
            Line(k, v, k.replace("_", " ").title(), width=2, opacity=0.7)
            for k, v in colors.items()
            if k in df.columns
        ],
        title=title,
        y_title="Metric",
    )
