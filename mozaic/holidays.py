import holidays
import numpy as np
import pandas as pd

from dataclasses import field
from datetime import datetime
from dateutil.easter import easter
from typing import Optional

NO_PASCHAL_CYCLE = ["IN", "JP", "IR", "CN"]


class PaschalCycleHolidays(holidays.HolidayBase):
    """
    Custom holiday calendar for key Christian holidays, mostly related to the
    Paschal cycle. Includes Mardi Gras, Palm Sunday, Good Friday, Easter,
    Ascension Day, Corpus Christi. Also includes All Saints' Day, which is not
    part of the Paschal Cycle.
    """

    def _populate(self, year):
        # Get Easter Sunday for the given year
        easter_sunday = easter(year)

        # Add pre- and post-Easter holidays
        self[easter_sunday - pd.Timedelta(days=47)] = "Mardi Gras"
        self[easter_sunday - pd.Timedelta(days=7)] = "Palm Sunday"
        self[easter_sunday - pd.Timedelta(days=2)] = "Good Friday"
        self[easter_sunday] = "Easter Sunday"
        self[easter_sunday + pd.Timedelta(days=39)] = "Ascension Day"
        self[easter_sunday + pd.Timedelta(days=60)] = "Corpus Christi"

        # Fixed-date holiday
        self[pd.Timestamp(year=year, month=11, day=1)] = "All Saints' Day"


class MozillaHolidays(holidays.HolidayBase):
    """
    Custom holiday calendar for Mozilla-specific historical Data Loss incidents.
    Currently includes Data Loss events in May and July 2019.
    """

    def _populate(self, year):
        # Only populate holidays if the year includes 2019
        if year == 2019:
            # Data Loss in May 6–13, 2019
            for day in range(6, 14):
                self[datetime(2019, 5, day).date()] = "Data Loss"

            # Additional Data Loss in July 15–17, 2019
            for day in range(15, 18):
                self[datetime(2019, 7, day).date()] = "Data Loss"


class DesktopBugs(holidays.HolidayBase):
    """
    Custom holiday calendar for Desktop-specific metric-impacting bugs.
    """

    def _populate(self, year):
        if year == 2025:
            # Incident: 2025-10-08 Legacy Telemetry Drop in 143.0.4
            # https://docs.google.com/document/d/1_qKP4LUNKfmatCR4OQxuBYJxMZXrvIixBqIcQcmKKws
            for day in pd.date_range("2025-10-04", "2025-10-31"):
                self[day.date()] = "Legacy Telemetry Drop in 143.0.4"


class IranInternetBlackout(holidays.HolidayBase):
    """
    Custom holiday calendar for Iran-specific historical Data Loss incidents.
    """

    def _populate(self, year):
        # Only populate holidays if the year includes 2019
        if year == 2025:
            # Internet Blackout June 18-27, 2025
            for day in range(18, 28):
                self[datetime(2025, 6, day).date()] = "Data Loss"


class CanadianHolidays(holidays.HolidayBase):
    """
    Custom holiday for Canadian holidays, which aren't always accounted for properly by default.
    """

    def _populate(self, year):
        # Fixed-date holiday
        self[pd.Timestamp(year=year, month=11, day=11)] = "Remembrance Day"


def get_calendar(
    country: str,
    holiday_years: list,
    exclude_paschal_cycle: list = NO_PASCHAL_CYCLE,
    split_concurrent_holidays: bool = False,
    additional_holidays: list = field(default_factory=list),
) -> pd.DataFrame:
    """
    Generate a cleaned and formatted DataFrame of holidays for a specific country.

    Args:
        country (str): Country code (e.g., "US", "FR", "ROW").
        holiday_years (list): List of years to include holidays from.
        exclude_paschal_cycle (list): A list of countries that aren't impacted by the Paschal Cycle.
        split_concurrent_holidays (bool): Whether to split semicolon-delimited holidays into multiple rows.

    Returns:
        pd.DataFrame: A DataFrame with holidays, labeled by date, country, and cleaned holiday name.
    """

    # Use US holidays as a default for ROW (Rest of World)
    if country == "ROW":
        country_holidays = holidays.US(years=holiday_years)
    else:
        country_holidays = getattr(holidays, country)(years=holiday_years)

    # Optionally add Paschal cycle holidays
    if country not in exclude_paschal_cycle:
        country_holidays += PaschalCycleHolidays(years=holiday_years)

    # Include Mozilla-specific holidays for 2019
    if 2019 in holiday_years:
        country_holidays += MozillaHolidays(years=holiday_years)

    # Include country-specific additional holidays
    if country == "IR":
        country_holidays += IranInternetBlackout(years=holiday_years)

    if country == "CA":
        country_holidays += CanadianHolidays(years=holiday_years)

    # Add any additional user-specified holidays
    for i in additional_holidays:
        country_holidays += i(years=holiday_years)

    # Convert holiday dictionary into DataFrame
    df = pd.DataFrame(
        {
            "submission_date": list(country_holidays.keys()),
            "holiday": list(country_holidays.values()),
            "country": country,
        }
    )

    # Clean holiday name text
    df["holiday"] = df["holiday"].str.replace(
        r"Day off \(substituted.*", "day off (substituted)", regex=True
    )
    df["holiday"] = df["holiday"].str.replace(" (observed)", "", regex=False)

    # Split concurrent holidays into separate rows if requested
    if split_concurrent_holidays:
        df = df.assign(holiday=df["holiday"].str.split("; ")).explode(
            "holiday", ignore_index=True
        )
    else:
        # Otherwise, append the country name to each concurrent holiday
        df["holiday"] = df["holiday"].str.replace(";", f"; {country}", regex=False)

    # Prefix holiday names with country for clarity
    df["holiday"] = df["country"] + " " + df["holiday"]

    df["submission_date"] = pd.to_datetime(df["submission_date"])

    return df.sort_values(by="submission_date").reset_index(drop=True)


def detrend(
    dates: pd.Series,
    y: pd.Series,
    holiday_df: pd.DataFrame,
    threshold: float = -0.05,
    max_radius: int = 7,
    min_radius: int = 3,
    spike_correction: Optional[float] = None,
) -> pd.DataFrame:
    """
    Applies a physics-inspired detrending algorithm to smooth out holiday-driven dips in DAU.

    Parameters:

        holiday_df (pd.DataFrame): DataFrame with submission_date and holiday columns.
        threshold (float): Minimum relative difference for adjustment to apply.
        max_radius (int): Maximum distance (in days) from a holiday for adjustment.
        min_radius (int): Minimum difference (in days) that is meaningful for smoothing.
            Must be 1 <= min_radius <= max_radius. A value of 1 means that the holiday +/-1
            day all have equal weight.
        spike_correction (Optional float): Correction multiple to clamp x, v, or a values.

    Returns:
        pd.DataFrame: Input dataframe with additional columns:
            'x' (position), 'v' (velocity), 'a' (acceleration), and 'expected' (detrended DAU).
    """

    df = pd.DataFrame({"submission_date": pd.to_datetime(dates), "y": y}).copy(
        deep=True
    )

    # Calculate a spike-correction factor:
    def scaled_sigmoid(x, L=1.5, U=8, x0=5e5, k=1e-5):
        return L + (U - L) / (1 + np.exp(-k * (x - x0)))

    if spike_correction is None:
        spike_correction = scaled_sigmoid(df.y.max())

    # Merge in holiday labels
    df = df.merge(holiday_df, how="left", on="submission_date")

    # Create a holiday date lookup
    holiday_dates = df.loc[df["holiday"].notna(), "submission_date"]

    # Compute days from the nearest holiday
    df["days_from_holiday"] = df["submission_date"].apply(
        lambda date: max((holiday_dates - date).abs().min().days, min_radius)
    )

    # Initialize series for kinematic components and expected values
    _x, _v, _a, _e = [], [], [], []

    for i in df.itertuples():
        idx = i.Index

        if idx >= 21 and all(pd.notna([_e[idx - 7], _e[idx - 14], _e[idx - 21]])):
            # Get lagged expected values for position/velocity/acceleration
            lag07, lag14, lag21 = _e[idx - 7], _e[idx - 14], _e[idx - 21]

            x = lag07
            v = lag07 - lag14
            a = lag07 - 2 * lag14 + lag21

            # Compute rolling averages for recent values
            x_bar = np.mean(_x[-7:])
            v_bar = np.mean(_v[-7:])
            a_bar = np.mean(_a[-7:])

            # Clamp spikes using relative thresholds
            if abs(x) > spike_correction * abs(x_bar):
                x = x_bar
            if abs(v) > spike_correction * abs(v_bar):
                v = v_bar
            if abs(a) > spike_correction * abs(a_bar):
                a = a_bar

            # Compute expected value using position + velocity + 0.5 * acceleration
            e = (x + v + 0.5 * a) or i.y

            _x.append(x)
            _v.append(v)
            _a.append(a)

            # If within holiday radius and relative error is below threshold, apply smoothing
            if not pd.isna(i.y) and (
                (i.days_from_holiday <= max_radius) and (i.y / abs(e) - 1) < threshold
            ):
                weight = (min_radius + 1) / (i.days_from_holiday + 1)
                blended = e * weight + i.y * (1 - weight)
                _e.append(blended)
            else:
                _e.append(i.y)
        else:
            # For early points, fall back to observed value
            _x.append(np.nan)
            _v.append(np.nan)
            _a.append(np.nan)
            _e.append(i.y)

    # Attach kinematic components and expected values to DataFrame
    df["x"] = _x
    df["v"] = _v
    df["a"] = _a
    df["expected"] = _e

    return pd.to_numeric(df["expected"])
