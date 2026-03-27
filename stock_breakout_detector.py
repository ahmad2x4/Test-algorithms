"""
Stock Breakout Detector
=======================
Identifies previous resistance levels and confirms breakouts on weekly bars.

Methodology (Weinstein Stage 2):
  - Resistance levels = swing highs detected via scipy.signal.find_peaks()
  - Breakout confirmation requires:
      1. Weekly close above a prior resistance level
      2. Breakout-bar volume > 1.5x the N-bar average volume
      3. Weekly close above the 30-week simple moving average
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_weekly_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download weekly OHLCV bars for *ticker* via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. ``"XLE"``.
    period : str
        Look-back period accepted by yfinance (e.g. ``"5y"``, ``"10y"``).

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume.  Index is a DatetimeIndex.
    """
    raw = yf.download(ticker, period=period, interval="1wk", auto_adjust=True, progress=False)

    # Flatten multi-level columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Swing-high detection
# ---------------------------------------------------------------------------

def detect_swing_highs(
    highs: pd.Series,
    lookback: int = 5,
    prominence: float = 0.5,
) -> list[int]:
    """Return integer positions of swing highs within *highs*.

    Parameters
    ----------
    highs : pd.Series
        Weekly High prices.
    lookback : int
        Minimum number of bars between consecutive peaks (``distance`` param
        of :func:`scipy.signal.find_peaks`).
    prominence : float
        Minimum prominence of a peak relative to surrounding bars.  Higher
        values filter out minor wiggles; lower values include more peaks.

    Returns
    -------
    list[int]
        Zero-based integer indices into *highs* where swing highs occur.
    """
    peaks, _ = find_peaks(highs.values, distance=lookback, prominence=prominence)
    return peaks.tolist()


# ---------------------------------------------------------------------------
# Resistance level storage
# ---------------------------------------------------------------------------

def get_resistance_levels(
    df: pd.DataFrame,
    lookback: int = 5,
    prominence: float = 0.5,
) -> list[dict]:
    """Build a list of resistance levels from swing highs in *df*.

    Each level is formed at the weekly High of a swing-high bar and is only
    considered valid for breakout detection on bars *after* it formed
    (no look-ahead bias).

    Parameters
    ----------
    df : pd.DataFrame
        Weekly OHLCV data (output of :func:`fetch_weekly_data`).
    lookback : int
        Passed to :func:`detect_swing_highs`.
    prominence : float
        Passed to :func:`detect_swing_highs`.

    Returns
    -------
    list[dict]
        Each element has keys:

        ``index``
            Integer row position in *df*.
        ``date``
            ``pd.Timestamp`` of the swing-high bar.
        ``level``
            Resistance price (the weekly High at that bar).
    """
    peak_indices = detect_swing_highs(df["High"], lookback=lookback, prominence=prominence)
    levels = []
    for idx in peak_indices:
        levels.append(
            {
                "index": idx,
                "date": df.index[idx],
                "level": float(df["High"].iloc[idx]),
            }
        )
    return levels


# ---------------------------------------------------------------------------
# 30-week moving average
# ---------------------------------------------------------------------------

def compute_ma30(close: pd.Series) -> pd.Series:
    """Return the 30-week simple moving average of *close*.

    Parameters
    ----------
    close : pd.Series
        Weekly Close prices.

    Returns
    -------
    pd.Series
        Rolling 30-bar SMA, aligned to the same index.  The first 29 bars
        will be ``NaN``.
    """
    return close.rolling(window=30).mean()


# ---------------------------------------------------------------------------
# Breakout confirmation
# ---------------------------------------------------------------------------

def confirm_breakouts(
    df: pd.DataFrame,
    resistance_levels: list[dict],
    volume_window: int = 10,
    volume_multiplier: float = 1.5,
) -> list[dict]:
    """Scan each weekly bar and flag confirmed breakouts.

    A breakout is confirmed when **all three** conditions hold simultaneously
    on the same weekly bar:

    1. **Price** – weekly Close is strictly above a prior resistance level.
    2. **Volume** – bar volume exceeds ``volume_multiplier × N-bar average``.
    3. **MA30 filter** – weekly Close is above the 30-week SMA (Stage 2).

    Each resistance level can only trigger **one** breakout signal (the first
    bar that satisfies all conditions); subsequent closes above the same level
    are ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly OHLCV data.
    resistance_levels : list[dict]
        Output of :func:`get_resistance_levels`.
    volume_window : int
        Number of bars used for the rolling average volume baseline.
    volume_multiplier : float
        Required volume multiple above the rolling average (default 1.5×).

    Returns
    -------
    list[dict]
        Each breakout event dict contains:

        ``date``
            Date of the breakout bar (``pd.Timestamp``).
        ``close``
            Closing price on the breakout bar.
        ``resistance_level``
            The resistance price that was broken.
        ``resistance_date``
            Date when the resistance level was formed.
        ``volume_ratio``
            Ratio of breakout-bar volume to the rolling average volume,
            rounded to two decimal places.
        ``ma30``
            Value of the 30-week MA on the breakout bar, rounded to two
            decimal places.
    """
    avg_volume = df["Volume"].rolling(window=volume_window).mean()
    ma30 = compute_ma30(df["Close"])

    # Track which levels have already fired to avoid duplicate signals
    broken_levels: set[int] = set()

    breakouts: list[dict] = []

    for bar_pos in range(len(df)):
        close = float(df["Close"].iloc[bar_pos])
        volume = float(df["Volume"].iloc[bar_pos])
        avg_vol = avg_volume.iloc[bar_pos]
        ma30_val = ma30.iloc[bar_pos]

        # Skip bars where rolling stats are not yet available
        if pd.isna(avg_vol) or pd.isna(ma30_val):
            continue

        # MA30 filter — must be in Stage 2
        if close <= ma30_val:
            continue

        # Volume filter
        if volume <= volume_multiplier * avg_vol:
            continue

        # Check each resistance level that formed before this bar
        for lvl in resistance_levels:
            if lvl["index"] >= bar_pos:
                continue  # level formed at or after current bar — skip
            if lvl["index"] in broken_levels:
                continue  # already triggered

            if close > lvl["level"]:
                broken_levels.add(lvl["index"])
                breakouts.append(
                    {
                        "date": df.index[bar_pos],
                        "close": round(close, 4),
                        "resistance_level": round(lvl["level"], 4),
                        "resistance_date": lvl["date"],
                        "volume_ratio": round(volume / avg_vol, 2),
                        "ma30": round(ma30_val, 4),
                    }
                )

    breakouts.sort(key=lambda x: x["date"])
    return breakouts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_breakouts(
    ticker: str,
    period: str = "5y",
    lookback: int = 5,
    prominence: float = 0.5,
    volume_window: int = 10,
    volume_multiplier: float = 1.5,
) -> list[dict]:
    """Detect all confirmed breakouts for *ticker* on weekly bars.

    Orchestrates data fetching, swing-high detection, and breakout
    confirmation in one call.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"XLE"``).
    period : str
        Historical look-back passed to yfinance (default ``"5y"``).
    lookback : int
        Minimum bars between swing highs (default 5 weeks).
    prominence : float
        Minimum peak prominence for swing-high detection (default 0.5).
    volume_window : int
        Rolling window for average volume baseline (default 10 bars).
    volume_multiplier : float
        Required volume multiple for breakout confirmation (default 1.5×).

    Returns
    -------
    list[dict]
        Sorted list of breakout event dicts (see :func:`confirm_breakouts`).

    Examples
    --------
    >>> results = detect_breakouts("XLE", period="5y")
    >>> for event in results:
    ...     print(event)
    """
    df = fetch_weekly_data(ticker, period=period)
    resistance_levels = get_resistance_levels(df, lookback=lookback, prominence=prominence)
    return confirm_breakouts(
        df,
        resistance_levels,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
    )
