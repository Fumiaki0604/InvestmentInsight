from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


@dataclass
class IndicatorResult:
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_sma(df: pd.DataFrame, period: int, price_column: str = "nav") -> IndicatorResult:
    data = df.copy()
    column_name = f"sma_{period}"
    data[column_name] = data[price_column].rolling(window=period, min_periods=period).mean()
    return IndicatorResult(dataframe=data, metadata={"period": period})


def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price_column: str = "nav",
) -> IndicatorResult:
    data = df.copy()
    sma_name = f"bb_sma_{period}"
    upper_name = f"bb_upper_{period}"
    lower_name = f"bb_lower_{period}"

    rolling = data[price_column].rolling(window=period, min_periods=period)
    data[sma_name] = rolling.mean()
    rolling_std = rolling.std()
    data[upper_name] = data[sma_name] + std_dev * rolling_std
    data[lower_name] = data[sma_name] - std_dev * rolling_std

    metadata = {"period": period, "std_dev": std_dev}
    return IndicatorResult(dataframe=data, metadata=metadata)


def compute_macd(
    df: pd.DataFrame,
    price_column: str = "nav",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> IndicatorResult:
    data = df.copy()
    price = data[price_column]

    try:
        import pandas_ta as ta

        macd_df = ta.macd(price, fast=fast_period, slow=slow_period, signal=signal_period)
        data = pd.concat([data, macd_df], axis=1)
        macd_col = macd_df.columns[0]
        signal_col = macd_df.columns[1]
        hist_col = macd_df.columns[2]
    except Exception:  # pragma: no cover - manual fallback when pandas_ta unavailable
        macd_col = "macd"
        signal_col = "macd_signal"
        hist_col = "macd_hist"

        ema_fast = price.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price.ewm(span=slow_period, adjust=False).mean()
        data[macd_col] = ema_fast - ema_slow
        data[signal_col] = data[macd_col].ewm(span=signal_period, adjust=False).mean()
        data[hist_col] = data[macd_col] - data[signal_col]

    metadata = {
        "fast_period": fast_period,
        "slow_period": slow_period,
        "signal_period": signal_period,
        "macd_column": macd_col,
        "signal_column": signal_col,
        "histogram_column": hist_col,
    }
    return IndicatorResult(dataframe=data, metadata=metadata)


def compute_rsi(
    df: pd.DataFrame,
    period: int = 14,
    price_column: str = "nav",
) -> IndicatorResult:
    data = df.copy()
    price = data[price_column]
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data[f"rsi_{period}"] = rsi

    metadata = {"period": period, "column": f"rsi_{period}"}
    return IndicatorResult(dataframe=data, metadata=metadata)


def compute_dmi(
    df: pd.DataFrame,
    period: int = 14,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "nav",
) -> IndicatorResult:
    required_columns = {high_column, low_column, close_column}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"Missing columns for DMI calculation: {missing}")

    data = df.copy()
    high = data[high_column]
    low = data[low_column]
    close = data[close_column]

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr = true_range.rolling(window=period, min_periods=period).mean()
    data["plus_di"] = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    data["minus_di"] = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)
    data["dx"] = (
        (data["plus_di"] - data["minus_di"]).abs()
        / (data["plus_di"] + data["minus_di"])
        * 100
    )
    data["adx"] = data["dx"].rolling(window=period, min_periods=period).mean()

    metadata = {
        "period": period,
        "plus_di": "plus_di",
        "minus_di": "minus_di",
        "adx": "adx",
    }
    return IndicatorResult(dataframe=data, metadata=metadata)


def compute_all_indicators(df: pd.DataFrame) -> IndicatorResult:
    data = df.copy()

    sma_result = compute_sma(data, period=25)
    data = sma_result.dataframe

    macd_result = compute_macd(data)
    data = macd_result.dataframe

    rsi_result = compute_rsi(data)
    data = rsi_result.dataframe

    bb_result = compute_bollinger_bands(data)
    data = bb_result.dataframe

    metadata: Dict[str, Any] = {
        "sma": sma_result.metadata,
        "macd": macd_result.metadata,
        "rsi": rsi_result.metadata,
        "bollinger_bands": bb_result.metadata,
    }

    return IndicatorResult(dataframe=data, metadata=metadata)
