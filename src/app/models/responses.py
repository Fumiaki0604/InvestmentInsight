from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PricePoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: datetime
    nav: Optional[float] = None
    change_prev_day: Optional[float] = None
    sma_25: Optional[float] = None
    sma_200: Optional[float] = None
    nav_minus_sma: Optional[float] = None
    range_percent: Optional[float] = None
    rsi_14: Optional[float] = Field(default=None, alias="rsi_14")
    macd: Optional[float] = Field(default=None, alias="macd")
    macd_signal: Optional[float] = Field(default=None, alias="macd_signal")
    macd_hist: Optional[float] = Field(default=None, alias="macd_hist")
    macd_12_26_9: Optional[float] = Field(default=None, alias="MACD_12_26_9")
    macds_12_26_9: Optional[float] = Field(default=None, alias="MACDs_12_26_9")
    macdh_12_26_9: Optional[float] = Field(default=None, alias="MACDh_12_26_9")
    bb_upper_20: Optional[float] = Field(default=None, alias="bb_upper_20")
    bb_lower_20: Optional[float] = Field(default=None, alias="bb_lower_20")
    bb_sma_20: Optional[float] = Field(default=None, alias="bb_sma_20")


class TickerMetadata(BaseModel):
    ticker_id: str
    records: int
    start_date: Optional[datetime]
    end_date: Optional[datetime]


class IndicatorMetadata(BaseModel):
    sma: Dict[str, Any]
    macd: Dict[str, Any]
    rsi: Dict[str, Any]
    bollinger_bands: Dict[str, Any]


class TickerPayload(BaseModel):
    ticker: TickerMetadata
    indicators: IndicatorMetadata
    points: List[PricePoint]


__all__ = [
    "PricePoint",
    "TickerMetadata",
    "IndicatorMetadata",
    "TickerPayload",
]
