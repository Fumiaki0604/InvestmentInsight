from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..data_sources import DataLoader
from ..indicators import IndicatorResult, compute_all_indicators
from ..models import IndicatorMetadata, PricePoint, TickerMetadata, TickerPayload


class AnalyticsService:
    def __init__(self, loader: Optional[DataLoader] = None) -> None:
        self._loader = loader or DataLoader()

    def list_tickers(self) -> list[str]:
        return self._loader.list_available_tickers()

    def get_ticker_payload(
        self,
        ticker_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> TickerPayload:
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None

        base_df = self._loader.load_ticker_data(ticker_id, start_ts, end_ts)
        indicator_result: IndicatorResult = compute_all_indicators(base_df)
        enriched_df = indicator_result.dataframe
        indicator_metadata = indicator_result.metadata

        safe_df = enriched_df.where(pd.notna(enriched_df), None)
        points = [PricePoint.model_validate(row) for row in safe_df.to_dict(orient="records")]

        ticker_meta = TickerMetadata(
            ticker_id=ticker_id,
            records=len(enriched_df),
            start_date=enriched_df["date"].min().to_pydatetime() if not enriched_df.empty else None,
            end_date=enriched_df["date"].max().to_pydatetime() if not enriched_df.empty else None,
        )

        indicators_meta = IndicatorMetadata(**indicator_metadata)

        return TickerPayload(ticker=ticker_meta, indicators=indicators_meta, points=points)
