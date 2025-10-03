from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .config import get_settings
from .services import AnalyticsService

app = FastAPI(title="Mutual Funds Insight", version="0.1.0")
settings = get_settings()
service = AnalyticsService()


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/settings")
def read_settings() -> dict[str, Optional[str]]:
    return {
        "default_local_data_path": str(settings.default_local_data_path),
        "google_spreadsheet_id": settings.google_spreadsheet_id,
    }


@app.get("/tickers")
def list_tickers() -> dict[str, list[str]]:
    tickers = service.list_tickers()
    return {"tickers": tickers}


@app.get("/tickers/{ticker_id}")
def get_ticker(
    ticker_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
):
    try:
        payload = service.get_ticker_payload(ticker_id, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return payload.model_dump(by_alias=True)
