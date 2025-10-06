"""Utilities for loading mutual fund datasets from various sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from google.oauth2.service_account import Credentials
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

LOGGER = logging.getLogger(__name__)


COLUMN_NAME_MAP = {
    "基準価額": "nav",
    "前日差": "change_prev_day",
    "移動平均線（25日）": "sma_25",
    "基準価額-移動平均線": "nav_minus_sma",
    "トレンド": "trend",
    "トレンドステータス": "trend_status",
    "上下幅％表示": "range_percent",
    "移動平均線（200日）": "sma_200",
}

PERCENT_SUFFIX = "%"


class DataLoader:
    """Load fund datasets from local CSV files or Google Sheets."""

    def __init__(self, local_path: Path | None = None) -> None:
        settings = get_settings()
        self._local_path = Path(local_path or settings.default_local_data_path)
        self._spreadsheet_id = settings.google_spreadsheet_id
        self._service_account_payload = settings.google_service_account_key

    def list_available_tickers(self) -> list[str]:
        """Return tickers discoverable from the default data source."""
        df = self._load_local_dataframe()
        tickers = sorted(df["ticker_id"].unique())
        return [str(ticker) for ticker in tickers if pd.notna(ticker)]

    def load_ticker_data(
        self,
        ticker_id: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load a single ticker dataset applying optional date filters."""
        df = self._load_local_dataframe()
        df = df[df["ticker_id"] == ticker_id]

        if df.empty:
            raise ValueError(f"Ticker '{ticker_id}' not found in local dataset")

        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]

        return df.sort_values("date").reset_index(drop=True)

    def _load_local_dataframe(self) -> pd.DataFrame:
        if not self._local_path.exists():
            raise FileNotFoundError(f"Local data file not found: {self._local_path}")

        df = pd.read_csv(self._local_path)
        df = self._normalize_dataframe(df)
        return df

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = list(df.columns)
        ticker_id: str | None = None
        if columns:
            maybe_ticker = columns[0]
            if maybe_ticker.isdigit():
                ticker_id = maybe_ticker
                df = df.rename(columns={maybe_ticker: "date_key"})

        rename_map = {col: COLUMN_NAME_MAP.get(col, col) for col in df.columns}
        df = df.rename(columns=rename_map)

        if "date" not in df.columns and "date_key" in df.columns:
            df["date"] = pd.to_datetime(df["date_key"], format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if ticker_id is not None:
            df["ticker_id"] = ticker_id
        elif "ticker_id" in df.columns:
            df["ticker_id"] = df["ticker_id"].astype(str)
        else:
            df["ticker_id"] = "unknown"

        df["date"] = df["date"].dt.tz_localize(None)

        object_columns = [col for col in df.columns if df[col].dtype == object]
        for col in object_columns:
            series = df[col].astype(str)
            if series.str.endswith(PERCENT_SUFFIX).all():
                df[col] = (
                    series.str.replace(PERCENT_SUFFIX, "", regex=False)
                    .replace({"": None})
                    .astype(float)
                )

        numeric_columns: Iterable[str] = [
            "nav",
            "change_prev_day",
            "sma_25",
            "nav_minus_sma",
            "sma_200",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def load_from_google_sheets(self, sheet_name: str) -> pd.DataFrame:
        """Fetch a worksheet and return a normalized dataframe."""
        if not self._spreadsheet_id:
            raise RuntimeError("GOOGLE_SPREADSHEET_ID is not configured")
        if not self._service_account_payload:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_KEY is not configured")

        values = self._fetch_sheet_values(sheet_name)
        if not values:
            raise ValueError(f"No data returned from sheet '{sheet_name}'")
        header, *rows = values
        if not rows:
            raise ValueError(f"Sheet '{sheet_name}' does not contain data rows")

        df = pd.DataFrame(rows, columns=header)
        return self._normalize_dataframe(df)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _fetch_sheet_values(self, sheet_name: str) -> list[list[str]]:
        credentials = Credentials.from_service_account_info(json.loads(self._service_account_payload))
        import gspread

        client = gspread.authorize(credentials)
        worksheet = client.open_by_key(self._spreadsheet_id).worksheet(sheet_name)
        return worksheet.get_all_values()
