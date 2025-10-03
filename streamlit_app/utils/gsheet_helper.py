from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


SCOPES: Iterable[str] = ("https://www.googleapis.com/auth/spreadsheets.readonly",)


def _load_service_account_info() -> dict[str, Any] | None:
    if "gcp_service_account" in st.secrets:
        return dict(st.secrets["gcp_service_account"])  # type: ignore[arg-type]

    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not key:
        return None

    import json

    try:
        return json.loads(key)
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=1)
def get_credentials() -> Credentials | None:
    info = _load_service_account_info()
    if not info:
        return None

    try:
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        return None


def get_sheet_data(spreadsheet_id: str, sheet_name: str | None = None) -> list[str] | pd.DataFrame | None:
    try:
        credentials = get_credentials()
        if not credentials:
            return None

        service = build("sheets", "v4", credentials=credentials)
        sheets = service.spreadsheets()

        if sheet_name is None:
            result = sheets.get(spreadsheetId=spreadsheet_id).execute()
            sheet_props = result.get("sheets", [])
            return [sheet["properties"]["title"] for sheet in sheet_props]

        range_name = f"{sheet_name}!B:J"
        result = sheets.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get("values", [])
        if not values:
            return None

        col_indices = [0, 1, 3, 8]
        col_names = ["“ú•t", "Šî€‰¿Šz", "25“úˆÚ“®•½‹Ï", "200“úˆÚ“®•½‹Ï"]

        processed_rows: list[list[Any | None]] = []
        for row in values[1:]:
            if len(row) <= 1:
                continue
            processed_row: list[Any | None] = []
            for idx in col_indices:
                value = row[idx] if idx < len(row) else None
                if isinstance(value, str) and value.strip() == "":
                    value = None
                processed_row.append(value)

            if processed_row[0]:
                processed_rows.append(processed_row)

        if not processed_rows:
            return None

        df = pd.DataFrame(processed_rows, columns=col_names)
        df["“ú•t"] = pd.to_datetime(df["“ú•t"], format="%Y/%m/%d", errors="coerce")
        for col in ("Šî€‰¿Šz", "25“úˆÚ“®•½‹Ï", "200“úˆÚ“®•½‹Ï"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        return df
    except Exception:
        return None
