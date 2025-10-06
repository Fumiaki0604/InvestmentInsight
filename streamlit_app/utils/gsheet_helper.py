from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


SCOPES: Iterable[str] = ("https://www.googleapis.com/auth/spreadsheets.readonly",)


def _load_service_account_info() -> dict[str, Any] | None:
    secret_payload = st.secrets.get("gcp_service_account")
    if secret_payload:
        if isinstance(secret_payload, str):
            try:
                return json.loads(secret_payload)
            except json.JSONDecodeError:
                st.warning("`gcp_service_account` secrets entry is not valid JSON. Trying structured access.")
        try:
            return dict(secret_payload)
        except Exception:  # noqa: BLE001
            st.warning("`gcp_service_account` secrets entry has unexpected format.")

    # Try loading from Render Secret Files locations
    for secret_path in ["/etc/secrets/secrets.toml", "secrets.toml"]:
        if Path(secret_path).exists():
            try:
                import toml
                secrets_data = toml.load(secret_path)
                if "gcp_service_account" in secrets_data:
                    return dict(secrets_data["gcp_service_account"])
            except Exception:  # noqa: BLE001
                pass

    raw_value = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not raw_value:
        path_hint = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY_PATH")
        if path_hint:
            file_path = Path(path_hint)
            if file_path.exists():
                raw_value = file_path.read_text(encoding="utf-8")
        if not raw_value:
            return None

    candidate_path = Path(raw_value)
    if candidate_path.exists():
        try:
            raw_value = candidate_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:  # noqa: F841
            raise RuntimeError(
                "`GOOGLE_SERVICE_ACCOUNT_KEY` points to a binary file. Please provide the JSON service account key."
            ) from exc

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration issue
        raise RuntimeError(
            "`GOOGLE_SERVICE_ACCOUNT_KEY` must contain the JSON payload or a path to the JSON service account key."
        ) from exc


@lru_cache(maxsize=1)
def get_credentials() -> Credentials | None:
    info = _load_service_account_info()
    if not info:
        return None

    try:
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:  # pragma: no cover - configuration issue
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
        col_names = ["日付", "基準価額", "25日移動平均", "200日移動平均"]

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
        df["日付"] = pd.to_datetime(df["日付"], format="%Y/%m/%d", errors="coerce")
        for col in ("基準価額", "25日移動平均", "200日移動平均"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        return df
    except Exception:
        return None
