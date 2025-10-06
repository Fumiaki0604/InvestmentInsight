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
    st.warning("🔍 DEBUG: _load_service_account_info() が呼び出されました")

    try:
        secret_payload = st.secrets.get("gcp_service_account")
        st.info(f"🔍 DEBUG: st.secrets.get('gcp_service_account') の結果: {type(secret_payload).__name__} = {secret_payload is not None}")
    except Exception as e:
        st.error(f"🔍 DEBUG: st.secretsへのアクセスでエラー: {e}")
        secret_payload = None

    if secret_payload:
        st.info("🔍 DEBUG: st.secrets に gcp_service_account が見つかりました")
        if isinstance(secret_payload, str):
            st.info(f"🔍 DEBUG: secret_payload は文字列型 (長さ: {len(secret_payload)})")
            try:
                return json.loads(secret_payload)
            except json.JSONDecodeError:
                st.warning("`gcp_service_account` secrets entry is not valid JSON. Trying structured access.")
        try:
            st.info(f"🔍 DEBUG: dict()で変換を試みます")
            return dict(secret_payload)
        except Exception as e:  # noqa: BLE001
            st.warning(f"`gcp_service_account` secrets entry has unexpected format: {e}")

    # Try loading from Render Secret Files locations
    st.info("🔍 DEBUG: Secret Filesのチェックをスキップして環境変数へ")

    raw_value = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")

    # Debug: Display in Streamlit UI
    if raw_value:
        st.info(f"🔍 DEBUG: GOOGLE_SERVICE_ACCOUNT_KEY が見つかりました (長さ: {len(raw_value)}文字)")
    else:
        st.error("🔍 DEBUG: GOOGLE_SERVICE_ACCOUNT_KEY が環境変数に見つかりません")

    if not raw_value:
        path_hint = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY_PATH")
        if path_hint:
            file_path = Path(path_hint)
            if file_path.exists():
                raw_value = file_path.read_text(encoding="utf-8")
        if not raw_value:
            return None

    try:
        candidate_path = Path(raw_value)
        st.info(f"🔍 DEBUG: ファイルパスとしてチェック: {str(candidate_path)[:100]}...")

        if candidate_path.exists():
            st.info(f"🔍 DEBUG: ファイルが存在しました。ファイルから読み込みます")
            try:
                raw_value = candidate_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:  # noqa: F841
                st.error(f"🔍 DEBUG: ファイル読み込みエラー: {exc}")
                return None
        else:
            st.info(f"🔍 DEBUG: ファイルは存在しません。JSON文字列として扱います")

        result = json.loads(raw_value)
        st.success(f"🔍 DEBUG: JSONのパースに成功しました（キー数: {len(result)}）")
        return result

    except json.JSONDecodeError as exc:
        st.error(f"🔍 DEBUG: JSONパースエラー: {exc}")
        st.error(f"🔍 DEBUG: raw_valueの最初の100文字: {raw_value[:100]}")
        return None
    except Exception as exc:
        st.error(f"🔍 DEBUG: 予期しないエラー: {type(exc).__name__}: {exc}")
        return None


# Temporarily disabled cache for debugging
# @lru_cache(maxsize=1)
def get_credentials() -> Credentials | None:
    st.info("🔍 DEBUG: get_credentials() が呼び出されました")
    info = _load_service_account_info()

    if not info:
        st.error("🔍 DEBUG: _load_service_account_info() が None を返しました")
        return None

    st.success(f"🔍 DEBUG: 認証情報を取得しました（タイプ: {type(info).__name__}）")

    try:
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        st.success("🔍 DEBUG: 認証情報から Credentials を作成しました")
        return creds
    except Exception as e:  # pragma: no cover - configuration issue
        st.error(f"🔍 DEBUG: Credentials作成エラー: {e}")
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
