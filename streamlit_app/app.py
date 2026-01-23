
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from utils.chart_helper import (
    calculate_dmi,
    calculate_ichimoku,
    calculate_macd,
    calculate_rsi,
    calculate_volatility,
    create_price_chart,
    generate_technical_summary,
    get_trend_arrow,
)
from utils.correlation_helper import (
    calculate_correlation_matrix,
    create_correlation_bar_chart,
    create_correlation_heatmap,
    create_correlation_summary_table,
    get_correlation_data,
    get_fund_correlations,
)
from utils.gpt_analysis import chat_with_ai_analyst, generate_personalized_analysis
from utils.gsheet_helper import get_sheet_data
from utils.news_feed import load_atom_entries
from utils.slack_notifier import (
    SlackNotifier,
    check_status_changes,
    load_previous_status,
    save_fund_status,
)

import os

STORAGE_PATH = Path("previous_fund_status.json")

# Get SPREADSHEET_ID from secrets or environment variable
if hasattr(st, "secrets") and "GOOGLE_SPREADSHEET_ID" in st.secrets:
    SPREADSHEET_ID = st.secrets["GOOGLE_SPREADSHEET_ID"]
else:
    SPREADSHEET_ID = os.environ.get("GOOGLE_SPREADSHEET_ID", "1O3nYKIHCrDbjz1yBGrrAnq883Lgotfvvq035tC9wMVM")


def get_latest_valid_value(series: pd.Series, current_index: int) -> float | None:
    valid = series[:current_index].dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def get_delta_display(value: float, format_type: str = "price") -> tuple[str, str]:
    if value == 0:
        return ("0円", "off") if format_type == "price" else ("0", "off")
    if value > 0:
        # プラスは緑（normal: 上昇=緑）
        return (f"+{value:,.0f}円", "normal") if format_type == "price" else (f"+{value:.1f}", "normal")
    # マイナスは赤（normal: 下降=赤）
    return (f"{value:,.0f}円", "normal") if format_type == "price" else (f"{value:.1f}", "normal")


st.set_page_config(page_title="投資信託ナビゲーター", page_icon="📊", layout="wide")
st.title("投資信託ナビゲーター")
st.markdown("各投資信託の基準価額、移動平均線の推移をグラフで表示します。")

tab_detail, tab_list, tab_corr, tab_news = st.tabs(["詳細分析", "銘柄一覧", "相関分析", "ニュース"])


with tab_detail:
    # セッションステート初期化（メモリ効率化）
    if "chat_history_per_fund" not in st.session_state:
        st.session_state.chat_history_per_fund = {}
    # technical_data_per_fundは不要なので削除（キャッシュを利用）

    @st.cache_data(ttl=1800)  # 30分に短縮（メモリ削減）
    def load_available_sheets() -> List[str]:
        sheets = get_sheet_data(SPREADSHEET_ID, None)
        if sheets is None:
            st.error("シートの読み込みに失敗しました。Google Sheetsの認証情報を確認してください。")
            return []
        return sheets  # type: ignore[return-value]

    available_sheets = load_available_sheets()

    selected_sheets = st.multiselect(
        "表示する投資信託を選択してください",
        options=available_sheets,
        default=available_sheets[:1] if available_sheets else None,
    )

    # 終了日を1営業日前に設定（土日を考慮）
    today = datetime.datetime.now()
    if today.weekday() == 0:  # 月曜日
        default_end_date = today - datetime.timedelta(days=3)  # 金曜日
    elif today.weekday() == 6:  # 日曜日
        default_end_date = today - datetime.timedelta(days=2)  # 金曜日
    else:
        default_end_date = today - datetime.timedelta(days=1)  # 前日

    default_start_date = default_end_date - datetime.timedelta(days=365)

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("開始日", default_start_date)
    with col_end:
        end_date = st.date_input("終了日", default_end_date)

    if selected_sheets:
        for sheet_name in selected_sheets:
            @st.cache_data(ttl=1800)  # 30分に短縮（メモリ削減）
            def load_sheet_data(sheet: str) -> pd.DataFrame | None:
                return get_sheet_data(SPREADSHEET_ID, sheet)  # type: ignore[return-value]

            try:
                df = load_sheet_data(sheet_name)
                if df is None or df.empty:
                    st.warning(f"{sheet_name} のデータが取得できませんでした。")
                    continue

                df["日付"] = pd.to_datetime(df["日付"])
                start_dt = pd.Timestamp(start_date.strftime("%Y-%m-%d"))
                end_dt = pd.Timestamp(end_date.strftime("%Y-%m-%d"))
                mask = (df["日付"] >= start_dt) & (df["日付"] <= end_dt)
                df = df[mask]
                if df.empty:
                    st.warning(f"{sheet_name} の指定期間にデータがありません。")
                    continue

                chart_container = st.container()
                with chart_container:
                    indicators = st.multiselect(
                        "表示するテクニカル指標を選択",
                        ["移動平均線", "RSI", "MACD", "ボリンジャーバンド", "DMI", "一目均衡表"],
                        default=["移動平均線", "RSI", "MACD", "ボリンジャーバンド", "DMI", "一目均衡表"],
                    )
                    indicator_flags = {indicator: True for indicator in indicators}
                    fig = create_price_chart(df, indicator_flags)
                    st.plotly_chart(fig, use_container_width=True)

                with st.sidebar:
                    st.markdown("### テクニカル指標詳細")
                    current_price = df["基準価額"].iloc[-1]
                    prev_price = get_latest_valid_value(df["基準価額"], -1)
                    price_delta = current_price - prev_price if prev_price is not None else 0
                    delta_text, delta_color = get_delta_display(price_delta)
                    st.metric("基準価額", f"{current_price:,.0f}円", delta=delta_text, delta_color=delta_color)

                    if "移動平均線" in indicators:
                        st.markdown("#### 移動平均線")
                        ma25_series = df["基準価額"].rolling(window=25).mean()
                        ma25 = ma25_series.iloc[-1]
                        ma25_prev = get_latest_valid_value(ma25_series, -1)
                        delta_text, delta_color = (
                            get_delta_display(ma25 - ma25_prev) if ma25_prev is not None else ("データなし", "off")
                        )
                        st.metric(
                            f"25日移動平均 {get_trend_arrow(ma25, ma25_prev)}",
                            f"{ma25:,.0f}円",
                            delta=delta_text,
                            delta_color=delta_color,
                        )

                        ma200_series = df["基準価額"].rolling(window=200).mean()
                        ma200 = ma200_series.iloc[-1]
                        ma200_prev = get_latest_valid_value(ma200_series, -1)
                        delta_text, delta_color = (
                            get_delta_display(ma200 - ma200_prev) if ma200_prev is not None else ("データなし", "off")
                        )
                        st.metric(
                            f"200日移動平均 {get_trend_arrow(ma200, ma200_prev)}",
                            f"{ma200:,.0f}円",
                            delta=delta_text,
                            delta_color=delta_color,
                        )

                        volatility_series = calculate_volatility(df["基準価額"], window=20)
                        if len(volatility_series.dropna()) >= 2:
                            volatility = volatility_series.iloc[-1]
                            volatility_prev = volatility_series.iloc[-2]
                            delta_text, delta_color = get_delta_display(volatility - volatility_prev)
                            st.metric(
                                "ボラティリティ（20日）",
                                f"{volatility:,.0f}円",
                                delta=delta_text,
                                delta_color=delta_color,
                            )
                            avg_volatility = volatility_series.mean()
                            if volatility > avg_volatility * 1.5:
                                st.warning("ボラティリティが平均より50%以上高い状態です")
                            elif volatility < avg_volatility * 0.5:
                                st.info("ボラティリティが平均より50%以上低い状態です")
                    if "RSI" in indicators:
                        st.markdown("#### RSI")
                        rsi_series = calculate_rsi(df["基準価額"])
                        rsi_value = rsi_series.iloc[-1]
                        rsi_prev = rsi_series.iloc[-2]
                        delta_text, delta_color = get_delta_display(rsi_value - rsi_prev, format_type="value")
                        st.metric("RSI (14)", f"{rsi_value:.1f}", delta=delta_text, delta_color=delta_color)
                        if rsi_value >= 70:
                            st.warning("RSIが70を超えており、買われ過ぎの状態です")
                        elif rsi_value <= 30:
                            st.warning("RSIが30を下回っており、売られ過ぎの状態です")

                    if "MACD" in indicators:
                        st.markdown("#### MACD")
                        macd_line, signal_line = calculate_macd(df["基準価額"])
                        macd_value = macd_line.iloc[-1]
                        macd_prev = macd_line.iloc[-2]
                        signal_value = signal_line.iloc[-1]
                        signal_prev = signal_line.iloc[-2]
                        hist_value = macd_value - signal_value
                        hist_prev = macd_prev - signal_prev

                        delta_text, delta_color = get_delta_display(macd_value - macd_prev, format_type="value")
                        st.metric("MACD", f"{macd_value:.2f}", delta=delta_text, delta_color=delta_color)

                        delta_text, delta_color = get_delta_display(signal_value - signal_prev, format_type="value")
                        st.metric("シグナル", f"{signal_value:.2f}", delta=delta_text, delta_color=delta_color)

                        delta_text, delta_color = get_delta_display(hist_value - hist_prev, format_type="value")
                        st.metric("MACDヒストグラム", f"{hist_value:.2f}", delta=delta_text, delta_color=delta_color)

                        if hist_value > 0 and hist_prev < 0:
                            st.info("📈 MACDがシグナルを上向きにクロス（買いシグナル）")
                        elif hist_value < 0 and hist_prev > 0:
                            st.info("📉 MACDがシグナルを下向きにクロス（売りシグナル）")

                    if "DMI" in indicators:
                        st.markdown("#### DMI")
                        plus_di, minus_di, adx = calculate_dmi(df["基準価額"])
                        plus_value = plus_di.iloc[-1]
                        plus_prev = plus_di.iloc[-2]
                        minus_value = minus_di.iloc[-1]
                        minus_prev = minus_di.iloc[-2]
                        adx_value = adx.iloc[-1]
                        adx_prev = adx.iloc[-2]

                        delta_text, delta_color = get_delta_display(plus_value - plus_prev, format_type="value")
                        st.metric("+DI", f"{plus_value:.1f}", delta=delta_text, delta_color=delta_color)
                        delta_text, delta_color = get_delta_display(minus_value - minus_prev, format_type="value")
                        st.metric("-DI", f"{minus_value:.1f}", delta=delta_text, delta_color=delta_color)
                        delta_text, delta_color = get_delta_display(adx_value - adx_prev, format_type="value")
                        st.metric("ADX", f"{adx_value:.1f}", delta=delta_text, delta_color=delta_color)

                        if plus_value > minus_value and plus_prev <= minus_prev:
                            st.info("📈 +DIが-DIを下から上抜け（買いシグナル）")
                        elif plus_value < minus_value and plus_prev >= minus_prev:
                            st.info("📉 +DIが-DIを上から下抜け（売りシグナル）")

                    if "一目均衡表" in indicators:
                        st.markdown("#### 一目均衡表")
                        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df["基準価額"])
                        tenkan_value = tenkan.iloc[-1] if not tenkan.isna().iloc[-1] else 0
                        kijun_value = kijun.iloc[-1] if not kijun.isna().iloc[-1] else 0
                        senkou_a_value = senkou_a.iloc[-1] if not senkou_a.isna().iloc[-1] else 0
                        senkou_b_value = senkou_b.iloc[-1] if not senkou_b.isna().iloc[-1] else 0

                        if len(tenkan.dropna()) >= 2:
                            tenkan_prev = tenkan.dropna().iloc[-2]
                            delta_text, delta_color = get_delta_display(tenkan_value - tenkan_prev)
                        else:
                            delta_text, delta_color = ("データなし", "off")
                        st.metric("転換線", f"{tenkan_value:,.0f}円", delta=delta_text, delta_color=delta_color)

                        if len(kijun.dropna()) >= 2:
                            kijun_prev = kijun.dropna().iloc[-2]
                            delta_text, delta_color = get_delta_display(kijun_value - kijun_prev)
                        else:
                            delta_text, delta_color = ("データなし", "off")
                        st.metric("基準線", f"{kijun_value:,.0f}円", delta=delta_text, delta_color=delta_color)

                        st.metric("先行スパン1", f"{senkou_a_value:,.0f}円")
                        st.metric("先行スパン2", f"{senkou_b_value:,.0f}円")

                        # シグナル判定
                        if len(tenkan.dropna()) >= 2 and len(kijun.dropna()) >= 2:
                            if tenkan_value > kijun_value and tenkan_prev <= kijun_prev:
                                st.info("📈 転換線が基準線を上抜け（好転シグナル）")
                            elif tenkan_value < kijun_value and tenkan_prev >= kijun_prev:
                                st.info("📉 転換線が基準線を下抜け（逆転シグナル）")

                        # 雲との位置関係
                        cloud_top = max(senkou_a_value, senkou_b_value)
                        cloud_bottom = min(senkou_a_value, senkou_b_value)
                        if current_price > cloud_top:
                            st.success("☁️ 基準価額が雲の上方（強気）")
                        elif current_price < cloud_bottom:
                            st.warning("☁️ 基準価額が雲の下方（弱気）")
                        else:
                            st.info("☁️ 基準価額が雲の中（様子見）")

                try:
                    summary, detailed = generate_technical_summary(df)
                    if summary:
                        col_l, col_mid, col_r = st.columns([1, 3, 1])
                        with col_mid:
                            st.markdown("### テクニカル分析サマリー")
                            for point in summary:
                                st.markdown(point)
                            st.markdown("### テクニカル分析詳細解説")
                            for analysis in detailed:
                                st.markdown(analysis)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"テクニカル分析の生成中にエラーが発生しました: {exc}")

                # technical_dataを事前に準備
                ma25_value = df["基準価額"].rolling(window=25).mean().iloc[-1]
                ma200_value = df["基準価額"].rolling(window=200).mean().iloc[-1]
                current_price = df["基準価額"].iloc[-1]
                ma25_prev = df["基準価額"].rolling(window=25).mean().iloc[-2]
                ma200_prev = df["基準価額"].rolling(window=200).mean().iloc[-2]

                if ma25_value > ma200_value and ma25_prev <= ma200_prev:
                    ma_cross_status = "ゴールデンクロス"
                elif ma25_value < ma200_value and ma25_prev >= ma200_prev:
                    ma_cross_status = "デッドクロス"
                elif ma25_value > ma200_value:
                    ma_cross_status = "25日線が200日線の上方"
                else:
                    ma_cross_status = "25日線が200日線の下方"

                _, analysis_details = generate_technical_summary(df)
                sentiment = "強気" if analysis_details and "強気傾向" in analysis_details[-1] else "弱気"
                decision = "様子見"
                if analysis_details:
                    for line in analysis_details[-1].split("\n"):
                        if "**" in line:
                            decision = line.replace("*", "").strip()
                            break

                technical_data = {
                    "price_info": f"基準価額: {current_price:,.0f}円",
                    "rsi_info": f"RSI: {calculate_rsi(df['基準価額']).iloc[-1]:.1f}",
                    "macd_info": f"MACD: {calculate_macd(df['基準価額'])[0].iloc[-1]:.2f}",
                    "trend": sentiment,
                    "recommendation": decision,
                    "ma25_value": float(ma25_value),
                    "ma200_value": float(ma200_value),
                    "price_ma25_ratio": float(((current_price - ma25_value) / ma25_value) * 100),
                    "price_ma200_ratio": float(((current_price - ma200_value) / ma200_value) * 100),
                    "ma_cross_status": ma_cross_status,
                }

                st.markdown(
                    """
<style>
div.stButton > button {
    background-color: #0066cc;
    color: white;
    font-size: 18px;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    margin: 10px 0;
}
div.stButton > button:hover {
    background-color: #0052a3;
}
</style>
""",
                    unsafe_allow_html=True,
                )

                if st.button("🤖 AIによる詳細分析を表示", key=f"ai_analysis_{sheet_name}"):
                    with st.spinner("AI分析を生成中..."):
                        try:
                            ai_analysis = generate_personalized_analysis(technical_data)
                            if ai_analysis:
                                st.markdown("### ■AIによる詳細分析")
                                st.markdown(ai_analysis)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"AI分析の生成中にエラーが発生しました: {exc}")

                if technical_data:
                    st.markdown("### 💬 AIアナリストとチャット")
                    st.markdown(f"**{sheet_name}** のテクニカル分析について、AIアナリストと対話できます。")

                    # チャット履歴を最新5件に制限してメモリ削減
                    MAX_HISTORY = 5
                    history = st.session_state.chat_history_per_fund.setdefault(sheet_name, [])

                    # 履歴が制限を超えたら古いものを削除
                    if len(history) > MAX_HISTORY:
                        history = history[-MAX_HISTORY:]
                        st.session_state.chat_history_per_fund[sheet_name] = history

                    for message in history:
                        if message["role"] == "user":
                            st.markdown(f"**👤 あなた**: {message['content']}")
                        else:
                            st.markdown(f"**🤖 AIアナリスト**: {message['content']}")

                    st.session_state.setdefault("chat_input_value", "")
                    st.session_state.setdefault("processing_message", False)

                    user_input = st.text_input(
                        "AIアナリストに質問する（例：「RSIが70を超えていますが、どう判断すべきですか？」）",
                        key=f"chat_input_{sheet_name}",
                    )

                    if st.button("送信", key=f"chat_send_{sheet_name}"):
                        if user_input and user_input.strip():
                            with st.spinner("AI分析中..."):
                                history.append({"role": "user", "content": user_input})
                                try:
                                    response = chat_with_ai_analyst(technical_data, user_input, history[-MAX_HISTORY:])
                                    history.append({"role": "assistant", "content": response})

                                    # 履歴を制限
                                    if len(history) > MAX_HISTORY * 2:
                                        st.session_state.chat_history_per_fund[sheet_name] = history[-MAX_HISTORY * 2:]

                                    st.rerun()
                                except Exception as e:
                                    st.error(f"エラーが発生しました: {str(e)}")
                        else:
                            st.warning("質問を入力してください")
            except Exception as exc:  # noqa: BLE001
                st.error(f"データの読み込み中にエラーが発生しました: {exc}")
    else:
        st.info("表示する投資信託を選択してください。")
with tab_list:
    st.markdown("### 📋 全銘柄一覧")
    st.markdown("登録されている全銘柄の現在の状況を一覧で確認できます。")

    def get_cache_key() -> str:
        now = datetime.datetime.now()
        cache_date = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d") if now.hour < 8 else now.strftime("%Y-%m-%d")
        return f"fund_data_{cache_date}"

    @st.cache_data(ttl=86400)
    def get_fund_status(sheet_name: str, cache_key: str) -> str:
        try:
            df = get_sheet_data(SPREADSHEET_ID, sheet_name)
            if df is None or df.empty:
                return "データなし"
            _, detailed = generate_technical_summary(df)
            conclusion = detailed[-1] if detailed else ""
            if "買い推奨" in conclusion:
                return "買い推奨"
            if "売り推奨" in conclusion:
                return "売り推奨"
            return "様子見"
        except Exception:
            return "分析エラー"

    @st.cache_data(ttl=86400)
    def get_fund_summary(sheet_name: str, cache_key: str) -> Dict[str, str]:
        try:
            df = get_sheet_data(SPREADSHEET_ID, sheet_name)
            if df is None or df.empty:
                return {
                    "銘柄名": sheet_name,
                    "ステータス": "データなし",
                    "基準価額": "-",
                    "前日比": "-",
                    "25日移動平均": "-",
                    "200日移動平均": "-",
                }

            latest_price = df["基準価額"].iloc[-1]
            previous_price = df["基準価額"].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - previous_price
            ma25_series = df["基準価額"].rolling(window=25).mean()
            ma200_series = df["基準価額"].rolling(window=200).mean()
            status = get_fund_status(sheet_name, cache_key)

            return {
                "銘柄名": sheet_name,
                "ステータス": status,
                "基準価額": f"{latest_price:,.0f}円",
                "前日比": f"{price_change:+.0f}円" if price_change != 0 else "0円",
                "25日移動平均": f"{ma25_series.iloc[-1]:,.0f}円" if len(ma25_series.dropna()) else "-",
                "200日移動平均": f"{ma200_series.iloc[-1]:,.0f}円" if len(ma200_series.dropna()) else "-",
            }
        except Exception:
            return {
                "銘柄名": sheet_name,
                "ステータス": "エラー",
                "基準価額": "-",
                "前日比": "-",
                "25日移動平均": "-",
                "200日移動平均": "-",
            }

    @st.cache_data(ttl=86400)
    def get_all_fund_data(sheet_list: List[str], cache_key: str) -> List[Dict[str, str]]:
        return [get_fund_summary(sheet, cache_key) for sheet in sheet_list]

    if available_sheets:
        cache_key = get_cache_key()
        now = datetime.datetime.now()
        next_update = (
            now.replace(hour=8, minute=30, second=0, microsecond=0) + datetime.timedelta(days=1)
            if now.hour >= 8
            else now.replace(hour=8, minute=30, second=0, microsecond=0)
        )

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col2:
            st.caption(f"次回更新予定: {next_update.strftime('%m/%d %H:%M')}")
        with col3:
            notifier = SlackNotifier()
            st.caption("✅ Slack通知: 有効" if notifier.is_configured() else "❌ Slack通知: 無効")
        with col4:
            c4a, c4b = st.columns(2)
            with c4a:
                if st.button("通知テスト"):
                    st.session_state.test_slack = True
            with c4b:
                if st.button("変更シミュレート"):
                    st.session_state.simulate_change = True

        session_key = f"fund_data_loaded_{cache_key}"
        if session_key not in st.session_state:
            with st.spinner("データを読み込み中... (毎朝8時に自動更新)"):
                fund_data = get_all_fund_data(available_sheets, cache_key)
                notifier = SlackNotifier()
                if notifier.is_configured():
                    try:
                        previous_status = load_previous_status(str(STORAGE_PATH))
                        status_changes = check_status_changes(previous_status, fund_data)
                        if status_changes:
                            sent = notifier.send_multiple_notifications(status_changes)
                            if sent:
                                st.success(f"✅ {sent}件の投資推奨変更をSlackに通知しました")
                        save_fund_status(fund_data, str(STORAGE_PATH))
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Slack通知処理でエラーが発生しました: {exc}")
                st.session_state[session_key] = True
                st.success("データを読み込みました（翌朝8時まで高速表示されます）")
        else:
            fund_data = get_all_fund_data(available_sheets, cache_key)
            st.info("⚡ キャッシュからデータを表示中（高速表示）")

        if st.session_state.get("test_slack"):
            notifier = SlackNotifier()
            if notifier.is_configured():
                import random

                if fund_data:
                    num_test = min(2, len(fund_data))
                    test_funds = random.sample(fund_data, num_test)
                    test_changes = [
                        {
                            "fund_name": fund["銘柄名"],
                            "old_status": "様子見",
                            "new_status": "買い推奨",
                            "price": fund["基準価額"],
                            "price_change": fund["前日比"],
                        }
                        for fund in test_funds
                    ]
                else:
                    test_changes = [
                        {
                            "fund_name": "テスト投資信託",
                            "old_status": "様子見",
                            "new_status": "買い推奨",
                            "price": "10,000円",
                            "price_change": "+50円",
                        }
                    ]
                try:
                    with st.spinner("Slackにテスト通知を送信中..."):
                        sent = notifier.send_multiple_notifications(test_changes)
                    if sent:
                        st.success(f"✅ {sent}件のテスト通知をSlackに送信しました！")
                        with st.expander("送信内容の詳細"):
                            for change in test_changes:
                                st.write(f"**{change['fund_name']}**")
                                st.write(f"ステータス変更: {change['old_status']} → {change['new_status']}")
                                st.write(f"現在価格: {change['price']} ({change['price_change']})")
                                st.write("---")
                    else:
                        st.error("❌ Slack通知の送信に失敗しました")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Slack通知エラー: {exc}")
            else:
                st.error("❌ SLACK_WEBHOOK_URLが設定されていません")
            st.session_state.test_slack = False
        if st.session_state.get("simulate_change"):
            notifier = SlackNotifier()
            if notifier.is_configured():
                try:
                    import random

                    previous_status = load_previous_status(str(STORAGE_PATH))
                    modified = fund_data.copy()
                    indices = random.sample(range(len(modified)), min(2, len(modified))) if modified else []
                    for idx in indices:
                        fund = modified[idx]
                        if fund["ステータス"] != "買い推奨":
                            fund["ステータス"] = "買い推奨"
                    status_changes = check_status_changes(previous_status, modified)
                    if status_changes:
                        with st.spinner("変更を検出してSlack通知を送信中..."):
                            sent = notifier.send_multiple_notifications(status_changes)
                        if sent:
                            st.success(f"✅ {sent}件のステータス変更をSlackに通知しました！")
                            with st.expander("検出された変更"):
                                for change in status_changes:
                                    st.write(f"**{change['fund_name']}**")
                                    st.write(f"ステータス変更: {change['old_status']} → {change['new_status']}")
                                    st.write(f"現在価格: {change['price']} ({change['price_change']})")
                                    st.write("---")
                        else:
                            st.error("❌ Slack通知の送信に失敗しました")
                    else:
                        st.info("ℹ️ 変更可能なファンドが見つかりませんでした")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ シミュレーションエラー: {exc}")
            else:
                st.error("❌ SLACK_WEBHOOK_URLが設定されていません")
            st.session_state.simulate_change = False

        summary_df = pd.DataFrame(fund_data)

        def style_status(val: str) -> str:
            if val == "買い推奨":
                return "background-color: #d4edda; color: #155724"
            if val == "売り推奨":
                return "background-color: #f8d7da; color: #721c24"
            if val == "様子見":
                return "background-color: #fff3cd; color: #856404"
            return "background-color: #f8f9fa; color: #6c757d"

        st.markdown("#### 銘柄一覧表")
        styled_df = summary_df.style.map(style_status, subset=["ステータス"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        st.markdown("#### 📊 サマリー統計")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("買い推奨", len([f for f in fund_data if f["ステータス"] == "買い推奨"]))
        with col_b:
            st.metric("売り推奨", len([f for f in fund_data if f["ステータス"] == "売り推奨"]))
        with col_c:
            st.metric("様子見", len([f for f in fund_data if f["ステータス"] == "様子見"]))
        with col_d:
            st.metric("データなし/エラー", len([f for f in fund_data if f["ステータス"] in {"データなし", "エラー", "分析エラー"}]))

        st.markdown("#### 🔍 フィルター")
        status_filter = st.selectbox("ステータスでフィルター", ["すべて", "買い推奨", "売り推奨", "様子見", "データなし/エラー"])
        if status_filter != "すべて":
            if status_filter == "データなし/エラー":
                filtered_df = summary_df[summary_df["ステータス"].isin(["データなし", "エラー", "分析エラー"])]
            else:
                filtered_df = summary_df[summary_df["ステータス"] == status_filter]
            st.markdown(f"#### フィルター結果: {status_filter}")
            st.dataframe(filtered_df.style.map(style_status, subset=["ステータス"]), use_container_width=True, hide_index=True)
    else:
        st.error("シートデータの読み込みに失敗しました。")
with tab_corr:
    st.header("📈 相関分析")
    st.markdown("銘柄間の価格変動相関関係を分析します（直近1年間のデータを使用）")

    try:
        available_sheets_corr = available_sheets if available_sheets else []
    except Exception:
        available_sheets_corr = []

    st.caption(f"利用可能な銘柄数: {len(available_sheets_corr)}銘柄")

    if available_sheets_corr:
        def get_cached_correlation_data(sheet_list: List[str], period_days: int):
            return get_correlation_data(sheet_list, period_days)

        def get_cached_correlation_matrix(key):
            correlation_data, _ = key
            return calculate_correlation_matrix(correlation_data)

        settings_col, result_col = st.columns([1, 3])
        with settings_col:
            st.subheader("⚙️ 分析設定")
            selected_fund = st.selectbox(
                "詳細分析する銘柄を選択",
                options=["全体マトリックス表示"] + available_sheets_corr,
                help="特定の銘柄を選択すると、その銘柄と他の銘柄との相関関係を表示します",
            )
            period_options = {
                "1年間（252営業日）": 252,
                "6ヶ月間（126営業日）": 126,
                "3ヶ月間（63営業日）": 63,
            }
            selected_period = st.selectbox("分析期間", options=list(period_options.keys()), index=0)
            period_days = period_options[selected_period]
            st.info(f"分析期間: {selected_period}")
            st.caption("すべての相関係数を表示します")

        with result_col:
            st.subheader("📊 相関分析結果")
            with st.spinner(f"相関分析データを計算中...（{selected_period}）"):
                correlation_data = get_cached_correlation_data(available_sheets_corr, period_days)
                if correlation_data and len(correlation_data) >= 2:
                    st.info(f"✅ データを取得した銘柄数: {len(correlation_data)}銘柄")
                    preview = [f"{name}: {len(series)}日分" for name, series in list(correlation_data.items())[:3]]
                    st.caption(", ".join(preview) + ("..." if len(correlation_data) > 3 else ""))
                elif correlation_data and len(correlation_data) < 2:
                    st.error(f"❌ 相関分析には最低2銘柄必要です。現在: {len(correlation_data)}銘柄")
                    st.info("💡 エラー詳細は上記の「⚠️ データ取得エラーの詳細」を確認してください。")
                    correlation_data = {}
                else:
                    st.error("❌ 相関分析データが取得できませんでした。")
                    st.info("💡 エラー詳細は上記の「⚠️ データ取得エラーの詳細」を確認してください。")
                correlation_matrix = get_cached_correlation_matrix((correlation_data, period_days)) if correlation_data and len(correlation_data) >= 2 else None

            if correlation_matrix is not None and not correlation_matrix.empty:
                if selected_fund == "全体マトリックス表示":
                    st.markdown("#### 🔥 全銘柄相関マトリックス")
                    fig = create_correlation_heatmap(correlation_matrix)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 📊 相関統計サマリー")
                    c1, c2, c3, c4 = st.columns(4)
                    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)).stack()
                    with c1:
                        st.metric("平均相関", f"{upper_triangle.mean():.3f}")
                    with c2:
                        st.metric("最大相関", f"{upper_triangle.max():.3f}")
                    with c3:
                        st.metric("最小相関", f"{upper_triangle.min():.3f}")
                    with c4:
                        st.metric("強相関ペア数", len(upper_triangle[upper_triangle.abs() > 0.7]))

                    st.markdown("#### 🔗 相関の強いペア（上位10位）")
                    pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i + 1, len(correlation_matrix.columns)):
                            fund_a = correlation_matrix.columns[i]
                            fund_b = correlation_matrix.columns[j]
                            corr_val = correlation_matrix.iloc[i, j]
                            strength = (
                                "強い正の相関"
                                if corr_val > 0.7
                                else "中程度の正の相関"
                                if corr_val > 0.3
                                else "弱い相関"
                                if corr_val > -0.3
                                else "中程度の負の相関"
                                if corr_val > -0.7
                                else "強い負の相関"
                            )
                            pairs.append({"銘柄A": fund_a, "銘柄B": fund_b, "相関係数": f"{corr_val:.3f}", "相関の強さ": strength})
                    pairs_df = pd.DataFrame(pairs)
                    pairs_df["abs_corr"] = pairs_df["相関係数"].astype(float).abs()
                    st.dataframe(pairs_df.nlargest(10, "abs_corr").drop(columns=["abs_corr"]), use_container_width=True, hide_index=True)

                    st.markdown("#### 🔗 相関の弱いペア（下位10位）")
                    st.dataframe(pairs_df.nsmallest(10, "abs_corr").drop(columns=["abs_corr"]), use_container_width=True, hide_index=True)
                else:
                    st.markdown(f"#### 📊 {selected_fund}との相関分析")
                    fund_corr = get_fund_correlations(correlation_matrix, selected_fund)
                    if fund_corr is not None and not fund_corr.empty:
                        bar_fig = create_correlation_bar_chart(fund_corr, selected_fund)
                        if bar_fig:
                            st.plotly_chart(bar_fig, use_container_width=True)
                        summary_table = create_correlation_summary_table(fund_corr, selected_fund)
                        if summary_table is not None:
                            st.markdown("#### 📋 詳細分析結果")
                            st.dataframe(summary_table, use_container_width=True, hide_index=True)
                        st.markdown("#### 💡 投資判断への活用")
                        high_pos = fund_corr[fund_corr > 0.7]
                        if not high_pos.empty:
                            st.success(f"**強い正の相関銘柄**: {', '.join(high_pos.index[:3])}")
                            st.markdown(f"→ {selected_fund}と同方向に動くため、分散効果は限定的")
                        high_neg = fund_corr[fund_corr < -0.7]
                        if not high_neg.empty:
                            st.error(f"**強い負の相関銘柄**: {', '.join(high_neg.index[:3])}")
                            st.markdown(f"→ {selected_fund}と逆方向に動くため、ヘッジ効果が期待できる")
                        low_corr = fund_corr[fund_corr.abs() < 0.3]
                        if not low_corr.empty:
                            st.info(f"**分散投資候補**: {', '.join(low_corr.index[:3])}")
                            st.markdown(f"→ {selected_fund}との関連性が低く、分散効果が期待できる")
                    else:
                        st.error(f"{selected_fund}の相関データが見つかりません。")
            else:
                st.error("相関分析に十分なデータがありません。銘柄や期間を確認してください。")

        st.markdown("---")
        st.markdown(
            """
#### 📚 相関分析について
**相関係数の解釈：**
- **+0.7～+1.0**: 強い正の相関（同じ方向に動く）
- **+0.3～+0.7**: 中程度の正の相関
- **-0.3～+0.3**: 弱い相関（関連性が低い）
- **-0.7～-0.3**: 中程度の負の相関
- **-1.0～-0.7**: 強い負の相関（逆方向に動く）

**投資への活用：**
- 相関の高い銘柄同士は分散投資効果が低い
- 相関の低い銘柄を組み合わせることでリスク分散が可能
- 負の相関がある銘柄はヘッジ効果が期待できる
"""
        )
    else:
        st.error("シートデータの読み込みに失敗しました。")

with tab_news:
    st.header("Finance News")
    st.caption("Source: RSS/Atom feed")
    if "news_refresh_key" not in st.session_state:
        st.session_state.news_refresh_key = "init"

    feed_url = st.text_input("Feed URL", value="https://www.dir.co.jp/feed/economics.atom")
    max_items = st.slider("Items", min_value=5, max_value=50, value=20, step=5)
    if st.button("Refresh news"):
        st.session_state.news_refresh_key = datetime.datetime.utcnow().isoformat()

    if feed_url.strip():
        @st.cache_data(ttl=1800)
        def load_news(url: str, limit: int, refresh_key: str):
            return load_atom_entries(url, limit)

        try:
            with st.spinner("Loading news..."):
                entries = load_news(feed_url, max_items, st.session_state.news_refresh_key)
            if not entries:
                st.info("No entries found.")
            for item in entries:
                title = item.get("title") or "(no title)"
                link = item.get("link") or ""
                updated = item.get("updated") or ""
                summary = item.get("summary") or ""
                if link:
                    st.markdown(f"- [{title}]({link})")
                else:
                    st.markdown(f"- {title}")
                if updated:
                    st.caption(updated)
                if summary:
                    with st.expander("Summary"):
                        st.write(summary)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load news: {exc}")
    else:
        st.info("Enter a feed URL to load news.")

st.markdown("---")
st.markdown("データ出典: Google Spreadsheet")
