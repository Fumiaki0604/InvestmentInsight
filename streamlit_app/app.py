
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from utils.chart_helper import (
    calculate_dmi,
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
from utils.slack_notifier import (
    SlackNotifier,
    check_status_changes,
    load_previous_status,
    save_fund_status,
)

STORAGE_PATH = Path("previous_fund_status.json")
SPREADSHEET_ID = "1O3nYKIHCrDbjz1yBGrrAnq883Lgotfvvq035tC9wMVM"


def get_latest_valid_value(series: pd.Series, current_index: int) -> float | None:
    valid = series[:current_index].dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def get_delta_display(value: float, format_type: str = "price") -> tuple[str, str]:
    if value == 0:
        return ("0�~", "off") if format_type == "price" else ("0", "off")
    if value > 0:
        return (f"{value:,.0f}�~ ��", "normal") if format_type == "price" else (f"{value:.1f} ��", "normal")
    abs_val = abs(value)
    return (f"{abs_val:,.0f}�~ ��", "inverse") if format_type == "price" else (f"{abs_val:.1f} ��", "inverse")


st.set_page_config(page_title="�����M���i�r�Q�[�^�[", page_icon="??", layout="wide")
st.title("�����M���i�r�Q�[�^�[")
st.markdown("�e�����M���̊���z�A�ړ����ϐ��̐��ڂ��O���t�ŕ\�����܂��B")

tab_detail, tab_list, tab_corr = st.tabs(["�ڍו���", "�����ꗗ", "���֕���"])


with tab_detail:
    if "selected_sheets" not in st.session_state:
        st.session_state.selected_sheets = []
    if "chat_history_per_fund" not in st.session_state:
        st.session_state.chat_history_per_fund = {}
    if "technical_data_per_fund" not in st.session_state:
        st.session_state.technical_data_per_fund = {}
    if "technical_data" not in st.session_state:
        st.session_state.technical_data = None

    @st.cache_data(ttl=3600)
    def load_available_sheets() -> List[str]:
        sheets = get_sheet_data(SPREADSHEET_ID, None)
        if sheets is None:
            st.error("�V�[�g�̓ǂݍ��݂Ɏ��s���܂����BGoogle Sheets�̔F�؏����m�F���Ă��������B")
            return []
        return sheets  # type: ignore[return-value]

    available_sheets = load_available_sheets()

    selected_sheets = st.multiselect(
        "�\�����铊���M����I�����Ă�������",
        options=available_sheets,
        default=available_sheets[:1] if available_sheets else None,
    )

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("�J�n��", datetime.datetime.now() - datetime.timedelta(days=365))
    with col_end:
        end_date = st.date_input("�I����", datetime.datetime.now())

    if selected_sheets:
        for sheet_name in selected_sheets:
            @st.cache_data(ttl=3600)
            def load_sheet_data(sheet: str) -> pd.DataFrame | None:
                return get_sheet_data(SPREADSHEET_ID, sheet)  # type: ignore[return-value]

            try:
                df = load_sheet_data(sheet_name)
                if df is None or df.empty:
                    st.warning(f"{sheet_name} �̃f�[�^���擾�ł��܂���ł����B")
                    continue

                df["���t"] = pd.to_datetime(df["���t"])
                start_dt = pd.Timestamp(start_date.strftime("%Y-%m-%d"))
                end_dt = pd.Timestamp(end_date.strftime("%Y-%m-%d"))
                mask = (df["���t"] >= start_dt) & (df["���t"] <= end_dt)
                df = df[mask]
                if df.empty:
                    st.warning(f"{sheet_name} �̎w����ԂɃf�[�^������܂���B")
                    continue

                chart_container = st.container()
                with chart_container:
                    indicators = st.multiselect(
                        "�\������e�N�j�J���w�W��I��",
                        ["�ړ����ϐ�", "RSI", "MACD", "�{�����W���[�o���h", "DMI"],
                        default=["�ړ����ϐ�", "RSI", "MACD", "�{�����W���[�o���h", "DMI"],
                    )
                    indicator_flags = {indicator: True for indicator in indicators}
                    fig = create_price_chart(df, indicator_flags)
                    st.plotly_chart(fig, use_container_width=True)

                with st.sidebar:
                    st.markdown("### �e�N�j�J���w�W�ڍ�")
                    current_price = df["����z"].iloc[-1]
                    prev_price = get_latest_valid_value(df["����z"], -1)
                    price_delta = current_price - prev_price if prev_price is not None else 0
                    delta_text, delta_color = get_delta_display(price_delta)
                    st.metric("����z", f"{current_price:,.0f}�~", delta=delta_text, delta_color=delta_color)

                    if "�ړ����ϐ�" in indicators:
                        st.markdown("#### �ړ����ϐ�")
                        ma25_series = df["����z"].rolling(window=25).mean()
                        ma25 = ma25_series.iloc[-1]
                        ma25_prev = get_latest_valid_value(ma25_series, -1)
                        delta_text, delta_color = (
                            get_delta_display(ma25 - ma25_prev) if ma25_prev is not None else ("�f�[�^�Ȃ�", "off")
                        )
                        st.metric(
                            f"25���ړ����� {get_trend_arrow(ma25, ma25_prev)}",
                            f"{ma25:,.0f}�~",
                            delta=delta_text,
                            delta_color=delta_color,
                        )

                        ma200_series = df["����z"].rolling(window=200).mean()
                        ma200 = ma200_series.iloc[-1]
                        ma200_prev = get_latest_valid_value(ma200_series, -1)
                        delta_text, delta_color = (
                            get_delta_display(ma200 - ma200_prev) if ma200_prev is not None else ("�f�[�^�Ȃ�", "off")
                        )
                        st.metric(
                            f"200���ړ����� {get_trend_arrow(ma200, ma200_prev)}",
                            f"{ma200:,.0f}�~",
                            delta=delta_text,
                            delta_color=delta_color,
                        )

                        volatility_series = calculate_volatility(df["����z"], window=20)
                        if len(volatility_series.dropna()) >= 2:
                            volatility = volatility_series.iloc[-1]
                            volatility_prev = volatility_series.iloc[-2]
                            delta_text, delta_color = get_delta_display(volatility - volatility_prev)
                            st.metric(
                                "�{���e�B���e�B�i20���j",
                                f"{volatility:,.0f}�~",
                                delta=delta_text,
                                delta_color=delta_color,
                            )
                            avg_volatility = volatility_series.mean()
                            if volatility > avg_volatility * 1.5:
                                st.warning("�{���e�B���e�B�����ς��50%�ȏ㍂����Ԃł�")
                            elif volatility < avg_volatility * 0.5:
                                st.info("�{���e�B���e�B�����ς��50%�ȏ�Ⴂ��Ԃł�")
                    if "RSI" in indicators:
                        st.markdown("#### RSI")
                        rsi_series = calculate_rsi(df["����z"])
                        rsi_value = rsi_series.iloc[-1]
                        rsi_prev = rsi_series.iloc[-2]
                        delta_text, delta_color = get_delta_display(rsi_value - rsi_prev, format_type="value")
                        st.metric("RSI (14)", f"{rsi_value:.1f}", delta=delta_text, delta_color=delta_color)
                        if rsi_value >= 70:
                            st.warning("RSI��70�𒴂��Ă���A�����߂��̏�Ԃł�")
                        elif rsi_value <= 30:
                            st.warning("RSI��30��������Ă���A�����߂��̏�Ԃł�")

                    if "MACD" in indicators:
                        st.markdown("#### MACD")
                        macd_line, signal_line = calculate_macd(df["����z"])
                        macd_value = macd_line.iloc[-1]
                        macd_prev = macd_line.iloc[-2]
                        signal_value = signal_line.iloc[-1]
                        signal_prev = signal_line.iloc[-2]
                        hist_value = macd_value - signal_value
                        hist_prev = macd_prev - signal_prev

                        delta_text, delta_color = get_delta_display(macd_value - macd_prev, format_type="value")
                        st.metric("MACD", f"{macd_value:.2f}", delta=delta_text, delta_color=delta_color)

                        delta_text, delta_color = get_delta_display(signal_value - signal_prev, format_type="value")
                        st.metric("�V�O�i��", f"{signal_value:.2f}", delta=delta_text, delta_color=delta_color)

                        delta_text, delta_color = get_delta_display(hist_value - hist_prev, format_type="value")
                        st.metric("MACD�q�X�g�O����", f"{hist_value:.2f}", delta=delta_text, delta_color=delta_color)

                        if hist_value > 0 and hist_prev < 0:
                            st.info("?? MACD���V�O�i����������ɃN���X�i�����V�O�i���j")
                        elif hist_value < 0 and hist_prev > 0:
                            st.info("?? MACD���V�O�i�����������ɃN���X�i����V�O�i���j")

                    if "DMI" in indicators:
                        st.markdown("#### DMI")
                        plus_di, minus_di, adx = calculate_dmi(df["����z"])
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
                            st.info("?? +DI��-DI��������㔲���i�����V�O�i���j")
                        elif plus_value < minus_value and plus_prev >= minus_prev:
                            st.info("?? +DI��-DI���ォ�牺�����i����V�O�i���j")

                try:
                    summary, detailed = generate_technical_summary(df)
                    if summary:
                        col_l, col_mid, col_r = st.columns([1, 3, 1])
                        with col_mid:
                            st.markdown("### �e�N�j�J�����̓T�}���[")
                            for point in summary:
                                st.markdown(point)
                            st.markdown("### �e�N�j�J�����͏ڍ׉��")
                            for analysis in detailed:
                                st.markdown(analysis)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"�e�N�j�J�����͂̐������ɃG���[���������܂���: {exc}")

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

                if st.button("?? AI�ɂ��ڍו��͂�\��", key=f"ai_analysis_{sheet_name}"):
                    with st.spinner("AI���͂𐶐���..."):
                        try:
                            _, analysis_details = generate_technical_summary(df)
                            sentiment = "���C" if analysis_details and "���C�X��" in analysis_details[-1] else "��C"
                            decision = "�l�q��"
                            if analysis_details:
                                for line in analysis_details[-1].split("\n"):
                                    if "**" in line:
                                        decision = line.replace("*", "").strip()
                                        break

                            ma25_value = df["����z"].rolling(window=25).mean().iloc[-1]
                            ma200_value = df["����z"].rolling(window=200).mean().iloc[-1]
                            current_price = df["����z"].iloc[-1]
                            ma25_prev = df["����z"].rolling(window=25).mean().iloc[-2]
                            ma200_prev = df["����z"].rolling(window=200).mean().iloc[-2]

                            if ma25_value > ma200_value and ma25_prev <= ma200_prev:
                                ma_cross_status = "�S�[���f���N���X"
                            elif ma25_value < ma200_value and ma25_prev >= ma200_prev:
                                ma_cross_status = "�f�b�h�N���X"
                            elif ma25_value > ma200_value:
                                ma_cross_status = "25������200�����̏��"
                            else:
                                ma_cross_status = "25������200�����̉���"

                            technical_data = {
                                "price_info": f"����z: {current_price:,.0f}�~",
                                "rsi_info": f"RSI: {calculate_rsi(df['����z']).iloc[-1]:.1f}",
                                "macd_info": f"MACD: {calculate_macd(df['����z'])[0].iloc[-1]:.2f}",
                                "trend": sentiment,
                                "recommendation": decision,
                                "ma25_value": float(ma25_value),
                                "ma200_value": float(ma200_value),
                                "price_ma25_ratio": float(((current_price - ma25_value) / ma25_value) * 100),
                                "price_ma200_ratio": float(((current_price - ma200_value) / ma200_value) * 100),
                                "ma_cross_status": ma_cross_status,
                            }
                            st.session_state.technical_data = technical_data
                            st.session_state.technical_data_per_fund[sheet_name] = technical_data

                            ai_analysis = generate_personalized_analysis(technical_data)
                            if ai_analysis:
                                st.markdown("### ��AI�ɂ��ڍו���")
                                st.markdown(ai_analysis)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"AI���͂̐������ɃG���[���������܂���: {exc}")

                if st.session_state.technical_data:
                    st.markdown("### ?? AI�A�i���X�g�ƃ`���b�g")
                    st.markdown(f"**{sheet_name}** �̃e�N�j�J�����͂ɂ��āAAI�A�i���X�g�ƑΘb�ł��܂��B")

                    history = st.session_state.chat_history_per_fund.setdefault(sheet_name, [])
                    for message in history:
                        if message["role"] == "user":
                            st.markdown(f"**?? ���Ȃ�**: {message['content']}")
                        else:
                            st.markdown(f"**?? AI�A�i���X�g**: {message['content']}")

                    st.session_state.setdefault("chat_input_value", "")
                    st.session_state.setdefault("processing_message", False)

                    def submit_message() -> None:
                        if st.session_state.processing_message:
                            return
                        message = st.session_state.chat_input_value.strip()
                        if not message:
                            return

                        st.session_state.processing_message = True
                        history.append({"role": "user", "content": message})
                        current_data = st.session_state.technical_data_per_fund.get(sheet_name, st.session_state.technical_data)
                        response = chat_with_ai_analyst(current_data, message, history)
                        history.append({"role": "assistant", "content": response})
                        st.session_state.chat_input_value = ""
                        st.session_state.processing_message = False

                    st.text_input(
                        "AI�A�i���X�g�Ɏ��₷��i��F�uRSI��70�𒴂��Ă��܂����A�ǂ����f���ׂ��ł����H�v�j",
                        key="chat_input_value",
                        on_change=submit_message,
                    )
                    if st.button("���M", disabled=st.session_state.processing_message, key=f"chat_send_{sheet_name}"):
                        if not st.session_state.processing_message:
                            submit_message()
            except Exception as exc:  # noqa: BLE001
                st.error(f"�f�[�^�̓ǂݍ��ݒ��ɃG���[���������܂���: {exc}")
    else:
        st.info("�\�����铊���M����I�����Ă��������B")
with tab_list:
    st.markdown("### ?? �S�����ꗗ")
    st.markdown("�o�^����Ă���S�����̌��݂̏󋵂��ꗗ�Ŋm�F�ł��܂��B")

    def get_cache_key() -> str:
        now = datetime.datetime.now()
        cache_date = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d") if now.hour < 8 else now.strftime("%Y-%m-%d")
        return f"fund_data_{cache_date}"

    @st.cache_data(ttl=86400)
    def get_fund_status(sheet_name: str, cache_key: str) -> str:
        try:
            df = get_sheet_data(SPREADSHEET_ID, sheet_name)
            if df is None or df.empty:
                return "�f�[�^�Ȃ�"
            _, detailed = generate_technical_summary(df)
            conclusion = detailed[-1] if detailed else ""
            if "��������" in conclusion:
                return "��������"
            if "���萄��" in conclusion:
                return "���萄��"
            return "�l�q��"
        except Exception:
            return "���̓G���["

    @st.cache_data(ttl=86400)
    def get_fund_summary(sheet_name: str, cache_key: str) -> Dict[str, str]:
        try:
            df = get_sheet_data(SPREADSHEET_ID, sheet_name)
            if df is None or df.empty:
                return {
                    "������": sheet_name,
                    "�X�e�[�^�X": "�f�[�^�Ȃ�",
                    "����z": "-",
                    "�O����": "-",
                    "25���ړ�����": "-",
                    "200���ړ�����": "-",
                }

            latest_price = df["����z"].iloc[-1]
            previous_price = df["����z"].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - previous_price
            ma25_series = df["����z"].rolling(window=25).mean()
            ma200_series = df["����z"].rolling(window=200).mean()
            status = get_fund_status(sheet_name, cache_key)

            return {
                "������": sheet_name,
                "�X�e�[�^�X": status,
                "����z": f"{latest_price:,.0f}�~",
                "�O����": f"{price_change:+.0f}�~" if price_change != 0 else "0�~",
                "25���ړ�����": f"{ma25_series.iloc[-1]:,.0f}�~" if len(ma25_series.dropna()) else "-",
                "200���ړ�����": f"{ma200_series.iloc[-1]:,.0f}�~" if len(ma200_series.dropna()) else "-",
            }
        except Exception:
            return {
                "������": sheet_name,
                "�X�e�[�^�X": "�G���[",
                "����z": "-",
                "�O����": "-",
                "25���ړ�����": "-",
                "200���ړ�����": "-",
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
            st.caption(f"����X�V�\��: {next_update.strftime('%m/%d %H:%M')}")
        with col3:
            notifier = SlackNotifier()
            st.caption("?? Slack�ʒm: �L��" if notifier.is_configured() else "?? Slack�ʒm: ����")
        with col4:
            c4a, c4b = st.columns(2)
            with c4a:
                if st.button("�ʒm�e�X�g"):
                    st.session_state.test_slack = True
            with c4b:
                if st.button("�ύX�V�~�����[�g"):
                    st.session_state.simulate_change = True

        session_key = f"fund_data_loaded_{cache_key}"
        if session_key not in st.session_state:
            with st.spinner("�f�[�^��ǂݍ��ݒ�... (����8���Ɏ����X�V)"):
                fund_data = get_all_fund_data(available_sheets, cache_key)
                notifier = SlackNotifier()
                if notifier.is_configured():
                    try:
                        previous_status = load_previous_status(str(STORAGE_PATH))
                        status_changes = check_status_changes(previous_status, fund_data)
                        if status_changes:
                            sent = notifier.send_multiple_notifications(status_changes)
                            if sent:
                                st.success(f"?? {sent}���̓��������ύX��Slack�ɒʒm���܂���")
                        save_fund_status(fund_data, str(STORAGE_PATH))
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Slack�ʒm�����ŃG���[���������܂���: {exc}")
                st.session_state[session_key] = True
                st.success("�f�[�^��ǂݍ��݂܂����i����8���܂ō����\������܂��j")
        else:
            fund_data = get_all_fund_data(available_sheets, cache_key)
            st.info("?? �L���b�V������f�[�^��\�����i�����\���j")

        if st.session_state.get("test_slack"):
            notifier = SlackNotifier()
            if notifier.is_configured():
                import random

                if fund_data:
                    num_test = min(2, len(fund_data))
                    test_funds = random.sample(fund_data, num_test)
                    test_changes = [
                        {
                            "fund_name": fund["������"],
                            "old_status": "�l�q��",
                            "new_status": "��������",
                            "price": fund["����z"],
                            "price_change": fund["�O����"],
                        }
                        for fund in test_funds
                    ]
                else:
                    test_changes = [
                        {
                            "fund_name": "�e�X�g�����M��",
                            "old_status": "�l�q��",
                            "new_status": "��������",
                            "price": "10,000�~",
                            "price_change": "+50�~",
                        }
                    ]
                try:
                    with st.spinner("Slack�Ƀe�X�g�ʒm�𑗐M��..."):
                        sent = notifier.send_multiple_notifications(test_changes)
                    if sent:
                        st.success(f"?? {sent}���̃e�X�g�ʒm��Slack�ɑ��M���܂����I")
                        with st.expander("���M���e�̏ڍ�"):
                            for change in test_changes:
                                st.write(f"**{change['fund_name']}**")
                                st.write(f"�X�e�[�^�X�ύX: {change['old_status']} �� {change['new_status']}")
                                st.write(f"���݉��i: {change['price']} ({change['price_change']})")
                                st.write("---")
                    else:
                        st.error("?? Slack�ʒm�̑��M�Ɏ��s���܂���")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"?? Slack�ʒm�G���[: {exc}")
            else:
                st.error("?? SLACK_WEBHOOK_URL���ݒ肳��Ă��܂���")
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
                        if fund["�X�e�[�^�X"] != "��������":
                            fund["�X�e�[�^�X"] = "��������"
                    status_changes = check_status_changes(previous_status, modified)
                    if status_changes:
                        with st.spinner("�ύX�����o����Slack�ʒm�𑗐M��..."):
                            sent = notifier.send_multiple_notifications(status_changes)
                        if sent:
                            st.success(f"?? {sent}���̃X�e�[�^�X�ύX��Slack�ɒʒm���܂����I")
                            with st.expander("���o���ꂽ�ύX"):
                                for change in status_changes:
                                    st.write(f"**{change['fund_name']}**")
                                    st.write(f"�X�e�[�^�X�ύX: {change['old_status']} �� {change['new_status']}")
                                    st.write(f"���݉��i: {change['price']} ({change['price_change']})")
                                    st.write("---")
                        else:
                            st.error("?? Slack�ʒm�̑��M�Ɏ��s���܂���")
                    else:
                        st.info("?? �ύX�\�ȃt�@���h��������܂���ł���")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"?? �V�~�����[�V�����G���[: {exc}")
            else:
                st.error("?? SLACK_WEBHOOK_URL���ݒ肳��Ă��܂���")
            st.session_state.simulate_change = False

        summary_df = pd.DataFrame(fund_data)

        def style_status(val: str) -> str:
            if val == "��������":
                return "background-color: #d4edda; color: #155724"
            if val == "���萄��":
                return "background-color: #f8d7da; color: #721c24"
            if val == "�l�q��":
                return "background-color: #fff3cd; color: #856404"
            return "background-color: #f8f9fa; color: #6c757d"

        st.markdown("#### �����ꗗ�\")
        styled_df = summary_df.style.map(style_status, subset=["�X�e�[�^�X"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        st.markdown("#### ?? �T�}���[���v")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("��������", len([f for f in fund_data if f["�X�e�[�^�X"] == "��������"]))
        with col_b:
            st.metric("���萄��", len([f for f in fund_data if f["�X�e�[�^�X"] == "���萄��"]))
        with col_c:
            st.metric("�l�q��", len([f for f in fund_data if f["�X�e�[�^�X"] == "�l�q��"]))
        with col_d:
            st.metric("�f�[�^�Ȃ�/�G���[", len([f for f in fund_data if f["�X�e�[�^�X"] in {"�f�[�^�Ȃ�", "�G���[", "���̓G���["}]))

        st.markdown("#### ?? �t�B���^�[")
        status_filter = st.selectbox("�X�e�[�^�X�Ńt�B���^�[", ["���ׂ�", "��������", "���萄��", "�l�q��", "�f�[�^�Ȃ�/�G���["])
        if status_filter != "���ׂ�":
            if status_filter == "�f�[�^�Ȃ�/�G���[":
                filtered_df = summary_df[summary_df["�X�e�[�^�X"].isin(["�f�[�^�Ȃ�", "�G���[", "���̓G���["])]
            else:
                filtered_df = summary_df[summary_df["�X�e�[�^�X"] == status_filter]
            st.markdown(f"#### �t�B���^�[����: {status_filter}")
            st.dataframe(filtered_df.style.map(style_status, subset=["�X�e�[�^�X"]), use_container_width=True, hide_index=True)
    else:
        st.error("�V�[�g�f�[�^�̓ǂݍ��݂Ɏ��s���܂����B")
with tab_corr:
    st.header("?? ���֕���")
    st.markdown("�����Ԃ̉��i�ϓ����֊֌W�𕪐͂��܂��i����1�N�Ԃ̃f�[�^���g�p�j")

    try:
        available_sheets_corr = available_sheets if available_sheets else []
    except Exception:
        available_sheets_corr = []

    st.caption(f"���p�\�Ȗ�����: {len(available_sheets_corr)}����")

    if available_sheets_corr:
        def get_cached_correlation_data(sheet_list: List[str], period_days: int):
            return get_correlation_data(sheet_list, period_days)

        def get_cached_correlation_matrix(key):
            correlation_data, _ = key
            return calculate_correlation_matrix(correlation_data)

        settings_col, result_col = st.columns([1, 3])
        with settings_col:
            st.subheader("?? ���͐ݒ�")
            selected_fund = st.selectbox(
                "�ڍו��͂��������I��",
                options=["�S�̃}�g���b�N�X�\��"] + available_sheets_corr,
                help="����̖�����I������ƁA���̖����Ƒ��̖����Ƃ̑��֊֌W��\�����܂�",
            )
            period_options = {
                "1�N�ԁi252�c�Ɠ��j": 252,
                "6�����ԁi126�c�Ɠ��j": 126,
                "3�����ԁi63�c�Ɠ��j": 63,
            }
            selected_period = st.selectbox("���͊���", options=list(period_options.keys()), index=0)
            period_days = period_options[selected_period]
            st.info(f"���͊���: {selected_period}")
            st.caption("���ׂĂ̑��֌W����\�����܂�")

        with result_col:
            st.subheader("?? ���֕��͌���")
            with st.spinner(f"���֕��̓f�[�^���v�Z��...�i{selected_period}�j"):
                correlation_data = get_cached_correlation_data(available_sheets_corr, period_days)
                if correlation_data:
                    st.info(f"?? �f�[�^���擾����������: {len(correlation_data)}����")
                    preview = [f"{name}: {len(series)}����" for name, series in list(correlation_data.items())[:3]]
                    st.caption(", ".join(preview) + ("..." if len(correlation_data) > 3 else ""))
                else:
                    st.error("?? ���֕��̓f�[�^���擾�ł��܂���ł����B")
                correlation_matrix = get_cached_correlation_matrix((correlation_data, period_days)) if correlation_data else None

            if correlation_matrix is not None and not correlation_matrix.empty:
                if selected_fund == "�S�̃}�g���b�N�X�\��":
                    st.markdown("#### ?? �S�������փ}�g���b�N�X")
                    fig = create_correlation_heatmap(correlation_matrix)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### ?? ���֓��v�T�}���[")
                    c1, c2, c3, c4 = st.columns(4)
                    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)).stack()
                    with c1:
                        st.metric("���ϑ���", f"{upper_triangle.mean():.3f}")
                    with c2:
                        st.metric("�ő告��", f"{upper_triangle.max():.3f}")
                    with c3:
                        st.metric("�ŏ�����", f"{upper_triangle.min():.3f}")
                    with c4:
                        st.metric("�����փy�A��", len(upper_triangle[upper_triangle.abs() > 0.7]))

                    st.markdown("#### ?? ���ւ̋����y�A�i���10�ʁj")
                    pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i + 1, len(correlation_matrix.columns)):
                            fund_a = correlation_matrix.columns[i]
                            fund_b = correlation_matrix.columns[j]
                            corr_val = correlation_matrix.iloc[i, j]
                            strength = (
                                "�������̑���"
                                if corr_val > 0.7
                                else "�����x�̐��̑���"
                                if corr_val > 0.3
                                else "�ア����"
                                if corr_val > -0.3
                                else "�����x�̕��̑���"
                                if corr_val > -0.7
                                else "�������̑���"
                            )
                            pairs.append({"����A": fund_a, "����B": fund_b, "���֌W��": f"{corr_val:.3f}", "���ւ̋���": strength})
                    pairs_df = pd.DataFrame(pairs)
                    pairs_df["abs_corr"] = pairs_df["���֌W��"].astype(float).abs()
                    st.dataframe(pairs_df.nlargest(10, "abs_corr").drop(columns=["abs_corr"]), use_container_width=True, hide_index=True)
                else:
                    st.markdown(f"#### ?? {selected_fund}�Ƃ̑��֕���")
                    fund_corr = get_fund_correlations(correlation_matrix, selected_fund)
                    if fund_corr is not None and not fund_corr.empty:
                        bar_fig = create_correlation_bar_chart(fund_corr, selected_fund)
                        if bar_fig:
                            st.plotly_chart(bar_fig, use_container_width=True)
                        summary_table = create_correlation_summary_table(fund_corr, selected_fund)
                        if summary_table is not None:
                            st.markdown("#### ?? �ڍו��͌���")
                            st.dataframe(summary_table, use_container_width=True, hide_index=True)
                        st.markdown("#### ?? �������f�ւ̊��p")
                        high_pos = fund_corr[fund_corr > 0.7]
                        if not high_pos.empty:
                            st.success(f"**�������̑��֖���**: {', '.join(high_pos.index[:3])}")
                            st.markdown(f"�� {selected_fund}�Ɠ������ɓ������߁A���U���ʂ͌���I")
                        high_neg = fund_corr[fund_corr < -0.7]
                        if not high_neg.empty:
                            st.error(f"**�������̑��֖���**: {', '.join(high_neg.index[:3])}")
                            st.markdown(f"�� {selected_fund}�Ƌt�����ɓ������߁A�w�b�W���ʂ����҂ł���")
                        low_corr = fund_corr[fund_corr.abs() < 0.3]
                        if not low_corr.empty:
                            st.info(f"**���U�������**: {', '.join(low_corr.index[:3])}")
                            st.markdown(f"�� {selected_fund}�Ƃ̊֘A�����Ⴍ�A���U���ʂ����҂ł���")
                    else:
                        st.error(f"{selected_fund}�̑��փf�[�^��������܂���B")
            else:
                st.error("���֕��͂ɏ\���ȃf�[�^������܂���B��������Ԃ��m�F���Ă��������B")

        st.markdown("---")
        st.markdown(
            """
#### ?? ���֕��͂ɂ���
**���֌W���̉��߁F**
- **+0.7�`+1.0**: �������̑��ցi���������ɓ����j
- **+0.3�`+0.7**: �����x�̐��̑���
- **-0.3�`+0.3**: �ア���ցi�֘A�����Ⴂ�j
- **-0.7�`-0.3**: �����x�̕��̑���
- **-1.0�`-0.7**: �������̑��ցi�t�����ɓ����j

**�����ւ̊��p�F**
- ���ւ̍����������m�͕��U�������ʂ��Ⴂ
- ���ւ̒Ⴂ������g�ݍ��킹�邱�ƂŃ��X�N���U���\
- ���̑��ւ���������̓w�b�W���ʂ����҂ł���
"""
        )
    else:
        st.error("�V�[�g�f�[�^�̓ǂݍ��݂Ɏ��s���܂����B")

st.markdown("---")
st.markdown("�f�[�^�o�T: Google Spreadsheet")
