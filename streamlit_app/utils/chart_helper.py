from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class BollingerAnalysis:
    buy_signal: bool
    sell_signal: bool
    position: str
    is_expanding: bool
    middle_band: float
    upper_band: float
    lower_band: float


def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_volatility(data: pd.Series, window: int = 20) -> pd.Series:
    return data.rolling(window=window).std()


def get_trend_arrow(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return ""
    if current > previous:
        return "��"
    if current < previous:
        return "��"
    return "��"


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band_1 = middle_band + std_dev
    lower_band_1 = middle_band - std_dev
    upper_band_2 = middle_band + num_std * std_dev
    lower_band_2 = middle_band - num_std * std_dev
    return middle_band, upper_band_1, lower_band_1, upper_band_2, lower_band_2


def calculate_dmi(price_series: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = price_series.rolling(window=2).max()
    low = price_series.rolling(window=2).min()
    tr = (high - low).fillna(0)

    price_diff = price_series.diff()
    high_diff = price_diff.clip(lower=0)
    low_diff = (-price_diff).clip(lower=0)

    plus_dm = ((high_diff > low_diff) & (high_diff > 0)).astype(float) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)).astype(float) * low_diff

    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()

    plus_di = pd.Series(0.0, index=price_series.index, dtype=float)
    minus_di = pd.Series(0.0, index=price_series.index, dtype=float)
    mask = tr_smooth != 0
    plus_di.loc[mask] = (plus_dm_smooth[mask] / tr_smooth[mask]) * 100
    minus_di.loc[mask] = (minus_dm_smooth[mask] / tr_smooth[mask]) * 100

    dx = pd.Series(0.0, index=price_series.index, dtype=float)
    di_sum = plus_di + minus_di
    mask_dx = di_sum != 0
    dx.loc[mask_dx] = (plus_di[mask_dx] - minus_di[mask_dx]).abs() / di_sum[mask_dx] * 100
    adx = dx.rolling(window=period).mean()

    return plus_di, minus_di, adx


def add_ma_highlight(fig: go.Figure, df: pd.DataFrame, ma_column: str, color: str, row: int = 1, col: int = 1) -> None:
    if ma_column not in df.columns or df[ma_column].dropna().empty:
        return

    ma_value = df[ma_column].iloc[-1]
    current_price = df["����z"].iloc[-1]

    fig.add_hline(
        y=ma_value,
        line=dict(color=color, width=1, dash="dot"),
        row=row,
        col=col,
        annotation=dict(text=f"{ma_column}: {ma_value:,.0f}�~", xref="paper", x=1.02, showarrow=False, font=dict(color=color)),
    )

    if ma_value and abs(current_price - ma_value) / ma_value <= 0.01:
        fig.add_hrect(
            y0=ma_value * 0.99,
            y1=ma_value * 1.01,
            fillcolor=color,
            opacity=0.1,
            line_width=0,
            row=row,
            col=col,
            annotation=dict(text="�d�v���x��", xref="paper", x=0, showarrow=False),
        )


def create_price_chart(df: pd.DataFrame, show_indicators: Dict[str, bool] | None = None) -> go.Figure:
    if show_indicators is None:
        show_indicators = {"�ړ����ϐ�": True}

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.2, 0.2, 0.15, 0.15],
    )

    fig.add_trace(
        go.Scatter(
            x=df["���t"],
            y=df["����z"],
            name="����z",
            line=dict(color="#1f77b4", width=2),
            connectgaps=True,
        ),
        row=1,
        col=1,
    )

    if show_indicators.get("�ړ����ϐ�", False):
        if "25���ړ�����" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["���t"],
                    y=df["25���ړ�����"],
                    name="25���ړ�����",
                    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                    connectgaps=True,
                ),
                row=1,
                col=1,
            )
            add_ma_highlight(fig, df, "25���ړ�����", "#ff7f0e")

        if "200���ړ�����" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["���t"],
                    y=df["200���ړ�����"],
                    name="200���ړ�����",
                    line=dict(color="#2ca02c", width=1.5, dash="dash"),
                    connectgaps=True,
                ),
                row=1,
                col=1,
            )
            add_ma_highlight(fig, df, "200���ړ�����", "#2ca02c")

    if show_indicators.get("�{�����W���[�o���h", False):
        middle, upper1, lower1, upper2, lower2 = calculate_bollinger_bands(df["����z"])
        fig.add_trace(
            go.Scatter(
                x=df["���t"],
                y=df["����z"],
                name="����z",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
                connectgaps=True,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(go.Scatter(x=df["���t"], y=middle, name="BB (SMA)", line=dict(color="gray", width=1), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=upper1, name="+1��", line=dict(color="orange", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=lower1, name="-1��", line=dict(color="orange", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=upper2, name="+2��", line=dict(color="red", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=lower2, name="-2��", line=dict(color="red", width=1, dash="dash"), connectgaps=True), row=2, col=1)

    if show_indicators.get("RSI", False):
        rsi = calculate_rsi(df["����z"])
        fig.add_trace(
            go.Scatter(
                x=df["���t"],
                y=rsi,
                name="RSI (14)",
                line=dict(color="#9467bd", width=1.5),
                connectgaps=True,
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    if show_indicators.get("MACD", False):
        macd, signal = calculate_macd(df["����z"])
        fig.add_trace(go.Scatter(x=df["���t"], y=macd, name="MACD", line=dict(color="#17becf", width=1.5), connectgaps=True), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=signal, name="Signal", line=dict(color="#bcbd22", width=1.5), connectgaps=True), row=4, col=1)

    if show_indicators.get("DMI", False):
        plus_di, minus_di, adx = calculate_dmi(df["����z"])
        fig.add_trace(go.Scatter(x=df["���t"], y=plus_di, name="+DI", line=dict(color="red", width=1.5), connectgaps=True), row=5, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=minus_di, name="-DI", line=dict(color="blue", width=1.5), connectgaps=True), row=5, col=1)
        fig.add_trace(go.Scatter(x=df["���t"], y=adx, name="ADX", line=dict(color="green", width=1.5), connectgaps=True), row=5, col=1)
        fig.update_yaxes(title_text="DMI", range=[-5, 105], row=5, col=1)

    fig.update_layout(
        title_text="�e�N�j�J�����̓`���[�g",
        height=1200,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    price_min = df["����z"].min()
    price_max = df["����z"].max()
    price_margin = (price_max - price_min) * 0.1 if price_max != price_min else price_max * 0.1
    fig.update_yaxes(title_text="����z", range=[price_min - price_margin, price_max + price_margin], row=1, col=1)

    if show_indicators.get("�{�����W���[�o���h", False):
        fig.update_yaxes(title_text="�{�����W���[�o���h", range=[price_min - price_margin, price_max + price_margin], row=2, col=1)

    if show_indicators.get("RSI", False):
        fig.update_yaxes(title_text="RSI", range=[-5, 105], row=3, col=1)

    if show_indicators.get("MACD", False):
        macd, signal = calculate_macd(df["����z"])
        macd_min = min(macd.min(), signal.min())
        macd_max = max(macd.max(), signal.max())
        macd_margin = (macd_max - macd_min) * 0.1 if macd_max != macd_min else abs(macd_max) * 0.1
        fig.update_yaxes(title_text="MACD", range=[macd_min - macd_margin, macd_max + macd_margin], row=4, col=1)

    fig.update_xaxes(title_text="���t", row=5, col=1)
    return fig


def is_range_bound_market(df: pd.DataFrame, window: int = 20, threshold: float = 0.05) -> Tuple[float, Dict[str, float]]:
    recent_data = df["����z"].tail(window)
    high_price = recent_data.max()
    low_price = recent_data.min()
    mid_price = (high_price + low_price) / 2
    price_range = (high_price - low_price) / mid_price if mid_price else 0

    if "25���ړ�����" in df.columns and df["25���ړ�����"].dropna().shape[0] >= window:
        ma25 = df["25���ړ�����"].tail(window)
        ma_slope = (ma25.iloc[-1] - ma25.iloc[0]) / ma25.iloc[0] if ma25.iloc[0] else 0
    else:
        ma_slope = 0

    price_range_score = max(0, 1 - (price_range / threshold))
    trend_score = max(0, 1 - (abs(ma_slope) / (threshold / 2)))
    range_score = (price_range_score + trend_score) / 2

    range_info = {
        "high_price": float(high_price),
        "low_price": float(low_price),
        "range_width": float(price_range * 100),
        "ma_slope": float(ma_slope * 100),
        "range_score": float(range_score),
    }

    return range_score, range_info


def is_band_expanding(std_dev_series: pd.Series, window: int = 5) -> bool:
    recent_std = std_dev_series.tail(window)
    if len(recent_std) < window or recent_std.iloc[0] == 0:
        return False
    std_change = (recent_std.iloc[-1] - recent_std.iloc[0]) / recent_std.iloc[0]
    return std_change > 0.02


def analyze_bollinger_bands(df: pd.DataFrame) -> BollingerAnalysis:
    window = 20
    middle_band = df["����z"].rolling(window=window).mean()
    std_dev = df["����z"].rolling(window=window).std()

    upper_band_2 = middle_band + (std_dev * 2)
    lower_band_2 = middle_band - (std_dev * 2)

    current_price = df["����z"].iloc[-1]
    prev_price = df["����z"].iloc[-2]

    expanding = is_band_expanding(std_dev)
    buy_signal = bool(current_price > upper_band_2.iloc[-1] and prev_price <= upper_band_2.iloc[-2] and expanding)
    sell_signal = bool(current_price < lower_band_2.iloc[-1] and prev_price >= lower_band_2.iloc[-2] and expanding)

    position = "����"
    if current_price > upper_band_2.iloc[-1]:
        position = "����˔j"
    elif current_price < lower_band_2.iloc[-1]:
        position = "�����˔j"

    return BollingerAnalysis(
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        position=position,
        is_expanding=expanding,
        middle_band=float(middle_band.iloc[-1]),
        upper_band=float(upper_band_2.iloc[-1]),
        lower_band=float(lower_band_2.iloc[-1]),
    )


def generate_technical_summary(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    summary: List[str] = []
    detailed: List[str] = []

    bb = analyze_bollinger_bands(df)
    bb_signal = ""
    if bb.buy_signal:
        bb_signal = "�����V�O�i��"
        summary.append("�E�{�����W���[�o���h: +2�Ђ��㔲���A�o���h�g�咆�i�����j")
    elif bb.sell_signal:
        bb_signal = "����V�O�i��"
        summary.append("�E�{�����W���[�o���h: -2�Ђ��������A�o���h�g�咆�i����j")
    else:
        summary.append(f"�E�{�����W���[�o���h: {bb.position}�̏��")

    detailed.append(
        f"""
#### �{�����W���[�o���h����
- **���݂̏��**: {bb.position}
- **�o���h�̏��**: {'�g��X��' if bb.is_expanding else '���k�X��'}
- **�V�O�i��**: {bb_signal or '���m�ȃV�O�i���Ȃ�'}
- **���f����**:
  - ����z: {df['����z'].iloc[-1]:,.0f}�~
  - ���S�� (20��SMA): {bb.middle_band:,.0f}�~
  - +2��: {bb.upper_band:,.0f}�~
  - -2��: {bb.lower_band:,.0f}�~
- **�g���[�h�헪**:
  - �o���h�����k����g��ɓ]���A+2�Ђ��㔲�����ꍇ�͔����V�O�i��
  - �o���h�����k����g��ɓ]���A-2�Ђ����������ꍇ�͔���V�O�i��
  - �o���h���ł̎���͍T���߂ɂ��A�u���C�N�A�E�g��҂�
""",
    )

    range_score, range_info = is_range_bound_market(df)
    if range_score > 0.8:
        range_message = "���m�ȃ{�b�N�X����"
    elif range_score > 0.6:
        range_message = "�{�b�N�X������̒l����"
    elif range_score > 0.4:
        range_message = "���������̂���l����"
    else:
        range_message = "�������ւ̒l����"

    summary.append(f"�E{range_message}�F���i�� {range_info['low_price']:,.0f}�~ �` {range_info['high_price']:,.0f}�~")
    detailed.append(
        f"""
#### ���i�ϓ��p�^�[������
- **{range_message}**���`�����Ă��܂�
- ���i��: {range_info['low_price']:,.0f}�~ �` {range_info['high_price']:,.0f}�~
- �ϓ���: {range_info['range_width']:.1f}%
- �g�����h�̌X��: {range_info['ma_slope']:.1f}%
- �����W����x: {range_info['range_score']:.1f}
- �����헪: {'�㉺�̉��i�т𗘗p�������������' if range_score > 0.6 else '�������ɉ���������𐄏�'}
""",
    )

    current_ma25 = None
    current_ma200 = None
    ma_status = "�f�[�^�s��"
    if "25���ړ�����" in df.columns and "200���ړ�����" in df.columns:
        ma25 = df["25���ړ�����"].dropna().iloc[-2:]
        ma200 = df["200���ړ�����"].dropna().iloc[-2:]
        if len(ma25) == 2 and len(ma200) == 2:
            current_ma25 = ma25.iloc[-1]
            current_ma200 = ma200.iloc[-1]
            price_vs_ma = df["����z"].iloc[-1]

            if ma25.iloc[0] < ma200.iloc[0] and ma25.iloc[1] > ma200.iloc[1]:
                ma_status = "�S�[���f���N���X�F���C"
                summary.append("�E�S�[���f���N���X�����F���C")
            elif ma25.iloc[0] > ma200.iloc[0] and ma25.iloc[1] < ma200.iloc[1]:
                ma_status = "�f�b�h�N���X�F��C"
                summary.append("�E�f�b�h�N���X�����F��C")
            elif current_ma25 > current_ma200:
                ma_status = "�㏸�g�����h�p��"
                summary.append("�E25���ړ����ϐ���200���ړ����ϐ��̏���F���C")
            else:
                ma_status = "���~�g�����h�p��"
                summary.append("�E25���ړ����ϐ���200���ړ����ϐ��̉����F��C")

            detailed.append(
                f"""
#### �ړ����ϐ�����
- **�g�����h���**: {ma_status}
- 25���ړ����ϐ�: {current_ma25:,.0f}�~
- 200���ړ����ϐ�: {current_ma200:,.0f}�~
- ����z�̈ʒu: �ړ����ϐ�����{((price_vs_ma - current_ma25) / current_ma25 * 100):.1f}%
- �g�����h�̋���: {'����' if abs((current_ma25 - current_ma200) / current_ma200) > 0.03 else '�ア'}
""",
            )

    rsi_series = calculate_rsi(df["����z"])
    rsi_value = float(rsi_series.iloc[-1])
    if rsi_value > 70:
        rsi_status = "����ꂷ��"
        summary.append(f"�ERSI��{rsi_value:.1f}�ŁA����ꂷ���̐���")
    elif rsi_value < 30:
        rsi_status = "����ꂷ��"
        summary.append(f"�ERSI��{rsi_value:.1f}�ŁA����ꂷ���̐���")
    else:
        rsi_status = "����"
        summary.append(f"�ERSI��{rsi_value:.1f}�ŁA�����I�Ȑ���")

    detailed.append(
        f"""
#### RSI�i���Η͎w���j����
- ���݂�RSI: {rsi_value:.1f}
- **�s����**: {rsi_status}
- ���f: {'�Z���I�Ȓ����̉\������' if rsi_value > 70 else '�Z���I�Ȕ����̉\������' if rsi_value < 30 else '�K�������Ő��ڒ�'}
""",
    )

    macd_series, signal_series = calculate_macd(df["����z"])
    latest_macd = macd_series.iloc[-1]
    latest_signal = signal_series.iloc[-1]
    prev_macd = macd_series.iloc[-2]
    prev_signal = signal_series.iloc[-2]

    if latest_macd > latest_signal and prev_macd <= prev_signal:
        macd_status = "�����V�O�i��"
        summary.append("�EMACD���V�O�i�����C����������ɃN���X�i�����V�O�i���j")
    elif latest_macd < latest_signal and prev_macd >= prev_signal:
        macd_status = "����V�O�i��"
        summary.append("�EMACD���V�O�i�����C�����������ɃN���X�i����V�O�i���j")
    elif latest_macd > latest_signal:
        macd_status = "�㏸�g�����h�p��"
        summary.append("�EMACD�̓V�O�i�����C���̏���Ő���")
    else:
        macd_status = "���~�g�����h�p��"
        summary.append("�EMACD�̓V�O�i�����C���̉����Ő���")

    detailed.append(
        f"""
#### MACD����
- **���݂̏��**: {macd_status}
- MACD�l: {latest_macd:.2f}
- �V�O�i���l: {latest_signal:.2f}
- MACD�q�X�g�O����: {(latest_macd - latest_signal):.2f}
- �g�����h�̋���: {'����' if abs(latest_macd - latest_signal) > abs(prev_macd - prev_signal) else '�ア'}
""",
    )

    plus_di, minus_di, adx = calculate_dmi(df["����z"])
    current_plus = plus_di.iloc[-1]
    current_minus = minus_di.iloc[-1]
    prev_plus = plus_di.iloc[-2]
    prev_minus = minus_di.iloc[-2]
    current_adx = adx.iloc[-1]

    if current_plus > current_minus and prev_plus <= prev_minus:
        dmi_status = "�����V�O�i��"
        summary.append("�EDMI: +DI��-DI��������㔲���i�����V�O�i���j")
    elif current_plus < current_minus and prev_plus >= prev_minus:
        dmi_status = "����V�O�i��"
        summary.append("�EDMI: +DI��-DI���ォ�牺�����i����V�O�i���j")
    elif current_plus > current_minus:
        dmi_status = "�㏸�g�����h"
        summary.append("�EDMI: +DI��-DI�̏���i�㏸�g�����h�j")
    else:
        dmi_status = "���~�g�����h"
        summary.append("�EDMI: +DI��-DI�̉����i���~�g�����h�j")

    detailed.append(
        f"""
#### DMI����
- **���݂̏��**: {dmi_status}
- +DI: {current_plus:.1f}
- -DI: {current_minus:.1f}
- ADX: {current_adx:.1f}�i�g�����h�̋����j
- **�g�����h�̋���**: {'����' if current_adx > 25 else '�ア'}
- **�V�O�i������**:
  - +DI��-DI��������㔲�����ꍇ�͔����V�O�i��
  - +DI��-DI���ォ�牺�������ꍇ�͔���V�O�i��
""",
    )

    buy_factors: List[str] = []
    sell_factors: List[str] = []
    neutral_factors: List[str] = []

    ma_bullish = current_ma25 is not None and current_ma200 is not None and current_ma25 > current_ma200
    if current_ma25 is not None and current_ma200 is not None:
        if ma_bullish:
            if ma_status == "�S�[���f���N���X�F���C":
                buy_factors.append("25���ړ����ϐ���200���ړ����ϐ����S�[���f���N���X")
            else:
                buy_factors.append("25���ړ����ϐ���200���ړ����ϐ��̏���Ő���")
        else:
            if ma_status == "�f�b�h�N���X�F��C":
                sell_factors.append("25���ړ����ϐ���200���ړ����ϐ����f�b�h�N���X")
            else:
                sell_factors.append("25���ړ����ϐ���200���ړ����ϐ��̉����Ő���")

    if bb.buy_signal:
        buy_factors.append("�{�����W���[�o���h������u���C�N�A�E�g�i�o���h�g�咆�j")
    elif bb.sell_signal:
        sell_factors.append("�{�����W���[�o���h�������u���C�N�A�E�g�i�o���h�g�咆�j")
    else:
        neutral_factors.append(f"�{�����W���[�o���h: {bb.position}�̏��")

    if macd_status == "�����V�O�i��":
        buy_factors.append("MACD���V�O�i�����C����������ɃN���X")
    elif macd_status == "����V�O�i��":
        sell_factors.append("MACD���V�O�i�����C�����������ɃN���X")
    elif latest_macd > latest_signal:
        buy_factors.append("MACD���V�O�i�����C���̏���Ő���")
    else:
        sell_factors.append("MACD���V�O�i�����C���̉����Ő���")

    if rsi_value < 30:
        buy_factors.append(f"RSI��{rsi_value:.1f}�Ŕ���ꂷ���̐���")
    elif rsi_value > 70:
        sell_factors.append(f"RSI��{rsi_value:.1f}�Ŕ���ꂷ���̐���")
    else:
        neutral_factors.append(f"RSI {rsi_value:.1f}�Œ����I�Ȑ���")

    if dmi_status == "�����V�O�i��":
        buy_factors.append("+DI��-DI��������㔲���i�㏸�g�����h�J�n�j")
    elif dmi_status == "����V�O�i��":
        sell_factors.append("+DI��-DI���ォ�牺�����i���~�g�����h�J�n�j")
    elif current_plus > current_minus:
        buy_factors.append("+DI��-DI�̏���i�㏸�g�����h�p���j")
    else:
        sell_factors.append("+DI��-DI�̉����i���~�g�����h�p���j")

    strong_buy = 0.0
    strong_sell = 0.0
    if current_ma25 is not None and current_ma200 is not None:
        if ma_bullish and ma_status in ("�S�[���f���N���X�F���C", "�㏸�g�����h�p��"):
            strong_buy += 1
        elif not ma_bullish and ma_status in ("�f�b�h�N���X�F��C", "���~�g�����h�p��"):
            strong_sell += 1
    if bb.buy_signal:
        strong_buy += 1
    elif bb.sell_signal:
        strong_sell += 1
    if macd_status == "�����V�O�i��":
        strong_buy += 1
    elif macd_status == "����V�O�i��":
        strong_sell += 1
    if rsi_value < 30:
        strong_buy += 0.5
    elif rsi_value > 70:
        strong_sell += 0.5

    if strong_buy >= 2 and strong_buy > strong_sell:
        decision = "��������"
    elif strong_sell >= 2 and strong_sell > strong_buy:
        decision = "���萄��"
    else:
        decision = "�l�q��"

    conclusion = """### �����_
���݂̎s�����**{decision}**�Ɣ��f����܂��B

**�����v�f**
""".format(decision=decision)

    conclusion += "".join(f"�E{factor}\n" for factor in buy_factors) if buy_factors else "�E�Y���Ȃ�\n"
    conclusion += "\n**����v�f**\n"
    conclusion += "".join(f"�E{factor}\n" for factor in sell_factors) if sell_factors else "�E�Y���Ȃ�\n"

    if neutral_factors:
        conclusion += "\n**�����v�f**\n"
        conclusion += "".join(f"�E{factor}\n" for factor in neutral_factors)

    if decision == "��������":
        reasoning = "�d�v�Ȏw�W�����������V�O�i���������Ă���A�㏸�g�����h���D���ł��B"
    elif decision == "���萄��":
        reasoning = "�d�v�Ȏw�W����������V�O�i���������Ă���A���~�g�����h���D���ł��B"
    else:
        reasoning = "�V�O�i�������݂��Ă��邽�߁A���m�ȕ��������o��܂őҋ@�𐄏����܂��B"

    conclusion += f"\n**�������f**: �����v�f{len(buy_factors)}�A����v�f{len(sell_factors)}�ł����A�d�v�w�W�̕��͂ɂ��{decision}�Ɣ��f���܂��B\n**���f���R**: {reasoning}"
    detailed.append(conclusion)
    return summary, detailed
