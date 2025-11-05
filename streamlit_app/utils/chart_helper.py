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
        return "↑"
    if current < previous:
        return "↓"
    return "→"


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


def calculate_ichimoku(price_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    一目均衡表の5つの構成要素を計算

    Returns:
        tenkan_sen: 転換線（9日間の最高値と最安値の平均）
        kijun_sen: 基準線（26日間の最高値と最安値の平均）
        senkou_span_a: 先行スパン1（転換線と基準線の平均を26日先行）
        senkou_span_b: 先行スパン2（52日間の最高値と最安値の平均を26日先行）
        chikou_span: 遅行スパン（当日の終値を26日遅行）
    """
    # 転換線：過去9日間の最高値と最安値の平均
    high_9 = price_series.rolling(window=9).max()
    low_9 = price_series.rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2

    # 基準線：過去26日間の最高値と最安値の平均
    high_26 = price_series.rolling(window=26).max()
    low_26 = price_series.rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2

    # 先行スパン1：転換線と基準線の平均を26日先行
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # 先行スパン2：過去52日間の最高値と最安値の平均を26日先行
    high_52 = price_series.rolling(window=52).max()
    low_52 = price_series.rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)

    # 遅行スパン：当日の終値を26日遅行
    chikou_span = price_series.shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


def add_ma_highlight(fig: go.Figure, df: pd.DataFrame, ma_column: str, color: str, row: int = 1, col: int = 1) -> None:
    if ma_column not in df.columns or df[ma_column].dropna().empty:
        return

    ma_value = df[ma_column].iloc[-1]
    current_price = df["基準価額"].iloc[-1]

    fig.add_hline(
        y=ma_value,
        line=dict(color=color, width=1, dash="dot"),
        row=row,
        col=col,
        annotation=dict(text=f"{ma_column}: {ma_value:,.0f}円", xref="paper", x=1.02, showarrow=False, font=dict(color=color)),
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
            annotation=dict(text="重要レベル", xref="paper", x=0, showarrow=False),
        )


def create_price_chart(df: pd.DataFrame, show_indicators: Dict[str, bool] | None = None) -> go.Figure:
    if show_indicators is None:
        show_indicators = {"移動平均線": True}

    # メモリ削減：データポイント数を制限（250件以上の場合は間引き）
    if len(df) > 250:
        step = max(1, len(df) // 250)
        df = df.iloc[::step].copy()

    # 一目均衡表が有効な場合は6行、それ以外は5行
    num_rows = 6 if show_indicators.get("一目均衡表", False) else 5
    row_heights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1] if num_rows == 6 else [0.3, 0.2, 0.2, 0.15, 0.15]

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=df["日付"],
            y=df["基準価額"],
            name="基準価額",
            line=dict(color="#1f77b4", width=2),
            connectgaps=True,
            mode="lines",  # マーカーなしでメモリ削減
        ),
        row=1,
        col=1,
    )

    if show_indicators.get("移動平均線", False):
        if "25日移動平均" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["日付"],
                    y=df["25日移動平均"],
                    name="25日移動平均",
                    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                    connectgaps=True,
                ),
                row=1,
                col=1,
            )
            add_ma_highlight(fig, df, "25日移動平均", "#ff7f0e")

        if "200日移動平均" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["日付"],
                    y=df["200日移動平均"],
                    name="200日移動平均",
                    line=dict(color="#2ca02c", width=1.5, dash="dash"),
                    connectgaps=True,
                ),
                row=1,
                col=1,
            )
            add_ma_highlight(fig, df, "200日移動平均", "#2ca02c")

    if show_indicators.get("ボリンジャーバンド", False):
        middle, upper1, lower1, upper2, lower2 = calculate_bollinger_bands(df["基準価額"])
        fig.add_trace(
            go.Scatter(
                x=df["日付"],
                y=df["基準価額"],
                name="基準価額",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
                connectgaps=True,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(go.Scatter(x=df["日付"], y=middle, name="BB (SMA)", line=dict(color="gray", width=1), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=upper1, name="+1σ", line=dict(color="orange", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=lower1, name="-1σ", line=dict(color="orange", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=upper2, name="+2σ", line=dict(color="red", width=1, dash="dash"), connectgaps=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=lower2, name="-2σ", line=dict(color="red", width=1, dash="dash"), connectgaps=True), row=2, col=1)

    if show_indicators.get("RSI", False):
        rsi = calculate_rsi(df["基準価額"])
        fig.add_trace(
            go.Scatter(
                x=df["日付"],
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
        macd, signal = calculate_macd(df["基準価額"])
        fig.add_trace(go.Scatter(x=df["日付"], y=macd, name="MACD", line=dict(color="#17becf", width=1.5), connectgaps=True), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=signal, name="Signal", line=dict(color="#bcbd22", width=1.5), connectgaps=True), row=4, col=1)

    dmi_row = 5
    ichimoku_row = 6

    if show_indicators.get("DMI", False):
        plus_di, minus_di, adx = calculate_dmi(df["基準価額"])
        fig.add_trace(go.Scatter(x=df["日付"], y=plus_di, name="+DI", line=dict(color="red", width=1.5), connectgaps=True), row=dmi_row, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=minus_di, name="-DI", line=dict(color="blue", width=1.5), connectgaps=True), row=dmi_row, col=1)
        fig.add_trace(go.Scatter(x=df["日付"], y=adx, name="ADX", line=dict(color="green", width=1.5), connectgaps=True), row=dmi_row, col=1)
        fig.update_yaxes(title_text="DMI", range=[-5, 105], row=dmi_row, col=1)

    if show_indicators.get("一目均衡表", False):
        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df["基準価額"])

        # 基準価額を表示
        fig.add_trace(
            go.Scatter(
                x=df["日付"],
                y=df["基準価額"],
                name="基準価額",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
                connectgaps=True,
            ),
            row=ichimoku_row,
            col=1,
        )

        # 転換線（赤）
        fig.add_trace(
            go.Scatter(x=df["日付"], y=tenkan, name="転換線", line=dict(color="red", width=1.5), connectgaps=True),
            row=ichimoku_row,
            col=1,
        )

        # 基準線（青）
        fig.add_trace(
            go.Scatter(x=df["日付"], y=kijun, name="基準線", line=dict(color="blue", width=1.5), connectgaps=True),
            row=ichimoku_row,
            col=1,
        )

        # 先行スパン1（緑）
        fig.add_trace(
            go.Scatter(
                x=df["日付"],
                y=senkou_a,
                name="先行スパン1",
                line=dict(color="green", width=1, dash="dot"),
                connectgaps=True,
            ),
            row=ichimoku_row,
            col=1,
        )

        # 先行スパン2（オレンジ）
        fig.add_trace(
            go.Scatter(
                x=df["日付"],
                y=senkou_b,
                name="先行スパン2",
                line=dict(color="orange", width=1, dash="dot"),
                connectgaps=True,
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.2)",
            ),
            row=ichimoku_row,
            col=1,
        )

        # 遅行スパン（紫）
        fig.add_trace(
            go.Scatter(x=df["日付"], y=chikou, name="遅行スパン", line=dict(color="purple", width=1.5), connectgaps=True),
            row=ichimoku_row,
            col=1,
        )

    fig.update_layout(
        title_text="テクニカル分析チャート",
        height=1200,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    price_min = df["基準価額"].min()
    price_max = df["基準価額"].max()
    price_margin = (price_max - price_min) * 0.1 if price_max != price_min else price_max * 0.1
    fig.update_yaxes(title_text="基準価額", range=[price_min - price_margin, price_max + price_margin], row=1, col=1)

    if show_indicators.get("ボリンジャーバンド", False):
        fig.update_yaxes(title_text="ボリンジャーバンド", range=[price_min - price_margin, price_max + price_margin], row=2, col=1)

    if show_indicators.get("RSI", False):
        fig.update_yaxes(title_text="RSI", range=[-5, 105], row=3, col=1)

    if show_indicators.get("MACD", False):
        macd, signal = calculate_macd(df["基準価額"])
        macd_min = min(macd.min(), signal.min())
        macd_max = max(macd.max(), signal.max())
        macd_margin = (macd_max - macd_min) * 0.1 if macd_max != macd_min else abs(macd_max) * 0.1
        fig.update_yaxes(title_text="MACD", range=[macd_min - macd_margin, macd_max + macd_margin], row=4, col=1)

    if show_indicators.get("一目均衡表", False):
        fig.update_yaxes(title_text="一目均衡表", row=ichimoku_row, col=1)
        fig.update_xaxes(title_text="日付", row=ichimoku_row, col=1)
    else:
        fig.update_xaxes(title_text="日付", row=dmi_row, col=1)
    return fig


def is_range_bound_market(df: pd.DataFrame, window: int = 20, threshold: float = 0.05) -> Tuple[float, Dict[str, float]]:
    recent_data = df["基準価額"].tail(window)
    high_price = recent_data.max()
    low_price = recent_data.min()
    mid_price = (high_price + low_price) / 2
    price_range = (high_price - low_price) / mid_price if mid_price else 0

    if "25日移動平均" in df.columns and df["25日移動平均"].dropna().shape[0] >= window:
        ma25 = df["25日移動平均"].tail(window)
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
    middle_band = df["基準価額"].rolling(window=window).mean()
    std_dev = df["基準価額"].rolling(window=window).std()

    upper_band_2 = middle_band + (std_dev * 2)
    lower_band_2 = middle_band - (std_dev * 2)

    current_price = df["基準価額"].iloc[-1]
    prev_price = df["基準価額"].iloc[-2]

    expanding = is_band_expanding(std_dev)
    buy_signal = bool(current_price > upper_band_2.iloc[-1] and prev_price <= upper_band_2.iloc[-2] and expanding)
    sell_signal = bool(current_price < lower_band_2.iloc[-1] and prev_price >= lower_band_2.iloc[-2] and expanding)

    position = "中立"
    if current_price > upper_band_2.iloc[-1]:
        position = "上限突破"
    elif current_price < lower_band_2.iloc[-1]:
        position = "下限突破"

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
        bb_signal = "買いシグナル"
        summary.append("・ボリンジャーバンド: +2σを上抜け、バンド拡大中（買い）")
    elif bb.sell_signal:
        bb_signal = "売りシグナル"
        summary.append("・ボリンジャーバンド: -2σを下抜け、バンド拡大中（売り）")
    else:
        summary.append(f"・ボリンジャーバンド: {bb.position}の状態")

    detailed.append(
        f"""
#### ボリンジャーバンド分析
- **現在の状態**: {bb.position}
- **バンドの状態**: {'拡大傾向' if bb.is_expanding else '収縮傾向'}
- **シグナル**: {bb_signal or '明確なシグナルなし'}
- **判断根拠**:
  - 基準価額: {df['基準価額'].iloc[-1]:,.0f}円
  - 中心線 (20日SMA): {bb.middle_band:,.0f}円
  - +2σ: {bb.upper_band:,.0f}円
  - -2σ: {bb.lower_band:,.0f}円
- **トレード戦略**:
  - バンドが収縮から拡大に転じ、+2σを上抜けた場合は買いシグナル
  - バンドが収縮から拡大に転じ、-2σを下抜けた場合は売りシグナル
  - バンド内での取引は控えめにし、ブレイクアウトを待つ
""",
    )

    range_score, range_info = is_range_bound_market(df)
    if range_score > 0.8:
        range_message = "明確なボックス相場"
    elif range_score > 0.6:
        range_message = "ボックス相場寄りの値動き"
    elif range_score > 0.4:
        range_message = "やや方向感のある値動き"
    else:
        range_message = "一定方向への値動き"

    summary.append(f"・{range_message}：価格帯 {range_info['low_price']:,.0f}円 ～ {range_info['high_price']:,.0f}円")
    detailed.append(
        f"""
#### 価格変動パターン分析
- **{range_message}**を形成しています
- 価格帯: {range_info['low_price']:,.0f}円 ～ {range_info['high_price']:,.0f}円
- 変動幅: {range_info['range_width']:.1f}%
- トレンドの傾き: {range_info['ma_slope']:.1f}%
- レンジ相場度: {range_info['range_score']:.1f}
- 投資戦略: {'上下の価格帯を利用した取引を検討' if range_score > 0.6 else '方向性に沿った取引を推奨'}
""",
    )

    current_ma25 = None
    current_ma200 = None
    ma_status = "データ不足"
    if "25日移動平均" in df.columns and "200日移動平均" in df.columns:
        ma25 = df["25日移動平均"].dropna().iloc[-2:]
        ma200 = df["200日移動平均"].dropna().iloc[-2:]
        if len(ma25) == 2 and len(ma200) == 2:
            current_ma25 = ma25.iloc[-1]
            current_ma200 = ma200.iloc[-1]
            price_vs_ma = df["基準価額"].iloc[-1]

            if ma25.iloc[0] < ma200.iloc[0] and ma25.iloc[1] > ma200.iloc[1]:
                ma_status = "ゴールデンクロス：強気"
                summary.append("・ゴールデンクロス発生：強気")
            elif ma25.iloc[0] > ma200.iloc[0] and ma25.iloc[1] < ma200.iloc[1]:
                ma_status = "デッドクロス：弱気"
                summary.append("・デッドクロス発生：弱気")
            elif current_ma25 > current_ma200:
                ma_status = "上昇トレンド継続"
                summary.append("・25日移動平均線が200日移動平均線の上方：強気")
            else:
                ma_status = "下降トレンド継続"
                summary.append("・25日移動平均線が200日移動平均線の下方：弱気")

            detailed.append(
                f"""
#### 移動平均線分析
- **トレンド状態**: {ma_status}
- 25日移動平均線: {current_ma25:,.0f}円
- 200日移動平均線: {current_ma200:,.0f}円
- 基準価額の位置: 移動平均線から{((price_vs_ma - current_ma25) / current_ma25 * 100):.1f}%
- トレンドの強さ: {'強い' if abs((current_ma25 - current_ma200) / current_ma200) > 0.03 else '弱い'}
""",
            )

    rsi_series = calculate_rsi(df["基準価額"])
    rsi_value = float(rsi_series.iloc[-1])
    if rsi_value > 70:
        rsi_status = "買われすぎ"
        summary.append(f"・RSIは{rsi_value:.1f}で、買われすぎの水準")
    elif rsi_value < 30:
        rsi_status = "売られすぎ"
        summary.append(f"・RSIは{rsi_value:.1f}で、売られすぎの水準")
    else:
        rsi_status = "中立"
        summary.append(f"・RSIは{rsi_value:.1f}で、中立的な水準")

    detailed.append(
        f"""
#### RSI（相対力指数）分析
- 現在のRSI: {rsi_value:.1f}
- **市場状態**: {rsi_status}
- 判断: {'短期的な調整の可能性あり' if rsi_value > 70 else '短期的な反発の可能性あり' if rsi_value < 30 else '適正水準で推移中'}
""",
    )

    macd_series, signal_series = calculate_macd(df["基準価額"])
    latest_macd = macd_series.iloc[-1]
    latest_signal = signal_series.iloc[-1]
    prev_macd = macd_series.iloc[-2]
    prev_signal = signal_series.iloc[-2]

    if latest_macd > latest_signal and prev_macd <= prev_signal:
        macd_status = "買いシグナル"
        summary.append("・MACDがシグナルラインを上向きにクロス（買いシグナル）")
    elif latest_macd < latest_signal and prev_macd >= prev_signal:
        macd_status = "売りシグナル"
        summary.append("・MACDがシグナルラインを下向きにクロス（売りシグナル）")
    elif latest_macd > latest_signal:
        macd_status = "上昇トレンド継続"
        summary.append("・MACDはシグナルラインの上方で推移")
    else:
        macd_status = "下降トレンド継続"
        summary.append("・MACDはシグナルラインの下方で推移")

    detailed.append(
        f"""
#### MACD分析
- **現在の状態**: {macd_status}
- MACD値: {latest_macd:.2f}
- シグナル値: {latest_signal:.2f}
- MACDヒストグラム: {(latest_macd - latest_signal):.2f}
- トレンドの強さ: {'強い' if abs(latest_macd - latest_signal) > abs(prev_macd - prev_signal) else '弱い'}
""",
    )

    plus_di, minus_di, adx = calculate_dmi(df["基準価額"])
    current_plus = plus_di.iloc[-1]
    current_minus = minus_di.iloc[-1]
    prev_plus = plus_di.iloc[-2]
    prev_minus = minus_di.iloc[-2]
    current_adx = adx.iloc[-1]

    if current_plus > current_minus and prev_plus <= prev_minus:
        dmi_status = "買いシグナル"
        summary.append("・DMI: +DIが-DIを下から上抜け（買いシグナル）")
    elif current_plus < current_minus and prev_plus >= prev_minus:
        dmi_status = "売りシグナル"
        summary.append("・DMI: +DIが-DIを上から下抜け（売りシグナル）")
    elif current_plus > current_minus:
        dmi_status = "上昇トレンド"
        summary.append("・DMI: +DIが-DIの上方（上昇トレンド）")
    else:
        dmi_status = "下降トレンド"
        summary.append("・DMI: +DIが-DIの下方（下降トレンド）")

    detailed.append(
        f"""
#### DMI分析
- **現在の状態**: {dmi_status}
- +DI: {current_plus:.1f}
- -DI: {current_minus:.1f}
- ADX: {current_adx:.1f}（トレンドの強さ）
- **トレンドの強さ**: {'強い' if current_adx > 25 else '弱い'}
- **シグナル判定**:
  - +DIが-DIを下から上抜けた場合は買いシグナル
  - +DIが-DIを上から下抜けた場合は売りシグナル
""",
    )

    # 一目均衡表の分析
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df["基準価額"])
    ichimoku_status = "データ不足"
    ichimoku_signal_count = 0

    if len(tenkan.dropna()) >= 2 and len(kijun.dropna()) >= 2:
        current_tenkan = tenkan.dropna().iloc[-1]
        current_kijun = kijun.dropna().iloc[-1]
        prev_tenkan = tenkan.dropna().iloc[-2]
        prev_kijun = kijun.dropna().iloc[-2]
        current_price = df["基準価額"].iloc[-1]

        # 先行スパンから雲の位置を計算
        current_senkou_a = senkou_a.dropna().iloc[-1] if len(senkou_a.dropna()) > 0 else current_price
        current_senkou_b = senkou_b.dropna().iloc[-1] if len(senkou_b.dropna()) > 0 else current_price
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)

        # 転換線と基準線のクロス
        tenkan_kijun_cross = ""
        if current_tenkan > current_kijun and prev_tenkan <= prev_kijun:
            tenkan_kijun_cross = "好転（転換線が基準線を上抜け）"
            summary.append("・一目均衡表: 転換線が基準線を上抜け（好転シグナル）")
            ichimoku_signal_count += 1
        elif current_tenkan < current_kijun and prev_tenkan >= prev_kijun:
            tenkan_kijun_cross = "逆転（転換線が基準線を下抜け）"
            summary.append("・一目均衡表: 転換線が基準線を下抜け（逆転シグナル）")
            ichimoku_signal_count -= 1
        elif current_tenkan > current_kijun:
            tenkan_kijun_cross = "転換線が基準線の上方（強気）"
            ichimoku_signal_count += 0.5
        else:
            tenkan_kijun_cross = "転換線が基準線の下方（弱気）"
            ichimoku_signal_count -= 0.5

        # 雲との位置関係
        cloud_position = ""
        if current_price > cloud_top:
            cloud_position = "基準価額が雲の上方（強気）"
            summary.append("・一目均衡表: 基準価額が雲の上方（強気）")
            ichimoku_signal_count += 1
        elif current_price < cloud_bottom:
            cloud_position = "基準価額が雲の下方（弱気）"
            summary.append("・一目均衡表: 基準価額が雲の下方（弱気）")
            ichimoku_signal_count -= 1
        else:
            cloud_position = "基準価額が雲の中（様子見）"

        # 総合判定
        if ichimoku_signal_count >= 1.5:
            ichimoku_status = "強気"
        elif ichimoku_signal_count <= -1.5:
            ichimoku_status = "弱気"
        else:
            ichimoku_status = "中立"

        detailed.append(
            f"""
#### 一目均衡表分析
- **現在の状態**: {ichimoku_status}
- 転換線: {current_tenkan:,.0f}円
- 基準線: {current_kijun:,.0f}円
- 先行スパン1: {current_senkou_a:,.0f}円
- 先行スパン2: {current_senkou_b:,.0f}円
- 雲の範囲: {cloud_bottom:,.0f}円 ～ {cloud_top:,.0f}円
- **転換線と基準線**: {tenkan_kijun_cross}
- **雲との位置**: {cloud_position}
- **シグナル判定**:
  - 転換線が基準線を上抜け、基準価額が雲の上方、遅行スパンが価格を上抜けで三役好転（強い買いシグナル）
  - 転換線が基準線を下抜け、基準価額が雲の下方、遅行スパンが価格を下抜けで三役逆転（強い売りシグナル）
""",
        )

    buy_factors: List[str] = []
    sell_factors: List[str] = []
    neutral_factors: List[str] = []

    ma_bullish = current_ma25 is not None and current_ma200 is not None and current_ma25 > current_ma200
    if current_ma25 is not None and current_ma200 is not None:
        if ma_bullish:
            if ma_status == "ゴールデンクロス：強気":
                buy_factors.append("25日移動平均線が200日移動平均線をゴールデンクロス")
            else:
                buy_factors.append("25日移動平均線が200日移動平均線の上方で推移")
        else:
            if ma_status == "デッドクロス：弱気":
                sell_factors.append("25日移動平均線が200日移動平均線をデッドクロス")
            else:
                sell_factors.append("25日移動平均線が200日移動平均線の下方で推移")

    if bb.buy_signal:
        buy_factors.append("ボリンジャーバンドが上方ブレイクアウト（バンド拡大中）")
    elif bb.sell_signal:
        sell_factors.append("ボリンジャーバンドが下方ブレイクアウト（バンド拡大中）")
    else:
        neutral_factors.append(f"ボリンジャーバンド: {bb.position}の状態")

    if macd_status == "買いシグナル":
        buy_factors.append("MACDがシグナルラインを上向きにクロス")
    elif macd_status == "売りシグナル":
        sell_factors.append("MACDがシグナルラインを下向きにクロス")
    elif latest_macd > latest_signal:
        buy_factors.append("MACDがシグナルラインの上方で推移")
    else:
        sell_factors.append("MACDがシグナルラインの下方で推移")

    if rsi_value < 30:
        buy_factors.append(f"RSIが{rsi_value:.1f}で売られすぎの水準")
    elif rsi_value > 70:
        sell_factors.append(f"RSIが{rsi_value:.1f}で買われすぎの水準")
    else:
        neutral_factors.append(f"RSI {rsi_value:.1f}で中立的な水準")

    if dmi_status == "買いシグナル":
        buy_factors.append("+DIが-DIを下から上抜け（上昇トレンド開始）")
    elif dmi_status == "売りシグナル":
        sell_factors.append("+DIが-DIを上から下抜け（下降トレンド開始）")
    elif current_plus > current_minus:
        buy_factors.append("+DIが-DIの上方（上昇トレンド継続）")
    else:
        sell_factors.append("+DIが-DIの下方（下降トレンド継続）")

    # 一目均衡表の判定を買い・売り要素に追加
    if ichimoku_status == "強気":
        buy_factors.append("一目均衡表が強気シグナル（転換線・基準線・雲の位置が好転）")
    elif ichimoku_status == "弱気":
        sell_factors.append("一目均衡表が弱気シグナル（転換線・基準線・雲の位置が逆転）")
    elif ichimoku_status == "中立":
        neutral_factors.append("一目均衡表は中立")

    strong_buy = 0.0
    strong_sell = 0.0
    if current_ma25 is not None and current_ma200 is not None:
        if ma_bullish and ma_status in ("ゴールデンクロス：強気", "上昇トレンド継続"):
            strong_buy += 1
        elif not ma_bullish and ma_status in ("デッドクロス：弱気", "下降トレンド継続"):
            strong_sell += 1
    if bb.buy_signal:
        strong_buy += 1
    elif bb.sell_signal:
        strong_sell += 1
    if macd_status == "買いシグナル":
        strong_buy += 1
    elif macd_status == "売りシグナル":
        strong_sell += 1
    if rsi_value < 30:
        strong_buy += 0.5
    elif rsi_value > 70:
        strong_sell += 0.5
    if ichimoku_status == "強気":
        strong_buy += 0.5
    elif ichimoku_status == "弱気":
        strong_sell += 0.5

    if strong_buy >= 2 and strong_buy > strong_sell:
        decision = "買い推奨"
    elif strong_sell >= 2 and strong_sell > strong_buy:
        decision = "売り推奨"
    else:
        decision = "様子見"

    conclusion = """### ■結論
現在の市場環境は**{decision}**と判断されます。

**買い要素**
""".format(decision=decision)

    conclusion += "".join(f"・{factor}\n" for factor in buy_factors) if buy_factors else "・該当なし\n"
    conclusion += "\n**売り要素**\n"
    conclusion += "".join(f"・{factor}\n" for factor in sell_factors) if sell_factors else "・該当なし\n"

    if neutral_factors:
        conclusion += "\n**中立要素**\n"
        conclusion += "".join(f"・{factor}\n" for factor in neutral_factors)

    if decision == "買い推奨":
        reasoning = "重要な指標が複数買いシグナルを示しており、上昇トレンドが優勢です。"
    elif decision == "売り推奨":
        reasoning = "重要な指標が複数売りシグナルを示しており、下降トレンドが優勢です。"
    else:
        reasoning = "シグナルが混在しているため、明確な方向性が出るまで待機を推奨します。"

    conclusion += f"\n**総合判断**: 買い要素{len(buy_factors)}個、売り要素{len(sell_factors)}個ですが、重要指標の分析により{decision}と判断します。\n**判断理由**: {reasoning}"
    detailed.append(conclusion)
    return summary, detailed
