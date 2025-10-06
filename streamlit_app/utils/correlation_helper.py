from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .gsheet_helper import get_sheet_data


SPREADSHEET_ID = "1O3nYKIHCrDbjz1yBGrrAnq883Lgotfvvq035tC9wMVM"


def calculate_price_changes(df: pd.DataFrame, period_days: int = 252) -> pd.Series | None:
    if df is None or df.empty or "基準価額" not in df.columns:
        return None

    data = df.copy()
    data["基準価額"] = pd.to_numeric(data["基準価額"], errors="coerce")
    data = data.dropna(subset=["基準価額"])
    if len(data) < 10:
        return None

    data = data.tail(min(period_days, len(data)))
    if "日付" in data.columns:
        data["日付"] = pd.to_datetime(data["日付"], errors="coerce")
        data = data.dropna(subset=["日付"])
        data = data.sort_values("日付").set_index("日付")

    returns = data["基準価額"].pct_change().dropna()
    return returns if len(returns) > 5 else None


def get_correlation_data(sheet_list: Iterable[str], period_days: int = 252) -> Dict[str, pd.Series]:
    correlation_data: Dict[str, pd.Series] = {}
    successful = 0
    failed: List[str] = []

    for sheet_name in sheet_list:
        try:
            df = get_sheet_data(SPREADSHEET_ID, sheet_name)
            if df is None or df.empty or "基準価額" not in df.columns:
                failed.append(f"{sheet_name}: データ不足")
                continue

            price_changes = calculate_price_changes(df, period_days)
            if price_changes is not None:
                correlation_data[sheet_name] = price_changes
                successful += 1
            else:
                failed.append(f"{sheet_name}: 有効な価格変動データなし")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{sheet_name}: {exc}")

    st.write(f"?? データ取得結果: 成功 {successful}銘柄, 失敗 {len(failed)}銘柄")
    if failed:
        with st.expander("?? データ取得エラーの詳細", expanded=False):
            for error in failed:
                st.caption(f"? {error}")
    if correlation_data:
        with st.expander(f"? 成功した銘柄 ({len(correlation_data)}件)", expanded=False):
            for fund_name, series in correlation_data.items():
                st.caption(f"? {fund_name}: {len(series)}日分のリターンデータ")

    return correlation_data


def calculate_correlation_matrix(correlation_data: Dict[str, pd.Series]) -> pd.DataFrame | None:
    if len(correlation_data) < 2:
        return None

    aligned = pd.DataFrame()
    min_length = min(len(series) for series in correlation_data.values())
    if min_length < 5:
        return None

    for fund_name, series in correlation_data.items():
        aligned[fund_name] = series.tail(min_length).values

    matrix = aligned.corr()
    return None if matrix.isna().values.all() else matrix


def create_correlation_heatmap(correlation_matrix: pd.DataFrame, selected_fund: str | None = None) -> go.Figure | None:
    if correlation_matrix is None or correlation_matrix.empty:
        return None

    if selected_fund and selected_fund in correlation_matrix.index:
        annotations = []
        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                value = correlation_matrix.loc[row, col]
                color = "white" if abs(value) < 0.5 else "black"
                annotations.append(dict(x=j, y=i, text=f"{value:.2f}", showarrow=False, font=dict(color=color, size=10)))

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="相関係数", titleside="right"),
            )
        )
        fig.update_layout(title="銘柄間相関関係マトリックス（直近1年間）", width=800, height=600, annotations=annotations)
        fig.update_xaxes(tickangle=45)
        return fig

    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="銘柄", y="銘柄", color="相関係数"),
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="銘柄間相関関係マトリックス（直近1年間）",
    )
    fig.update_layout(width=800, height=600)
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(value) < 0.5 else "black", size=10),
            )
    return fig


def get_fund_correlations(correlation_matrix: pd.DataFrame, selected_fund: str) -> pd.Series | None:
    if correlation_matrix is None or selected_fund not in correlation_matrix.index:
        return None
    correlations = correlation_matrix[selected_fund].drop(selected_fund)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def create_correlation_bar_chart(fund_correlations: pd.Series, selected_fund: str) -> go.Figure | None:
    if fund_correlations is None or fund_correlations.empty:
        return None

    colors: List[str] = []
    for corr in fund_correlations.values:
        if corr > 0.7:
            colors.append("#2E8B57")
        elif corr > 0.3:
            colors.append("#90EE90")
        elif corr > -0.3:
            colors.append("#FFD700")
        elif corr > -0.7:
            colors.append("#FFA07A")
        else:
            colors.append("#DC143C")

    fig = go.Figure(
        data=go.Bar(
            x=fund_correlations.values,
            y=fund_correlations.index,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.3f}" for value in fund_correlations.values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"{selected_fund}との相関関係",
        xaxis_title="相関係数",
        yaxis_title="銘柄",
        height=max(400, len(fund_correlations) * 25),
        showlegend=False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.3, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_vline(x=-0.3, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_vline(x=0.7, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_vline(x=-0.7, line_dash="dot", line_color="red", opacity=0.3)
    return fig


def create_correlation_summary_table(fund_correlations: pd.Series, selected_fund: str) -> pd.DataFrame | None:
    if fund_correlations is None or fund_correlations.empty:
        return None

    summary_rows: List[Dict[str, str]] = []
    for fund_name, correlation in fund_correlations.items():
        if correlation > 0.7:
            strength = "強い正の相関"
            interpretation = f"{selected_fund}が上昇する時、この銘柄も強く上昇する傾向"
        elif correlation > 0.3:
            strength = "中程度の正の相関"
            interpretation = f"{selected_fund}が上昇する時、この銘柄もやや上昇する傾向"
        elif correlation > -0.3:
            strength = "弱い相関"
            interpretation = f"{selected_fund}との関連性は低い"
        elif correlation > -0.7:
            strength = "中程度の負の相関"
            interpretation = f"{selected_fund}が上昇する時、この銘柄はやや下落する傾向"
        else:
            strength = "強い負の相関"
            interpretation = f"{selected_fund}が上昇する時、この銘柄は強く下落する傾向"

        summary_rows.append(
            {
                "銘柄": fund_name,
                "相関係数": f"{correlation:.3f}",
                "相関の強さ": strength,
                "解釈": interpretation,
            }
        )

    return pd.DataFrame(summary_rows)
