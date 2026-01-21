from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI


def generate_personalized_analysis(technical_data: Dict[str, Any]) -> str:
    try:
        # Try Streamlit secrets first, fallback to environment variable
        api_key = None
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)
        price_info = technical_data.get("price_info", "不明")
        rsi_info = technical_data.get("rsi_info", "不明")
        macd_info = technical_data.get("macd_info", "不明")
        trend = technical_data.get("trend", "不明")
        recommendation = technical_data.get("recommendation", "不明")

        ma25_value = technical_data.get("ma25_value", 0.0)
        ma200_value = technical_data.get("ma200_value", 0.0)
        price_ma25_ratio = technical_data.get("price_ma25_ratio", 0.0)
        price_ma200_ratio = technical_data.get("price_ma200_ratio", 0.0)
        ma_cross_status = technical_data.get("ma_cross_status", "unknown")

        try:
            current_price = float(str(price_info).split(":")[1].strip().replace("円", "").replace(",", ""))
        except (IndexError, ValueError):
            current_price = 0.0

        prompt = f"""
あなたは投資信託の分析に特化した金融アドバイザーです。
以下のテクニカル指標データに基づいて、簡潔で具体的な投資分析と1ヶ月後の価格予測レンジを提供してください。

テクニカル指標の状況：
- 価格動向: {price_info}
- 25日移動平均線: {ma25_value:,.0f}円
- 200日移動平均線: {ma200_value:,.0f}円
- 25日移動平均線との乖離率: {price_ma25_ratio:.1f}%
- 200日移動平均線との乖離率: {price_ma200_ratio:.1f}%
- 移動平均線の状態: {ma_cross_status}
- RSI: {rsi_info}
- MACD: {macd_info}
- 全体的なトレンド: {trend}
- 現在の投資判断: {recommendation}

以下の4点に焦点を当てた分析を提供してください：

1. 移動平均線分析
2. 市場分析
3. 投資判断
4. 1ヶ月後の価格予測レンジ（具体的なレンジと根拠）
"""

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": "投資信託アナリストとして、具体的な数値に基づいた簡潔で実用的な分析を提供してください。予測では根拠を明示してください。",
                },
                {"role": "user", "content": prompt},
            ],
        )

        analysis = response.choices[0].message.content
        if not analysis or not analysis.strip():
            raise ValueError("Empty response")
        return analysis
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).lower()
        if "rate limit" in detail:
            return "⚠️ APIリクエスト制限に達しました。時間をおいて再試行してください。"
        if "timeout" in detail:
            return "⏱️ 通信がタイムアウトしました。再度お試しください。"
        return "❌ AI分析の生成中にエラーが発生しました。テクニカル指標の分析結果をご参照ください。"


def chat_with_ai_analyst(technical_data: Dict[str, Any], user_message: str, chat_history: List[Dict[str, str]] | None = None) -> str:
    try:
        # Try Streamlit secrets first, fallback to environment variable
        api_key = None
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)
        system_context = f"""
あなたは投資信託の分析に特化した金融アドバイザーです。
以下のテクニカル指標データに基づいて、ユーザーの質問に答えてください。

テクニカル指標の状況：
- 価格動向: {technical_data.get('price_info', '不明')}
- RSI: {technical_data.get('rsi_info', '不明')}
- MACD: {technical_data.get('macd_info', '不明')}
- トレンド: {technical_data.get('trend', '不明')}
- 投資判断: {technical_data.get('recommendation', '不明')}
- 移動平均線の状態: {technical_data.get('ma_cross_status', '不明')}
"""

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_context}]
        if chat_history:
            messages.extend(chat_history[-5:])
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
        )

        analysis = response.choices[0].message.content
        if not analysis or not analysis.strip():
            raise ValueError("Empty response")
        return analysis
    except Exception as exc:
        import traceback
        error_detail = f"{str(exc)}\n{traceback.format_exc()}"
        return f"❌ チャット機能でエラーが発生しました: {str(exc)}"
