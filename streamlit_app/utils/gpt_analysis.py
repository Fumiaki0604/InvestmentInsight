from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI


def generate_personalized_analysis(technical_data: Dict[str, Any]) -> str:
    try:
        client = OpenAI()
        price_info = technical_data.get("price_info", "�s��")
        rsi_info = technical_data.get("rsi_info", "�s��")
        macd_info = technical_data.get("macd_info", "�s��")
        trend = technical_data.get("trend", "�s��")
        recommendation = technical_data.get("recommendation", "�s��")

        ma25_value = technical_data.get("ma25_value", 0.0)
        ma200_value = technical_data.get("ma200_value", 0.0)
        price_ma25_ratio = technical_data.get("price_ma25_ratio", 0.0)
        price_ma200_ratio = technical_data.get("price_ma200_ratio", 0.0)
        ma_cross_status = technical_data.get("ma_cross_status", "unknown")

        try:
            current_price = float(str(price_info).split(":")[1].strip().replace("�~", "").replace(",", ""))
        except (IndexError, ValueError):
            current_price = 0.0

        prompt = f"""
���Ȃ��͓����M���̕��͂ɓ����������Z�A�h�o�C�U�[�ł��B
�ȉ��̃e�N�j�J���w�W�f�[�^�Ɋ�Â��āA�Ȍ��ŋ�̓I�ȓ������͂�1������̉��i�\�������W��񋟂��Ă��������B

�e�N�j�J���w�W�̏󋵁F
- ���i����: {price_info}
- 25���ړ����ϐ�: {ma25_value:,.0f}�~
- 200���ړ����ϐ�: {ma200_value:,.0f}�~
- 25���ړ����ϐ��Ƃ̘�����: {price_ma25_ratio:.1f}%
- 200���ړ����ϐ��Ƃ̘�����: {price_ma200_ratio:.1f}%
- �ړ����ϐ��̏��: {ma_cross_status}
- RSI: {rsi_info}
- MACD: {macd_info}
- �S�̓I�ȃg�����h: {trend}
- ���݂̓������f: {recommendation}

�ȉ���4�_�ɏœ_�𓖂Ă����͂�񋟂��Ă��������F

1. �ړ����ϐ�����
2. �s�ꕪ��
3. �������f
4. 1������̉��i�\�������W�i��̓I�ȃ����W�ƍ����j
"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "�����M���A�i���X�g�Ƃ��āA��̓I�Ȑ��l�Ɋ�Â����Ȍ��Ŏ��p�I�ȕ��͂�񋟂��Ă��������B�\���ł͍����𖾎����Ă��������B",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.5,
        )

        analysis = response.choices[0].message.content
        if not analysis or not analysis.strip():
            raise ValueError("Empty response")
        return analysis
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).lower()
        if "rate limit" in detail:
            return "?? API���N�G�X�g�����ɒB���܂����B���Ԃ������čĎ��s���Ă��������B"
        if "timeout" in detail:
            return "?? �ʐM���^�C���A�E�g���܂����B�ēx���������������B"
        return "?? AI���͂̐������ɃG���[���������܂����B�e�N�j�J���w�W�̕��͌��ʂ����Q�Ƃ��������B"


def chat_with_ai_analyst(technical_data: Dict[str, Any], user_message: str, chat_history: List[Dict[str, str]] | None = None) -> str:
    try:
        client = OpenAI()
        system_context = f"""
���Ȃ��͓����M���̕��͂ɓ����������Z�A�h�o�C�U�[�ł��B
�ȉ��̃e�N�j�J���w�W�f�[�^�Ɋ�Â��āA���[�U�[�̎���ɓ����Ă��������B

�e�N�j�J���w�W�̏󋵁F
- ���i����: {technical_data.get('price_info', '�s��')}
- RSI: {technical_data.get('rsi_info', '�s��')}
- MACD: {technical_data.get('macd_info', '�s��')}
- �g�����h: {technical_data.get('trend', '�s��')}
- �������f: {technical_data.get('recommendation', '�s��')}
- �ړ����ϐ��̏��: {technical_data.get('ma_cross_status', '�s��')}
"""

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_context}]
        if chat_history:
            messages.extend(chat_history[-5:])
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        analysis = response.choices[0].message.content
        if not analysis or not analysis.strip():
            raise ValueError("Empty response")
        return analysis
    except Exception:
        return "?? �`���b�g�@�\�ŃG���[���������܂����B���΂炭���Ԃ������čēx���������������B"
