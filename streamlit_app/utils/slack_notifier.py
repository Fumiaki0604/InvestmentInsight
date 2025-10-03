from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import requests


class SlackNotifier:
    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send_status_change_notification(
        self,
        fund_name: str,
        old_status: str,
        new_status: str,
        price: str,
        price_change: str,
    ) -> bool:
        if not self.is_configured() or old_status != "�l�q��" or new_status != "��������":
            return False

        message = {
            "text": "?? ���������ύX�ʒm",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*������*: {fund_name}\n*�ω�*: {old_status} �� {new_status}\n*����z*: {price}\n*�O����*: {price_change}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"�ʒm����: {datetime.now().strftime('%Y/%m/%d %H:%M')}",
                        }
                    ],
                },
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=message, timeout=10)
            return response.status_code == 200
        except Exception as exc:  # noqa: BLE001
            print(f"Slack notification failed: {exc}")
            return False

    def send_multiple_notifications(self, changes: List[Dict[str, str]]) -> int:
        count = 0
        for change in changes:
            if self.send_status_change_notification(
                change["fund_name"],
                change["old_status"],
                change["new_status"],
                change["price"],
                change["price_change"],
            ):
                count += 1
        return count


def check_status_changes(previous_data: Dict[str, Dict[str, str]], current_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    changes: List[Dict[str, str]] = []
    for fund in current_data:
        name = fund["������"]
        current_status = fund["�X�e�[�^�X"]
        previous_status = previous_data.get(name, {}).get("�X�e�[�^�X")
        if previous_status == "�l�q��" and current_status == "��������":
            changes.append(
                {
                    "fund_name": name,
                    "old_status": previous_status,
                    "new_status": current_status,
                    "price": fund["����z"],
                    "price_change": fund["�O����"],
                }
            )
    return changes


def save_fund_status(fund_data: List[Dict[str, str]], filepath: str = "previous_fund_status.json") -> bool:
    try:
        status_dict = {
            fund["������"]: {
                "�X�e�[�^�X": fund["�X�e�[�^�X"],
                "����z": fund["����z"],
                "�O����": fund["�O����"],
                "timestamp": datetime.now().isoformat(),
            }
            for fund in fund_data
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(status_dict, file, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to save fund status: {exc}")
        return False


def load_previous_status(filepath: str = "previous_fund_status.json") -> Dict[str, Dict[str, str]]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
