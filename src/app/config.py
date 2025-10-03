import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_service_account_key: str | None = Field(default=None, alias="GOOGLE_SERVICE_ACCOUNT_KEY")
    google_spreadsheet_id: str | None = Field(default=None, alias="GOOGLE_SPREADSHEET_ID")
    default_local_data_path: Path = Field(default=Path("data/sample_nav.csv"), alias="DEFAULT_LOCAL_DATA_PATH")

    data_cache_ttl_seconds: int = Field(default=6 * 60 * 60)
    enable_debug_logging: bool = Field(default=False, alias="ENABLE_DEBUG_LOGGING")

    def service_account_json(self) -> Optional[dict[str, str]]:
        """Return parsed service account JSON if available."""
        if not self.google_service_account_key:
            return None

        return json.loads(self.google_service_account_key)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()  # type: ignore[arg-type]
