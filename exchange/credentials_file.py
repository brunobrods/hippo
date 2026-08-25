from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Credentials:
    api_key:    str
    api_secret: str


# Shared YAML parsing for every exchange's credentials file. The per-exchange
# modules own only the default path — the file shape is identical.
class YamlCredentialsFile:
    def __init__(self, path: Path) -> None:
        self._path = path

    def credentials(self) -> Credentials:
        with open(self._path, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        if raw is None:
            raise ValueError(f"{self._path} is empty — expected api_key and api_secret")
        return Credentials(api_key=raw["api_key"], api_secret=raw["api_secret"])
