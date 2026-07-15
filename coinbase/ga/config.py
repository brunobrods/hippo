from typing import Any

import yaml


class ConfigFile:
    def __init__(self, path: str) -> None:
        self._path = path

    def raw(self) -> dict[str, Any]:
        with open(self._path) as handle:
            return yaml.safe_load(handle)
