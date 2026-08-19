from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Fixed, outside any git checkout/worktree — every worktree of this repo
# shares one real credentials file instead of each needing its own gitignored
# coinbase/credentials.py copy.
DEFAULT_PATH = Path.home() / ".coinbase" / "credentials.yaml"


@dataclass(frozen=True)
class Credentials:
    api_key:    str
    api_secret: str


class CredentialsFile:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path

    def credentials(self) -> Credentials:
        with open(self._path, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        if raw is None:
            raise ValueError(f"{self._path} is empty — expected api_key and api_secret")
        return Credentials(api_key=raw["api_key"], api_secret=raw["api_secret"])
