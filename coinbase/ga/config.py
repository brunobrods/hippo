from pathlib import Path
from typing import Any

import yaml

# Shared default root for the GA module's cache/results, outside the repo so
# every git worktree of this checkout uses the same location instead of each
# needing its own — same reasoning as CredentialsFile's fixed home-dir path.
GA_RESULTS_ROOT = Path.home() / ".coinbase" / "ga"


class ConfigFile:
    def __init__(self, path: str) -> None:
        self._path = path

    def raw(self) -> dict[str, Any]:
        with open(self._path) as handle:
            return yaml.safe_load(handle)
