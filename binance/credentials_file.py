from pathlib import Path

from exchange.credentials_file import Credentials, YamlCredentialsFile

# Fixed, outside any git checkout/worktree — every worktree of this repo shares
# one real credentials file. Mirrors coinbase.credentials_file.DEFAULT_PATH.
DEFAULT_PATH = Path.home() / ".binance" / "credentials.yaml"


class CredentialsFile:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path

    def credentials(self) -> Credentials:
        return YamlCredentialsFile(self._path).credentials()
