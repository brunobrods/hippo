from pathlib import Path

import pytest
import yaml

from binance.credentials_file import DEFAULT_PATH, CredentialsFile
from exchange.credentials_file import Credentials


def test_default_path_is_under_the_home_directory():
    assert DEFAULT_PATH == Path.home() / ".binance" / "credentials.yaml"


def test_binance_and_coinbase_credentials_live_in_separate_files():
    from coinbase.credentials_file import DEFAULT_PATH as COINBASE_PATH

    assert DEFAULT_PATH != COINBASE_PATH


def test_credentials_file_loads_api_key_and_secret(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text(yaml.dump({"api_key": "abc123", "api_secret": "def456"}))

    credentials = CredentialsFile(path).credentials()

    assert credentials == Credentials(api_key="abc123", api_secret="def456")


def test_credentials_file_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        CredentialsFile(tmp_path / "nonexistent.yaml").credentials()


def test_credentials_file_raises_when_a_key_is_missing(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text(yaml.dump({"api_key": "abc123"}))

    with pytest.raises(KeyError):
        CredentialsFile(path).credentials()


def test_credentials_file_raises_a_clear_error_when_empty(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        CredentialsFile(path).credentials()
