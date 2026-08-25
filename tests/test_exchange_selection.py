import pytest

from binance.binance_adapter import BinanceAdapter
from coinbase.coinbase_adapter import CoinbaseAdapter
from exchange.credentials_file import Credentials
from exchange.selection import ConfiguredExchange


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch):
    # ConfiguredExchange reads the real ~/.coinbase and ~/.binance files; these
    # tests are about which adapter it picks, not about credential loading.
    import binance.credentials_file as binance_credentials
    import coinbase.credentials_file as coinbase_credentials

    for module in (binance_credentials, coinbase_credentials):
        monkeypatch.setattr(
            module.CredentialsFile,
            "credentials",
            lambda self: Credentials(api_key="key", api_secret="secret"),
        )


def test_coinbase_is_the_default_when_no_exchange_is_configured():
    assert ConfiguredExchange({"data": {}}).name() == "coinbase"


def test_an_absent_data_section_still_defaults_to_coinbase():
    assert ConfiguredExchange({}).name() == "coinbase"


def test_the_configured_name_is_case_insensitive():
    assert ConfiguredExchange({"data": {"exchange": "Binance"}}).name() == "binance"


def test_coinbase_config_builds_a_coinbase_adapter():
    adapter = ConfiguredExchange({"data": {"exchange": "coinbase"}}).adapter()

    assert isinstance(adapter, CoinbaseAdapter)


def test_binance_config_builds_a_binance_adapter():
    adapter = ConfiguredExchange({"data": {"exchange": "binance"}}).adapter()

    assert isinstance(adapter, BinanceAdapter)


def test_an_unknown_exchange_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="Unknown exchange"):
        ConfiguredExchange({"data": {"exchange": "kraken"}}).adapter()


def test_each_adapter_reports_its_own_candle_ceiling():
    coinbase = ConfiguredExchange({"data": {"exchange": "coinbase"}}).adapter()
    binance  = ConfiguredExchange({"data": {"exchange": "binance"}}).adapter()

    assert coinbase.max_candles_per_request() == 300
    assert binance.max_candles_per_request() == 1000


def test_each_adapter_names_itself_so_cached_candles_never_cross_venues():
    coinbase = ConfiguredExchange({"data": {"exchange": "coinbase"}}).adapter()
    binance  = ConfiguredExchange({"data": {"exchange": "binance"}}).adapter()

    assert (coinbase.name(), binance.name()) == ("coinbase", "binance")


def test_both_adapters_drive_as_async_context_managers():
    # Every entry point does `async with ConfiguredExchange(...).adapter()`.
    for raw in ({"data": {"exchange": "coinbase"}}, {"data": {"exchange": "binance"}}):
        adapter = ConfiguredExchange(raw).adapter()
        assert hasattr(adapter, "__aenter__")
        assert hasattr(adapter, "__aexit__")
