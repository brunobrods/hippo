from typing import Any

from exchange.adapter import ExchangeAdapter

DEFAULT_EXCHANGE = "coinbase"


# Builds the adapter named by `data.exchange` in config.yaml, already loaded
# with that exchange's credentials. The caller still owns its lifecycle:
#
#     async with ConfiguredExchange(raw_config).adapter() as adapter:
#
# The concrete adapters are imported inside the method on purpose — they each
# import exchange.adapter, so importing them at module scope here would close
# an import cycle, and it keeps a Binance-only run from loading Coinbase's JWT
# dependency (and vice versa).
class ConfiguredExchange:
    def __init__(self, raw_config: dict[str, Any]) -> None:
        self._raw_config = raw_config

    def name(self) -> str:
        return str(self._raw_config.get("data", {}).get("exchange", DEFAULT_EXCHANGE)).lower()

    def adapter(self) -> ExchangeAdapter:
        name = self.name()
        if name == "coinbase":
            from coinbase.coinbase_adapter import CoinbaseAdapter
            from coinbase.credentials_file import CredentialsFile
            credentials = CredentialsFile().credentials()
            return CoinbaseAdapter(credentials.api_key, credentials.api_secret)
        if name == "binance":
            from binance.binance_adapter import BinanceAdapter
            from binance.credentials_file import CredentialsFile
            credentials = CredentialsFile().credentials()
            return BinanceAdapter(credentials.api_key, credentials.api_secret)
        raise ValueError(f"Unknown exchange {name!r} — expected 'coinbase' or 'binance'")
