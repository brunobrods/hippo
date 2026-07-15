from coinbase.ga.config import ConfigFile


def test_config_file_loads_shipped_yaml():
    raw = ConfigFile("coinbase/ga/config.yaml").raw()
    assert raw["data"]["pair"] == "BTC-USDC"
    assert raw["strategy"]["indicators"]["rsi_period"] == 14
