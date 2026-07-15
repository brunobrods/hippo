from coinbase.market_scanner import Snapshot


class MarketData:

    def snapshot(self)->Snapshot:
        pass

class Strategy:

    def __init__(self, market_data):
        self._market_data = market_data


    def onTimer(self):
        pass