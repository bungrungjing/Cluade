import sys
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

CSV = sys.argv[1]


def RSI(close, n=14):
    s = pd.Series(close)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


class SmaCross(Strategy):
    n1, n2 = 5, 20

    def init(self):
        self.ma1 = self.I(SMA, self.data.Close, self.n1)
        self.ma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position.close()


class RsiSwing(Strategy):
    lo, hi, stop = 35, 60, 0.05

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, 14)

    def next(self):
        if not self.position and self.rsi[-1] < self.lo:
            self.buy(sl=self.data.Close[-1] * (1 - self.stop))
        elif self.position and self.rsi[-1] > self.hi:
            self.position.close()


def show(name, stats):
    keys = ["Start", "End", "# Trades", "Return [%]", "Buy & Hold Return [%]",
            "Max. Drawdown [%]", "Win Rate [%]"]
    print(f"--- {name}")
    for k in keys:
        v = stats[k]
        print(f"{k:24s} {v if isinstance(v, (str, int)) or hasattr(v, 'year') else round(float(v), 2)}")


# 1) 설치 확인용: 패키지에 들어있는 GOOG 샘플(장기 데이터)
show("GOOG sample / SmaCross(5,20)",
     Backtest(GOOG, SmaCross, cash=10_000, commission=0.002, exclusive_orders=True).run())

# 2) 퀄컴 최근 100거래일
df = pd.read_csv(CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
df.columns = [c.capitalize() for c in df.columns]
print(f"\nQCOM rows={len(df)} {df.index[0].date()} ~ {df.index[-1].date()}")
for cls in (SmaCross, RsiSwing):
    st = Backtest(df, cls, cash=10_000, commission=0.002, exclusive_orders=True).run()
    show(f"QCOM / {cls.__name__}", st)
    tr = st["_trades"]
    if len(tr):
        print(tr[["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "ReturnPct"]].round(4).to_string(index=False))
