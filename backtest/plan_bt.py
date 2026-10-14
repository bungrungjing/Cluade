import sys, warnings, importlib.util, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
from backtesting import Backtest, Strategy
spec=importlib.util.spec_from_file_location("s","../screener.py"); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
df=pd.read_csv(sys.argv[1],parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
df.columns=[c.capitalize() for c in df.columns]
X=s.indicators(df)
def col(name): return lambda _: X[name].values
class PlanA(Strategy):  # 지식문서 A안: 추세 + 스토캐스틱 눌림 교차, 손절 max(7%,2ATR), 목표 3R
    def init(self):
        for n in ("adx","pdi","mdi","sma50","k","d","atr","vwap20"): setattr(self,n,self.I(col(n),self.data.Close,name=n))
    def next(self):
        if self.position: return
        i=-1
        if self.adx[i]>25 and self.pdi[i]>self.mdi[i] and self.data.Close[i]>self.sma50[i] and self.k[-2]<=30 and self.k[i]>self.d[i] and self.k[-2]<=self.d[-2]:
            c=self.data.Close[i]; risk=max(0.07*c,2*self.atr[i]); self.buy(sl=c-risk,tp=c+3*risk)
class Breakout(Strategy):  # 이번 NTSK 매수 방식: 20일 고점 돌파 + ADX>30 + VWAP 강세, 손절 -6.3%, 목표 +20.8%
    def init(self):
        for n in ("adx","pdi","mdi","vwap5","vwap20"): setattr(self,n,self.I(col(n),self.data.Close,name=n))
        self.hh=self.I(lambda c: pd.Series(self.data.High).shift(1).rolling(20).max().values,self.data.Close,name="hh20")
    def next(self):
        if self.position: return
        c=self.data.Close[-1]
        if self.adx[-1]>30 and self.pdi[-1]>self.mdi[-1] and c>self.hh[-1] and c>self.vwap5[-1]>self.vwap20[-1]:
            self.buy(sl=c*(1-0.063),tp=c*1.208)
for cls in (PlanA,Breakout):
    st=Backtest(df,cls,cash=10_000,commission=0.002,exclusive_orders=True).run()
    print("---",cls.__name__, {k:(round(float(st[k]),2) if not isinstance(st[k],(str,pd.Timestamp)) else st[k]) for k in ["# Trades","Return [%]","Buy & Hold Return [%]","Max. Drawdown [%]","Win Rate [%]"]})
    tr=st["_trades"]
    if len(tr): print(tr[["EntryTime","ExitTime","EntryPrice","ExitPrice","ReturnPct"]].round(4).to_string(index=False))
