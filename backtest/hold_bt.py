"""장투(고정 보유) 검증 — v3 BUY 진입, 손절 없음/넓은 손절/200일선 이탈 청산 vs 같은 종목 아무 날 매수
사용법: python hold_bt.py <일봉 폴더>"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators
FEE, SPLIT = 0.002, pd.Timestamp("2025-01-01")

def run(d, mask, H, mode):
    out, nxt, n = [], 0, len(d)
    for i in np.where(mask.values)[0]:
        if i < nxt or i + H >= n: continue   # 보유기간 못 채우면 제외
        e = d.close.iloc[i]; px = None
        for j in range(i+1, i+H+1):
            if mode == "wide25" and d.low.iloc[j] <= e*0.75: px = min(e*0.75, d.open.iloc[j]); break
            if mode == "sma200" and d.close.iloc[j] < d.sma200.iloc[j]: px = d.close.iloc[j]; break
        if px is None: px = d.close.iloc[i+H]
        out.append(dict(date=d.index[i], ret=px/e-1-FEE, maxdd=d.low.iloc[i+1:i+H+1].min()/e-1))
        nxt = i + H + 1
    return out

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    d = indicators(df); d["relvol"] = d.volume/d.volume.rolling(20).mean(); d = d.iloc[200:]
    gate = (d.close >= 5) & d.atr_pct.between(1,4) & (d.dvol20 >= 40)
    v3 = gate & (d.signal=="BUY") & (d.relvol >= 1.2)
    t = os.path.basename(f)[:-4]
    for H in (60,120,250):
        for mode in ("무손절","wide25","sma200"):
            for name, m in (("v3", v3), ("아무날", gate)):
                for tr in run(d, m, H, mode): tr.update(H=H, mode=mode, set=name, ticker=t); rows.append(tr)
r = pd.DataFrame(rows); r["win"] = r.ret > 0
print("종목", r.ticker.nunique())
g = r.groupby(["H","mode","set"]).agg(매매=("ret","size"), 승률=("win","mean"), 평균=("ret","mean"), 중앙=("ret","median"),
    최악=("ret","min"), 평균최대낙폭=("maxdd","mean"))
for c in ("승률","평균","중앙","최악","평균최대낙폭"): g[c] *= 100
print(g.round(1).to_string())
print("\n검증구간(2025~) v3")
te = r[(r.date>=SPLIT)&(r.set=="v3")].groupby(["H","mode"]).agg(매매=("ret","size"), 승률=("win","mean"), 평균=("ret","mean"))
te["승률"]*=100; te["평균"]*=100; print(te.round(1).to_string())
