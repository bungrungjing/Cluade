"""손절·추적 ATR 배수 비교 — v3 BUY 진입 고정, 초기 손절 max(7%, ks×ATR), +1R에서 본전, 이후 최고가 − kt×ATR 추적
사용법: python atr_mult_bt.py <일봉 폴더>   (수익률 기준 비교: 손절폭이 달라 R은 직접 비교 불가)"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators
FEE, SPLIT = 0.002, pd.Timestamp("2025-01-01")

def sim(d, mask, ks, kt, maxhold=250):
    out, nxt, n = [], 0, len(d)
    for i in np.where(mask.values)[0]:
        if i < nxt or i >= n - 1: continue
        e = d.close.iloc[i]; risk = max(0.07 * e, ks * d.atr.iloc[i])
        stop, hh, moved, px, why, j = e - risk, e, False, None, "기간", i
        for j in range(i + 1, min(n, i + 1 + maxhold)):
            if d.low.iloc[j] <= stop:
                px, why = min(stop, d.open.iloc[j]), "손절" if not moved else "추적"; break
            hh = max(hh, d.high.iloc[j])
            if not moved and hh >= e + risk: stop, moved = max(stop, e), True
            if moved: stop = max(stop, hh - kt * d.atr.iloc[j])
        if px is None: px = d.close.iloc[j]
        out.append(dict(date=d.index[i], ret=px / e - 1 - FEE, hold=j - i, why=why, risk=risk / e)); nxt = j + 1
    return out

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    d = indicators(df); d["relvol"] = d.volume / d.volume.rolling(20).mean(); d = d.iloc[200:]
    buy = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40) & (d.signal == "BUY") & (d.relvol >= 1.2)
    for ks, kt in [(2, 4), (2.1, 4.2), (2.5, 4), (3, 4), (3, 5), (4, 5), (4, 6)]:
        for t in sim(d, buy, ks, kt): t.update(ks=ks, kt=kt); rows.append(t)
r = pd.DataFrame(rows)
def agg(x):
    g, l = x.ret[x.ret > 0].sum(), -x.ret[x.ret < 0].sum()
    return pd.Series(dict(매매=len(x), 손절률=(x.why == "손절").mean() * 100, 승률=(x.ret > 0).mean() * 100,
        평균수익률=x.ret.mean() * 100, 평균손절폭=x.risk.mean() * 100, 최악=x.ret.min() * 100, PF=g / l if l else np.nan, 보유=x.hold.mean()))
for lab, sub in (("전 기간", r), ("학습 ~2024", r[r.date < SPLIT]), ("검증 2025~", r[r.date >= SPLIT])):
    print(f"\n=== {lab} ==="); print(sub.groupby(["ks", "kt"]).apply(agg).round(2).to_string())
