"""200일선 이탈 후 회복 매수 검증 — 종가가 200일선 아래였다가 위로 올라서는 날 진입
사용법: python reclaim_bt.py <일봉 폴더>
 신호 R1: 어제 종가 < 200일선, 오늘 종가 > 200일선 (상향 돌파) 이면서 최근 20일 중 10일 이상 200일선 아래
 신호 R2: R1 + 신호일 거래량 ≥ 20일 평균 1.5배
 신호 R3: R1 + 200일선 기울기 ≥ 0 (장기 추세가 꺾이지 않은 이탈)
 비교: v3 BUY(기존), 통과 조건만 만족한 아무 날
 청산: 방어형(3R·+1R→+0.3R·60일) / 수익추종(추적 4ATR·250일)  (combo_bt.sim 재사용)"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators
from combo_bt import sim, DEF, TRD
SPLIT = pd.Timestamp("2025-01-01")
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    d = indicators(df); d["relvol"] = d.volume/d.volume.rolling(20).mean(); d = d.iloc[200:]
    gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
    below = (d.close < d.sma200).astype(int)
    r1 = gate & (d.close > d.sma200) & (d.close.shift(1) < d.sma200.shift(1)) & (below.rolling(20).sum().shift(1) >= 10)
    r2 = r1 & (d.relvol >= 1.5)
    r3 = r1 & (d.sma200 >= d.sma200.shift(10))
    v3 = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
    t = os.path.basename(f)[:-4]
    sets = {"R1 200일선 상향 회복": r1, "R2 R1+거래량1.5배": r2, "R3 R1+200일선 기울기≥0": r3, "기준 v3 BUY": v3, "아무 날(통과조건만)": gate}
    for model, kw in (("방어형", DEF), ("수익추종", TRD)):
        for name, m in sets.items():
            for tr in sim(d, m, **kw): tr.update(model=model, test=name, ticker=t); rows.append(tr)
r = pd.DataFrame(rows); r["win"] = r.ret > 0
def ci(x):
    bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(1000)]
    return f"{np.percentile(bs,2.5)*100:+.1f}~{np.percentile(bs,97.5)*100:+.1f}"
for model in ("방어형", "수익추종"):
    s = r[r.model == model]; out = []
    for name in dict.fromkeys(s.test):
        x = s[s.test == name]; tr, te = x[x.date < SPLIT], x[x.date >= SPLIT]
        out.append(dict(조건=name, 매매=len(x), 종목=x.ticker.nunique(), 승률=x.win.mean()*100, 평균수익률=x.ret.mean()*100, 평균R=x.R.mean(),
                        구간=ci(x.ret.values) if len(x) > 10 else "-", 학습승률=tr.win.mean()*100, 검증승률=te.win.mean()*100,
                        검증수익률=te.ret.mean()*100, 검증n=len(te)))
    print(f"\n=== {model} ==="); print(pd.DataFrame(out).set_index("조건").round(1).to_string())
r.to_csv(os.path.join(sys.argv[1], "..", "reclaim_bt_trades.csv"), index=False)
