"""바닥 다지기 후 BUY 검증 — 하락 → 5일 이상 횡보(저점 미갱신) → 매수 신호
사용법: python base_bt.py <일봉 폴더>
 공통 조건(바닥): 신호일 종가가 60일 고점 대비 -DD% 이하, 40일 저점이 5~20일 전에 형성되고 이후 저점 미갱신,
                  저점 이후 구간 폭(고−저)이 ATR 배수 이내
 신호: ① v3 BUY(기존, 추세 조건 포함) ② 바닥형 트리거: K가 D 상향 교차(최근 2일) + 양봉 + 거래량 1.2배
       ③ 바닥형 트리거 + 종가가 20일선 회복
 비교: 바닥 조건만(신호 없음), 기존 v3 BUY
 청산: 방어형(3R·+1R→+0.3R·60일) / 수익추종(추적 4ATR·250일) — combo_bt.sim 재사용"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators
from combo_bt import sim, DEF, TRD
SPLIT = pd.Timestamp("2025-01-01")

def since_low(low, n):  # 최근 n일 최저가가 몇 일 전인지
    return low.rolling(n).apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    d = indicators(df); d["relvol"] = d.volume/d.volume.rolling(20).mean()
    d["sl40"] = since_low(d.low, 40)
    d = d.iloc[200:]
    gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
    dd60 = d.close / d.high.rolling(60).max() - 1
    low40 = d.low.rolling(40).min()
    out = {}
    for dd in (0.15, 0.10):
        # 저점 이후 구간 폭: 최근 5일 고-저
        rng5 = (d.high.rolling(5).max() - d.low.rolling(5).min()) / d.atr
        base = gate & (dd60 <= -dd) & d.sl40.between(5, 20) & (rng5 <= 3.5)
        cross = (d.k > d.d) & (d.k.shift() <= d.d.shift())
        trig = (cross | cross.shift(fill_value=False)) & (d.close > d.open) & (d.relvol >= 1.2)
        out[f"하락{int(dd*100)}% 바닥만"] = base
        out[f"하락{int(dd*100)}% 바닥+v3 BUY"] = base & (d.signal == "BUY") & (d.relvol >= 1.2)
        out[f"하락{int(dd*100)}% 바닥+반등트리거"] = base & trig
        out[f"하락{int(dd*100)}% 바닥+트리거+20일선"] = base & trig & (d.close > d.sma20)
    out["기준 v3 BUY"] = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
    out["아무 날(통과조건만)"] = gate
    t = os.path.basename(f)[:-4]
    for model, kw in (("방어형", DEF), ("수익추종", TRD)):
        for name, m in out.items():
            for tr in sim(d, m, **kw): tr.update(model=model, test=name, ticker=t); rows.append(tr)
r = pd.DataFrame(rows); r["win"] = r.ret > 0
def ci(x):
    bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(1000)]
    return f"{np.percentile(bs,2.5)*100:+.1f}~{np.percentile(bs,97.5)*100:+.1f}"
for model in ("방어형", "수익추종"):
    s = r[r.model == model]; res = []
    for name in dict.fromkeys(s.test):
        x = s[s.test == name]; tr, te = x[x.date < SPLIT], x[x.date >= SPLIT]
        res.append(dict(조건=name, 매매=len(x), 종목=x.ticker.nunique(), 승률=x.win.mean()*100, 평균수익률=x.ret.mean()*100, 평균R=x.R.mean(),
                        구간=ci(x.ret.values) if len(x) > 10 else "-", 학습승률=tr.win.mean()*100 if len(tr) else np.nan,
                        검증승률=te.win.mean()*100 if len(te) else np.nan, 검증수익률=te.ret.mean()*100 if len(te) else np.nan, 검증n=len(te)))
    print(f"\n=== {model} ==="); print(pd.DataFrame(res).set_index("조건").round(1).to_string())
r.to_csv(os.path.join(sys.argv[1], "..", "base_bt_trades.csv"), index=False)
