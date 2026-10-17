"""백테스트 5 보강 — 약세장(시장 폭) 구간 분리 + 바닥 정의 민감도 + 200일선 회복의 약세장 구간
사용법: python sens_bt.py <일봉 폴더>
 시장 폭 = 표본 종목 중 50일선 위 비율(%). 약세장 = 폭 ≤ 40%, 중립 = 40~60, 강세장 = 60 이상
 모든 변형을 빠짐없이 보고한다(여러 변형을 시험했으므로 우연히 좋은 결과가 나올 수 있음 → 다중비교 주의)"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators
from combo_bt import sim, DEF, TRD
SPLIT = pd.Timestamp("2025-01-01")

def since_low(low, n):
    return low.rolling(n).apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)

data = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    d = indicators(df); d["relvol"] = d.volume/d.volume.rolling(20).mean(); d["sl40"] = since_low(d.low, 40)
    data[os.path.basename(f)[:-4]] = d.iloc[200:]
a50 = pd.DataFrame({t: d.above50 for t, d in data.items()}).mean(axis=1) * 100

rows = []
for t, d in data.items():
    gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
    br = a50.reindex(d.index)
    weak, mid, strong = br <= 40, br.between(40, 60, inclusive="neither"), br >= 60
    dd60 = d.close / d.high.rolling(60).max() - 1
    rng5 = (d.high.rolling(5).max() - d.low.rolling(5).min()) / d.atr
    cross = (d.k > d.d) & (d.k.shift() <= d.d.shift())
    trig = (cross | cross.shift(fill_value=False)) & (d.close > d.open) & (d.relvol >= 1.2)
    below = (d.close < d.sma200).astype(int)
    r1 = gate & (d.close > d.sma200) & (d.close.shift(1) < d.sma200.shift(1)) & (below.rolling(20).sum().shift(1) >= 10)
    v3 = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
    S = {}
    # (1) 시장 구간별: v3, 아무 날, 200일선 회복, 바닥(15%)+트리거, 바닥(10%)+트리거
    base15 = gate & (dd60 <= -0.15) & d.sl40.between(5, 20) & (rng5 <= 3.5)
    base10 = gate & (dd60 <= -0.10) & d.sl40.between(5, 20) & (rng5 <= 3.5)
    for reg, rm in (("약세장", weak), ("중립", mid), ("강세장", strong)):
        S[f"[{reg}] 기준 v3 BUY"] = v3 & rm
        S[f"[{reg}] 아무 날"] = gate & rm
        S[f"[{reg}] 200일선 회복"] = r1 & rm
        S[f"[{reg}] 바닥10%만"] = base10 & rm
        S[f"[{reg}] 바닥10%+트리거"] = base10 & trig & rm
    # (2) 바닥 정의 민감도 (전 구간): 하락폭 × 횡보 일수 × 변동폭
    for dd in (0.10, 0.20):
        for lo_d in (3, 5, 10):
            for rg in (2.5, 3.5, 5.0):
                b = gate & (dd60 <= -dd) & d.sl40.between(lo_d, 25) & (rng5 <= rg)
                S[f"[민감도] 하락{int(dd*100)}% 횡보{lo_d}일+ 폭{rg}ATR 바닥만"] = b
                S[f"[민감도] 하락{int(dd*100)}% 횡보{lo_d}일+ 폭{rg}ATR +트리거"] = b & trig
    for model, kw in (("방어형", DEF), ("수익추종", TRD)):
        for name, m in S.items():
            for tr in sim(d, m, **kw): tr.update(model=model, test=name, ticker=t); rows.append(tr)
r = pd.DataFrame(rows); r["win"] = r.ret > 0
def ci(x):
    bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(500)]
    return f"{np.percentile(bs,2.5)*100:+.1f}~{np.percentile(bs,97.5)*100:+.1f}"
for model in ("방어형", "수익추종"):
    s = r[r.model == model]; res = []
    for name in dict.fromkeys(s.test):
        x = s[s.test == name]; te = x[x.date >= SPLIT]
        res.append(dict(조건=name, 매매=len(x), 승률=x.win.mean()*100, 평균수익률=x.ret.mean()*100,
                        구간=ci(x.ret.values) if len(x) > 10 else "-", 검증n=len(te),
                        검증승률=te.win.mean()*100 if len(te) else np.nan, 검증수익률=te.ret.mean()*100 if len(te) else np.nan))
    out = pd.DataFrame(res).set_index("조건").round(1)
    print(f"\n=== {model} — 시장 구간별 ==="); print(out[~out.index.str.startswith("[민감도]")].to_string())
    sens = out[out.index.str.startswith("[민감도]")]
    print(f"\n=== {model} — 바닥 정의 민감도 ({len(sens)}개 변형) ===")
    print(sens.to_string())
    pos = sens[sens.평균수익률 > 0]
    print(f"평균수익 플러스 변형 {len(pos)}/{len(sens)}, 95% 구간 하한이 0 위인 변형:",
          [i for i in sens.index if sens.loc[i,'구간'] != '-' and float(sens.loc[i,'구간'].split('~')[0]) > 0])
r.to_csv(os.path.join(sys.argv[1], "..", "sens_bt_trades.csv"), index=False)
