"""정성 단계의 '데이터로 재현 가능한 대리 변수' 검증 — combo_bt 의 ① 초안 매매에 진입일 환경을 붙여 비교

사용법: python context_bt.py <combo_bt_trades.csv> <VIX.csv> <US10Y.csv> <재무 JSON> <일봉 폴더>
 공포탐욕 대체: VIX 수준·VIX 5일 변화·시장 폭(표본 중 50일선 위 비율)
 경제지표 대체: 10년물 금리 20일 변화(bp)
 수익구조(품질): TTM 순이익>0 & TTM FCF>0 (분기말+50일/4분기+80일 뒤부터)
 재현 불가(과거를 모르는 상태로 판정할 수 없음): 12인 위원회, 뉴스·회사뉴스 해석, 사람들의 반응, 미래·혁신 판단
"""
import sys, os, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_bt import ttm_series  # noqa: E402

SPLIT = pd.Timestamp("2025-01-01")
trades, vixf, tnf, fundf, folder = sys.argv[1:6]
r = pd.read_csv(trades, parse_dates=["date"])
r = r[r.test == "① 초안 그대로"].copy()
vix = pd.read_csv(vixf, parse_dates=["timestamp"]).set_index("timestamp").close
tn = pd.read_csv(tnf, parse_dates=["timestamp"]).set_index("timestamp").close
vix_ch5 = vix.pct_change(5) * 100
tn_ch20 = (tn - tn.shift(20)) * 100
a50 = {}
for f in glob.glob(os.path.join(folder, "*.csv")):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    a50[os.path.basename(f)[:-4]] = df.close > df.close.rolling(50).mean()
breadth = pd.DataFrame(a50).mean(axis=1) * 100
for name, s in (("vix", vix), ("vix5", vix_ch5), ("tn20", tn_ch20), ("breadth", breadth)):
    r[name] = s.reindex(r.date, method="ffill").values
fund = json.load(open(fundf))
qual = {}
for sym, rows in fund.items():
    tt = ttm_series(rows)
    qual[sym.split(":")[1]] = tt
def q(row):
    tt = qual.get(row.ticker)
    if tt is None: return np.nan
    x = tt[tt.index <= row.date]
    if x.empty: return np.nan
    return float(x.iloc[-1].ttm_ni > 0 and x.iloc[-1].ttm_fcf > 0)
r["quality"] = r.apply(q, axis=1)
r["win"] = r.ret > 0

def show(model):
    s = r[r.model == model]
    print(f"\n=== {model} (진입 {len(s)}회) 기준: 승률 {s.win.mean()*100:.0f}% / 평균 {s.ret.mean()*100:+.2f}% ===")
    print(f"{'조건':26s} {'전체 n':>6} {'승률':>5} {'평균%':>6} | {'학습 n':>6} {'승률':>5} {'평균%':>6} | {'검증 n':>6} {'승률':>5} {'평균%':>6}")
    rules = [
        ("VIX < 16 (안일)", s.vix < 16), ("VIX 16~22", s.vix.between(16, 22)), ("VIX > 22 (공포)", s.vix > 22),
        ("VIX 5일 상승 > +10%", s.vix5 > 10), ("VIX 5일 하락", s.vix5 < 0),
        ("금리 20일 +20bp 이상 상승", s.tn20 >= 20), ("금리 20일 하락", s.tn20 < 0),
        ("시장 폭 ≤ 40%", s.breadth <= 40), ("시장 폭 ≥ 70%", s.breadth >= 70),
        ("수익구조 양호(순이익·FCF 흑자)", s.quality == 1), ("수익구조 미달", s.quality == 0),
        ("양호 + VIX ≤ 22", (s.quality == 1) & (s.vix <= 22)),
    ]
    for name, m in rules:
        x = s[m.fillna(False)]; a, b = x[x.date < SPLIT], x[x.date >= SPLIT]
        f = lambda z: f"{len(z):6d} {z.win.mean()*100:5.0f} {z.ret.mean()*100:+6.2f}" if len(z) >= 8 else f"{len(z):6d}     -      -"
        print(f"{name:26s} {f(x)} | {f(a)} | {f(b)}")

for m in ("방어형", "수익추종"): show(m)
r.to_csv(os.path.join(folder, "..", "context_bt_trades.csv"), index=False)
