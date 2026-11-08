"""전략 실험실 2차 — 1차 결과를 본 뒤 만든 조합 (사전 등록, 2026-11-08, 다중비교 주의: 1차 14개 + 2차 8개 = 22개 시험)

1차에서 보인 것: 수익을 올린 건 거의 다 레버리지였고 샤프(위험 대비 수익)는 모두 0.9 안팎 → '위험 대비 수익'을 실제로 높이는 조합을 찾는다.
구성 요소(1차에서 정의 그대로): 200일선 필터, 시장 폭 공포 60일 예외(F), RSI(2) 되돌림 예외(R), 시장 폭 급반등 120일(T), P2 모멘텀 10종목+SPY 필터
  R1 2배 + 200일선 + (F 또는 R) 예외
  R2 2배 + 200일선 + (F 또는 T) 예외
  R3 2배 + 200일선 + (F 또는 R 또는 T) 예외
  R4 R3인데 예외 구간에는 3배 (평소 200일선 위 2배)
  R5 변동성 목표: QQQ 20일 변동성으로 배수 = 35% / 연환산 변동성 (최대 3배), 200일선 아래면 0, 단 F·R·T 예외 구간은 200일선 무시
  R6 70% R3 + 30% P2
  R7 50% R3 + 50% P2
  R8 50% R4 + 50% P2
합격 A: 설계(2011~18)·검증(2019~26) 연복리 모두 QQQ보다 높고 최대 낙폭 ≥ −52.7%
합격 S(위험 대비 개선): 합격 A + 설계·검증 두 구간 모두 샤프가 QQQ와 QQQ 2배 보유보다 높음
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, subprocess
from strat_lab import rd, per, stat_daily, REPO
names = [n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
qqq = rd("daily/QQQ.csv.gz"); idx = qqq.index; q = qqq.close; qr = q.pct_change().fillna(0)
ab = {}
for p in names:
    t = os.path.basename(p).split(".")[0]
    if t in ("SPY","QQQ","_VIX"): continue
    try: d = rd(p)
    except Exception: continue
    if len(d) < 300: continue
    s = d.close.rolling(50).mean(); ab[t] = (d.close > s).where(s.notna())
a = pd.DataFrame(ab).reindex(idx); br = a.mean(axis=1).where(a.count(axis=1) >= 200)
lev = lambda k: k * qr - (k - 1) * 0.01 / 252
up1 = (q > q.rolling(200).mean()).shift(1, fill_value=False)
def window(ev, n):
    on = pd.Series(False, index=idx)
    for i in np.where(ev.shift(1, fill_value=False).values)[0]: on.iloc[i:i + n] = True
    return on
f30 = br <= 0.30; F = window(f30 & ~f30.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool), 60)
th = (br >= 0.60) & (br.rolling(20, min_periods=1).min() <= 0.30)
T = window(th & ~th.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool), 120)
d_ = q.diff(); u = d_.clip(lower=0).ewm(alpha=.5, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=.5, adjust=False).mean()
rsi2 = 100 - 100 / (1 + u / dn); sma5 = q.rolling(5).mean()
st_ = np.zeros(len(idx), bool); cur = False
for i in range(len(idx)):
    if not cur and rsi2.iloc[i] < 10: cur = True
    elif cur and q.iloc[i] > sma5.iloc[i]: cur = False
    st_[i] = cur
Rr = pd.Series(st_, index=idx).shift(1, fill_value=False)
lab = pd.read_csv(os.environ["LAB_OUT"], index_col=0, parse_dates=True).reindex(idx).fillna(0)
P2 = lab["P2 P1 + SPY 200일선"]
R = {"QQQ 보유": qr, "QQQ 2배 보유": lev(2), "H(1차 기준)": lev(2).where(up1 | F, 0.0), "C1(1차) 50% H + 50% P2": lab["C1 50% H + 50% P2"]}
R["R1 2배+200일선+(공포|RSI2)"] = lev(2).where(up1 | F | Rr, 0.0)
R["R2 2배+200일선+(공포|급반등)"] = lev(2).where(up1 | F | T, 0.0)
ex = F | Rr | T
R["R3 2배+200일선+(공포|RSI2|급반등)"] = lev(2).where(up1 | ex, 0.0)
R["R4 R3, 예외 구간 3배"] = lev(3).where(ex, lev(2).where(up1, 0.0))
rv = qr.rolling(20).std() * np.sqrt(252)
k = (0.35 / rv).clip(upper=3).shift(1).fillna(1)
R["R5 변동성 목표 35% + 필터 + 예외"] = (k * qr - (k - 1).clip(lower=0) * 0.01 / 252).where(up1 | ex, 0.0)
R["R6 70% R3 + 30% P2"] = 0.7 * R["R3 2배+200일선+(공포|RSI2|급반등)"] + 0.3 * P2
R["R7 50% R3 + 50% P2"] = 0.5 * R["R3 2배+200일선+(공포|RSI2|급반등)"] + 0.5 * P2
R["R8 50% R4 + 50% P2"] = 0.5 * R["R4 R3, 예외 구간 3배"] + 0.5 * P2
st = idx >= "2011-01-01"; res = {}
for name, r in R.items():
    r = r[st]; cg, md, wy, sh = stat_daily(r)
    pv = {p: stat_daily(r[per(r.index) == p]) for p in ("설계", "검증")}
    res[name] = (cg, md, wy, sh, pv)
Q, Q2 = res["QQQ 보유"], res["QQQ 2배 보유"]
L = ["\n# 전략 실험실 2차 — 조합 (사전 등록 8개, 누적 22개 시험)\n", "| 전략 | 연복리 | 최대낙폭 | 최악 연도 | 샤프 | 설계 연복리/샤프 | 검증 연복리/샤프 | 합격A | 합격S |", "|---|---|---|---|---|---|---|---|---|"]
for name, (cg, md, wy, sh, pv) in res.items():
    base = name in ("QQQ 보유", "QQQ 2배 보유") or "1차" in name
    pa = all(pv[p][0] > Q[4][p][0] for p in ("설계", "검증")) and md >= 1.5 * Q[1]
    ps = pa and all(pv[p][3] > max(Q[4][p][3], Q2[4][p][3]) for p in ("설계", "검증"))
    L.append(f"| {name} | {cg*100:.1f}% | {md*100:.1f}% | {wy*100:.1f}% | {sh:.2f} | {pv['설계'][0]*100:.1f}% / {pv['설계'][3]:.2f} | {pv['검증'][0]*100:.1f}% / {pv['검증'][3]:.2f} | {'기준' if base else ('합격' if pa else '—')} | {'기준' if base else ('합격' if ps else '—')} |")
keys = [k for k in R if k.startswith(("QQQ", "R3", "R4", "R7", "R8"))]
yr = pd.DataFrame({k: (1 + R[k][st]).groupby(idx[st].year).prod() - 1 for k in keys}) * 100
L += ["\n연도별(%)", yr.round(0).astype(int).T.to_markdown()]
L.append(f"\n예외 구간 비율: 공포 {F[st].mean()*100:.0f}% / RSI2 {Rr[st].mean()*100:.0f}% / 급반등 {T[st].mean()*100:.0f}% / 합집합 {ex[st].mean()*100:.0f}% / 200일선 위 {up1[st].mean()*100:.0f}%")
now = idx[-1]; L.append(f"현재({now.date()}): 200일선 {'위' if q.iloc[-1] > q.rolling(200).mean().iloc[-1] else '아래'}, 공포 예외 {bool(F.iloc[-1])}, RSI2 {rsi2.iloc[-1]:.1f}(예외 {bool(Rr.iloc[-1])}), 급반등 {bool(T.iloc[-1])}, 시장 폭 {br.iloc[-1]*100:.1f}%")
txt = "\n".join(L); print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "strat_lab_2026-11-08.md"), "a").write(txt + "\n")
