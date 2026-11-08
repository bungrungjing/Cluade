"""40주선 아래에서 '더 담기' (팔지 않음) — 사전 등록 (2026-11-08, 사용자 "좋은 거 같은데?" 후속, 다중비교 주의)

기본: QQQ 1배 계속 보유. 주봉 종가 < 40주 이평이면 다음 주 비중을 올리고, 위로 돌아오면 1배로 복귀.
  U1 아래일 때 1.5배 / U2 아래일 때 2배 / U3 아래일 때 3배
  비교: A QQQ 보유 / H(오늘 합격) 2배 + 200일선 + 시장 폭 공포 60일 예외 (주봉 근사 아님, 기존 결과 33.0% / −51.9%)
비용: 레버리지 부분 연 (배수−1)%, 비중 변경 시 0.1%
합격: 연복리가 두 구간 모두 A보다 높고 최대 낙폭이 A의 1.5배(약 −53%) 이내
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from edge_hunt import rd, half
from ma40w import wk, st
c = wk(rd("daily/QQQ.csv.gz")); r = c.pct_change().fillna(0); ma = c.rolling(40).mean()
below = (c < ma).shift(1, fill_value=False); ok = ma.shift(1).notna()
L = ["\n## 후속: 40주선 아래에서 더 담기 (팔지 않음)", "| 전략 | 연복리 | 최대낙폭 | 최악 연도 | 2010~17 | 2018~26 | 판정 |", "|---|---|---|---|---|---|---|"]
A = r[ok]; cA, mA = st(A); hA = [st(A[half(A.index) == h])[0] for h in ("2010~17", "2018~26")]
for nm, k in (("A QQQ 보유", 1.0), ("U1 아래 1.5배", 1.5), ("U2 아래 2배", 2.0), ("U3 아래 3배", 3.0)):
    lev = pd.Series(np.where(below, k, 1.0), index=c.index)
    x = (lev * r - (lev - 1) * 0.01 / 52 - lev.diff().abs().fillna(0) * 0.001)[ok]
    cg, md = st(x); hv = [st(x[half(x.index) == h])[0] for h in ("2010~17", "2018~26")]
    yr = (1 + x).groupby(x.index.year).prod() - 1
    v = "—" if k == 1 else ("합격" if all(hv[i] > hA[i] for i in range(2)) and md >= 1.5 * mA else "불합격")
    L.append(f"| {nm} | {cg*100:.1f}% | {md*100:.1f}% | {yr.min()*100:.1f}% | {hv[0]*100:.1f}% | {hv[1]*100:.1f}% | {v} |")
L.append("| (참고) H 2배+200일선+공포 예외 | 33.0% | -51.9% | -38.8% | 35.2% | 31.1% | 합격(일봉) |")
yrs = {}
for nm, k in (("A", 1.0), ("U2", 2.0)):
    lev = pd.Series(np.where(below, k, 1.0), index=c.index); x = (lev * r - (lev - 1) * 0.01 / 52)[ok]
    yrs[nm] = (1 + x).groupby(x.index.year).prod() - 1
L.append("\n연도별 A / U2: " + ", ".join(f"{y} {yrs['A'][y]*100:+.0f}/{yrs['U2'][y]*100:+.0f}" for y in yrs["A"].index))
print("\n".join(L)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ma40w_2026-11-08.md"), "a").write("\n".join(L) + "\n")
