"""40주 이동평균선 아래 매수 / 위 매도 — 사전 등록 (2026-11-08, 사용자 제안)

규칙(주봉, 금요일 종가 기준, 다음 주 첫 거래일 시가 대신 금요일 종가로 근사):
  역추세 R: 주봉 종가 < 40주 이평 → 보유(다음 주), 종가 > 40주 이평 → 현금
  비교 T(추세): 반대로 위에서 보유, 아래에서 현금 / 비교 BH: 계속 보유
대상: QQQ, SPY, QQQ 2배·3배(연 2%·3% 비용 근사), 개별 종목(관문 통과 1,087종목 각각 같은 규칙, 결과 평균)
합격(QQQ·SPY 각각): R의 연복리가 두 구간(2010~17 / 2018~26) 모두 BH보다 높거나,
  또는 '보유 중 연환산 수익률'이 두 구간 모두 BH보다 높고 최대 낙폭이 BH 이내
개별 종목 합격: R 평균 연복리가 BH 평균보다 두 구간 모두 높음
비용: 매매당 0.1%(편도)
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from edge_hunt import rd, half, REPO
def wk(d): return d.close.resample("W-FRI").last().dropna()
def run(c, lev=1, cost=0.0):
    r = c.pct_change().fillna(0); ma = c.rolling(40).mean()
    below = (c < ma).shift(1, fill_value=False) & ma.shift(1).notna()
    above = (c > ma).shift(1, fill_value=False) & ma.shift(1).notna()
    lr = lev * r - cost / 52
    out = {}
    for k, pos in (("BH", ma.shift(1).notna()), ("R 아래 보유", below), ("T 위 보유", above)):
        sw = pos.astype(int).diff().abs().fillna(0)
        out[k] = lr.where(pos, 0.0) - sw * 0.001
    return {k: v[c.index >= c.index[40]] for k, v in out.items()}, below
def st(r, inpos=None):
    eq = (1 + r).cumprod(); y = len(r) / 52
    cg = eq.iloc[-1] ** (1 / y) - 1 if eq.iloc[-1] > 0 else -1
    md = (eq / eq.cummax() - 1).min()
    return cg, md
def main():
    L = ["# 40주 이동평균선 아래 매수 / 위 매도 (2026-11-08)\n", "| 대상 | 규칙 | 연복리 | 최대낙폭 | 2010~17 | 2018~26 | 보유 비율 | 보유 중 연환산 |", "|---|---|---|---|---|---|---|---|"]
    for nm, d, lev, cost in (("QQQ", rd("daily/QQQ.csv.gz"), 1, 0), ("SPY", rd("daily/SPY.csv.gz"), 1, 0), ("QQQ 2배", rd("daily/QQQ.csv.gz"), 2, .02), ("QQQ 3배", rd("daily/QQQ.csv.gz"), 3, .03)):
        c = wk(d); res, below = run(c, lev, cost)
        for k, r in res.items():
            cg, md = st(r); hv = [st(r[half(r.index) == h])[0] for h in ("2010~17", "2018~26")]
            held = (r != 0).mean()
            ann = (1 + r[r != 0]).prod() ** (52 / max((r != 0).sum(), 1)) - 1
            L.append(f"| {nm} | {k} | {cg*100:.1f}% | {md*100:.1f}% | {hv[0]*100:.1f}% | {hv[1]*100:.1f}% | {held*100:.0f}% | {ann*100:.1f}% |")
    # 개별 종목
    names = [n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
    agg = {k: {"2010~17": [], "2018~26": [], "전체": []} for k in ("BH", "R 아래 보유", "T 위 보유")}
    for p in names:
        t = os.path.basename(p).split(".")[0]
        if t in ("SPY", "QQQ", "_VIX"): continue
        try: d = rd(p)
        except Exception: continue
        c = wk(d)
        if len(c) < 200 or c.iloc[-1] < 5: continue
        res, _ = run(c)
        for k, r in res.items():
            agg[k]["전체"].append(st(r)[0])
            for h in ("2010~17", "2018~26"):
                x = r[half(r.index) == h]
                if len(x) > 150: agg[k][h].append(st(x)[0])
    L += ["\n| 개별 종목 (각 종목 같은 규칙, 연복리 중앙값) | 전체 | 2010~17 | 2018~26 | 종목 수 |", "|---|---|---|---|---|"]
    for k, v in agg.items():
        L.append(f"| {k} | {np.median(v['전체'])*100:.1f}% | {np.median(v['2010~17'])*100:.1f}% | {np.median(v['2018~26'])*100:.1f}% | {len(v['전체'])} |")
    q = wk(rd("daily/QQQ.csv.gz")); L.append(f"\n현재 QQQ 주봉 {q.iloc[-1]:.2f} / 40주 이평 {q.rolling(40).mean().iloc[-1]:.2f}")
    print("\n".join(L)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ma40w_2026-11-08.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
