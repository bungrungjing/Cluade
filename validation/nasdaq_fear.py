"""나스닥 한정 공포 기준 시험 — 사전 등록 (2026-11-08, 사용자 질문 "시장 전체가 아니라 나스닥 한정으로 봐야 하지 않나")

공포 측정 후보(사건 = 조건이 처음 성립한 날, 직전 20거래일 동안 미성립):
  M  시장 전체 폭 ≤30% (기존, 비교 기준)
  N1 나스닥 상장 종목 폭 ≤30% (universe exchange == NASDAQ)
  N2 기술 업종 종목 폭 ≤30% (universe sector == Technology)
  N3 QQQ 52주(252일) 고점 대비 −15% 이하 (원칙 13번과 같은 성격)
  N4 QQQ 52주 고점 대비 −20% 이하
시험 1: 사건 다음 날 QQQ 매수 → 60거래일 뒤 수익률. 합격: 두 구간(2010~17 / 2018~26) 모두 전체 평균보다 높음
시험 2: 2배 + QQQ 200일선 필터 + 사건 후 60일 예외 (H 구조). 합격: 연복리가 두 구간 모두 QQQ보다 높고 최대 낙폭이 QQQ의 1.5배 이내
'M보다 낫다' 판정: 시험 2에서 연복리가 두 구간 모두 M보다 높고 최대 낙폭이 M보다 3%p 넘게 나쁘지 않음
비용: 2배 연 2%. 한계: 생존 편향, 나스닥100 과거 구성 종목 자료 없음(현재 상장 나스닥 종목으로 근사)
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from edge_hunt import rd, half, stats, REPO
uni = pd.read_csv(os.path.join(REPO, "validation", "universe.csv")).set_index("ticker")
names = [n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
qqq = rd("daily/QQQ.csv.gz"); ab = {}
for p in names:
    t = os.path.basename(p).split(".")[0]
    if t in ("SPY","QQQ","_VIX"): continue
    try: d = rd(p)
    except Exception: continue
    if len(d) < 300: continue
    s = d.close.rolling(50).mean(); ab[t] = (d.close > s).where(s.notna())
a = pd.DataFrame(ab)
def breadth(cols, minn):
    x = a[cols]; return x.mean(axis=1)[x.count(axis=1) >= minn].reindex(qqq.index)
nas = [t for t in a if uni.exchange.get(t) == "NASDAQ"]; tech = [t for t in a if uni.sector.get(t) == "Technology"]
B = {"M 시장 전체": breadth(list(a), 200), "N1 나스닥 상장": breadth(nas, 100), "N2 기술 업종": breadth(tech, 40)}
dd = qqq.close / qqq.close.rolling(252, min_periods=120).max() - 1
cond = {k: v <= 0.30 for k, v in B.items()}
cond["N3 QQQ 고점 −15%"] = dd <= -0.15; cond["N4 QQQ 고점 −20%"] = dd <= -0.20
out = [f"# 나스닥 한정 공포 기준 (2026-11-08)\n\n종목 수: 시장 전체 {a.shape[1]} / 나스닥 상장 {len(nas)} / 기술 업종 {len(tech)}\n",
       "| 기준 | 사건 수 | 60일 뒤 QQQ (2010~17) | (2018~26) | 전체 평균 (2010~17 / 2018~26) | 시험1 |", "|---|---|---|---|---|---|"]
q = qqq.close; f60 = q.shift(-60) / q - 1
qr = q.pct_change().fillna(0); l2 = 2 * qr - 0.02 / 252
up = (q > q.rolling(200).mean()).shift(1, fill_value=False)
st = q.index >= "2010-04-01"
res = {}
for k, c in cond.items():
    c = c.fillna(False) & st
    ev = c & ~c.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    e = ev[ev].index; e1 = q.index[np.searchsorted(q.index, e) + 1] if len(e) and np.searchsorted(q.index, e).max() + 1 < len(q) else e
    hv = [f60[e1[half(e1) == h]].mean() for h in ("2010~17", "2018~26")]
    base = [f60[st & (half(q.index) == h)].mean() for h in ("2010~17", "2018~26")]
    ok1 = all(hv[i] > base[i] for i in range(2))
    out.append(f"| {k} | {len(e)} | {hv[0]*100:+.1f}% | {hv[1]*100:+.1f}% | {base[0]*100:+.1f}% / {base[1]*100:+.1f}% | {'합격' if ok1 else '불합격'} |")
    on = pd.Series(False, index=q.index)
    for i in np.where(ev.shift(1, fill_value=False).values)[0]: on.iloc[i:i + 60] = True
    res[k] = (l2.where(up | on, 0.0)[st], e)
A = qr[st]; cA, mA, _ = stats(A, 252); hA = [stats(A[half(A.index) == h], 252)[0] for h in ("2010~17", "2018~26")]
out += ["\n| 2배+200일선+공포 예외 | 연복리 | 최대낙폭 | 최악 연도 | 2010~17 | 2018~26 | 시험2 | M보다 나음 |", "|---|---|---|---|---|---|---|---|",
        f"| (기준) QQQ 보유 | {cA*100:.1f}% | {mA*100:.1f}% | | {hA[0]*100:.1f}% | {hA[1]*100:.1f}% | | |"]
rm = res["M 시장 전체"][0]; cM, mM, _ = stats(rm, 252); hM = [stats(rm[half(rm.index) == h], 252)[0] for h in ("2010~17", "2018~26")]
for k, (r, e) in res.items():
    c_, m_, _ = stats(r, 252); h_ = [stats(r[half(r.index) == h], 252)[0] for h in ("2010~17", "2018~26")]
    yr = (1 + r).groupby(r.index.year).prod() - 1
    ok2 = all(h_[i] > hA[i] for i in range(2)) and m_ >= 1.5 * mA
    better = "—" if k.startswith("M") else ("예" if all(h_[i] > hM[i] for i in range(2)) and m_ >= mM - 0.03 else "아니오")
    out.append(f"| {k} | {c_*100:.1f}% | {m_*100:.1f}% | {yr.min()*100:.1f}% | {h_[0]*100:.1f}% | {h_[1]*100:.1f}% | {'합격' if ok2 else '불합격'} | {better} |")
cur = {k: (f"{v.dropna().iloc[-1]*100:.1f}%" ) for k, v in B.items()}
out.append(f"\n현재({q.index[-1].date()}): 시장 전체 폭 {cur['M 시장 전체']} / 나스닥 상장 폭 {cur['N1 나스닥 상장']} / 기술 업종 폭 {cur['N2 기술 업종']} / QQQ 52주 고점 대비 {dd.iloc[-1]*100:.1f}% / QQQ 200일선 {'위' if q.iloc[-1] > q.rolling(200).mean().iloc[-1] else '아래'}")
for k, (r, e) in res.items():
    out.append(f"- {k} 사건: " + ", ".join(x.strftime('%Y-%m') for x in e))
print("\n".join(out)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "nasdaq_fear_2026-11-08.md"), "w").write("\n".join(out) + "\n")
