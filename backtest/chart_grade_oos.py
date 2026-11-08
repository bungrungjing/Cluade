"""차트 등급 표본 밖 검증 (2026-11-08, 실행 전에 아래 가설·합격 기준을 먼저 적음 — 사전 등록)

데이터: data 갈래 daily/ 1,124종목 중 기존 52종목을 뺀 무작위 300종목(seed 7), 2010~2026-11
비교 점수:
  ① 기존 차트 점수(chart_grade_bt.py, S~F)
  ② 새 점수 v2 — 52종목 시험(chart_item_bt.py)에서 학습·검증 두 구간 모두 같은 방향이었던 항목만:
     ADX>25 & +DI>−DI 40 / 50일선 > 200일선 25 / OBV > 20일 평균 20 / 52주 고점 −25% 이내 15  (=100)
     등급 컷은 ①과 같음: S 85↑ A 75↑ B 65↑ C 50↑ D 35↑ F
  ③ 시장 폭 ≤30%(표본 300종목 중 50일선 위 비율) — 등급과 별개로 따로 표시
매매: 종가 진입, 손절 max(7%,2ATR), 목표 3R, +1R 후 본전, 최대 60일, 왕복 0.2%, 종목당 한 포지션(겹침 없음)
매도 쪽: 그날 등급별 '20거래일 뒤 수익률' 평균 — 낮은 등급이 이후 하락을 예고하는지
합격 기준(둘 다 충족해야 '매수에 유의미'):
  (a) 2010~2017, 2018~2026 두 구간 모두에서 S+A 평균R이 기준선(아무 날)보다 높고, 전체 기간 S+A의 95% 구간 하한이 기준선 평균보다 높다
  (b) 등급 순서가 대체로 단조(S+A > B·C > D·F)
매도 쪽 합격: 두 구간 모두 F의 20일 뒤 수익률이 기준선보다 낮고, 전체 95% 구간 상한이 기준선보다 낮다
"""
import sys, os, subprocess, io, random
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_grade_bt import chart_score, GRADES  # noqa: E402

FEE, MAXHOLD = 0.002, 60
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@njit
def sim(c, h, l, atr, mask):
    n = len(c); R = np.full(n, np.nan); nxt = 0
    for i in range(n - 1):
        if not mask[i] or i < nxt:
            continue
        e = c[i]; risk = max(0.07 * e, 2 * atr[i])
        if not np.isfinite(risk):
            continue
        stop = e - risk; tgt = e + 3 * risk; be = False; px = -1.0; j = i
        for j in range(i + 1, min(n, i + 1 + MAXHOLD)):
            if l[j] <= stop:
                px = stop; break
            if h[j] >= tgt:
                px = tgt; break
            if (not be) and h[j] >= e + risk:
                stop = e; be = True
        if px < 0:
            if j < i + MAXHOLD:
                break
            px = c[j]
        R[i] = (px - e) / risk - FEE * e / risk
        nxt = j + 1
    return R


def grade(s):
    return np.select([s >= t for t, _ in GRADES], [g for _, g in GRADES], "F")


def main():
    names = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", "origin/data"], capture_output=True, text=True).stdout.split()
    old = {f[:-4] for f in os.listdir(sys.argv[1])} if len(sys.argv) > 1 else set()
    pool = sorted(n for n in names if n.startswith("daily/") and not os.path.basename(n).split(".")[0] in old | {"QQQ", "_VIX"})
    random.Random(7).shuffle(pool)
    def rd(p):
        raw = subprocess.run(["git", "-C", REPO, "show", f"origin/data:{p}"], capture_output=True).stdout
        df = pd.read_csv(io.BytesIO(raw), compression="gzip", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)
    qqq = rd("daily/QQQ.csv.gz")
    data = {}
    for p in pool[:300]:
        try:
            df = rd(p)
        except Exception:
            continue
        if len(df) < 400:
            continue
        d = chart_score(df, qqq)
        c = d.close; v = d.volume
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        d["v2"] = (((d.adx > 25) & (d.pdi > d.mdi)) * 40 + (d.sma50 > d.sma200) * 25 + (obv > obv.rolling(20).mean()) * 20
                   + (c >= d.high.rolling(252, min_periods=120).max() * 0.75) * 15)
        d["fwd20"] = c.shift(-20) / c - 1
        data[os.path.basename(p).split(".")[0]] = d.iloc[252:]
    a50 = pd.DataFrame({t: d.close > d.sma50 for t, d in data.items()}).mean(axis=1)
    rows, fw = [], []
    for t, d in data.items():
        gate = ((d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)).values
        g1, g2 = grade(d.chart_score.values), grade(d.v2.values)
        br = (a50.reindex(d.index) <= 0.30).values
        arr = [d.close.values, d.high.values, d.low.values, d.atr.values]
        groups = {"기준선": gate, "시장폭≤30%": gate & br}
        for g in "SABCDF":
            groups[f"①{g}"] = gate & (g1 == g); groups[f"②{g}"] = gate & (g2 == g)
        groups["①S+A"] = gate & np.isin(g1, ["S", "A"]); groups["②S+A"] = gate & np.isin(g2, ["S", "A"])
        for k, m in groups.items():
            R = sim(*arr, m.astype(np.bool_))
            ok = ~np.isnan(R)
            for dt, r in zip(d.index[ok], R[ok]):
                rows.append((k, dt.year, r))
        f = pd.DataFrame({"g1": g1, "g2": g2, "fwd": d.fwd20.values, "y": d.index.year})[gate]
        fw.append(f)
    r = pd.DataFrame(rows, columns=["group", "year", "R"])
    r["구간"] = np.where(r.year <= 2017, "2010~17", "2018~26")
    def s(x):
        m, sd, n = x.mean(), x.std(), len(x)
        return pd.Series({"매매": n, "평균R": m, "하한": m - 1.96 * sd / np.sqrt(n), "상한": m + 1.96 * sd / np.sqrt(n), "승률%": (x > 0).mean() * 100})
    order = ["①S", "①A", "①S+A", "①B", "①C", "①D", "①F", "②S", "②A", "②S+A", "②B", "②C", "②D", "②F", "시장폭≤30%", "기준선"]
    print(f"종목 {len(data)}개 (표본 밖)")
    print(r.groupby("group").R.apply(s).unstack().reindex(order).round(3).to_string())
    print("\n구간별 평균R")
    print(r.groupby(["group", "구간"]).R.mean().unstack().reindex(order).round(3).to_string())
    f = pd.concat(fw); f["구간"] = np.where(f.y <= 2017, "2010~17", "2018~26")
    print("\n매도 쪽: 등급별 20거래일 뒤 평균 수익률% (겹침 포함)")
    for col in ("g1", "g2"):
        t = f.groupby([col, "구간"]).fwd.mean().unstack() * 100
        t.loc["기준선"] = f.groupby("구간").fwd.mean() * 100
        allm = f.groupby(col).fwd.agg(["mean", "std", "size"])
        t["전체"] = allm["mean"] * 100
        print(("①기존" if col == "g1" else "②새 v2"), "\n", t.round(2).to_string())


if __name__ == "__main__":
    main()
