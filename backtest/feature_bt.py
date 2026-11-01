"""승·패 매매의 진입 시점 특징 비교 — 어떤 조건에서 오르고 어떤 조건에서 떨어졌나

사용법: python feature_bt.py <일봉 폴더>
 1) 통과 전체(아무 날 진입) 매매와 v3 BUY 매매에 진입일 특징을 붙인다
 2) 특징별 5분위로 평균 R·승률을 본다 (학습 구간 2022-10~2024-12)
 3) 학습 구간에서 고른 조건을 검증 구간(2025-01~2026-10)에 그대로 적용해 확인 (과적합 점검)
매매 규칙은 score_bt.trades_for 와 같음 (종가 진입, 손절 max(7%,2ATR), 3R, +1R 본전, 60일)
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators, trades_for  # noqa: E402

SPLIT = pd.Timestamp("2025-01-01")
FEATS = ["relvol", "adx", "k", "dist20", "dist50", "dist200", "atr_pct", "vwap_dist", "chop",
         "ret5", "ret63", "mkt_a50", "mkt_ret20", "hi_dd"]


def ci(x, n=1000):
    if len(x) < 10:
        return "  표본부족"
    bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(n)]
    return f"{np.percentile(bs, 2.5):+.2f}~{np.percentile(bs, 97.5):+.2f}"


def load(folder):
    data = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < 260:
            continue
        d = indicators(df)
        d["relvol"] = d.volume / d.volume.rolling(20).mean()
        for n, c in (("dist20", "sma20"), ("dist50", "sma50"), ("dist200", "sma200")):
            d[n] = (d.close / d[c] - 1) * 100
        d["ret5"] = d.close.pct_change(5) * 100
        d["ret63"] = d.close.pct_change(63) * 100
        d["ret20"] = d.close.pct_change(20) * 100
        d["hi_dd"] = (d.close / d.close.rolling(252, min_periods=120).max() - 1) * 100
        data[os.path.basename(f)[:-4]] = d.iloc[200:]
    a50 = pd.DataFrame({t: d.above50 for t, d in data.items()}).mean(axis=1) * 100
    r20 = pd.DataFrame({t: d.ret20 for t, d in data.items()}).median(axis=1)
    for d in data.values():
        d["mkt_a50"] = a50.reindex(d.index)
        d["mkt_ret20"] = r20.reindex(d.index)
    return data


def collect(data):
    rows = []
    for t, d in data.items():
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        v3 = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
        for name, m in (("전체", gate), ("v3", v3)):
            for tr in trades_for(d, m):
                tr.update({f: d.at[tr["date"], f] for f in FEATS})
                tr.update(ticker=t, set=name)
                rows.append(tr)
    r = pd.DataFrame(rows)
    r["win"] = r.ret > 0
    r["train"] = r.date < SPLIT
    return r


def quintiles(r, label):
    print(f"\n=== [{label}] 특징별 5분위 (학습 2022-10~2024-12, 매매 {len(r)}) — 평균R / 승률% ===")
    out = []
    for f in FEATS:
        x = r[[f, "R", "win"]].dropna()
        try:
            q = pd.qcut(x[f], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = x.groupby(q).agg(lo=(f, "min"), hi=(f, "max"), R=("R", "mean"), w=("win", "mean"), n=("R", "size"))
        spread = g.R.iloc[-1] - g.R.iloc[0]
        cells = "  ".join(f"[{a:.1f}~{b:.1f}] {rr:+.2f}/{ww*100:.0f}" for a, b, rr, ww in
                          zip(g.lo, g.hi, g.R, g.w))
        out.append((abs(spread), f, spread, cells))
    for _, f, s, c in sorted(out, reverse=True):
        print(f"{f:10s} 상-하 {s:+.2f}R | {c}")


def evaluate(r, rules, label):
    print(f"\n=== [{label}] 학습에서 고른 조건 → 검증(2025~2026) ===")
    print(f"{'조건':38s} {'학습 n':>6} {'승률':>5} {'평균R':>6} | {'검증 n':>6} {'승률':>5} {'평균R':>6} {'95%구간':>13}")
    for name, fn in rules:
        for part in (True, False):
            pass
        tr, te = r[r.train], r[~r.train]
        a, b = tr[fn(tr)], te[fn(te)]
        print(f"{name:38s} {len(a):6d} {a.win.mean()*100:5.1f} {a.R.mean():+6.2f} | "
              f"{len(b):6d} {b.win.mean()*100:5.1f} {b.R.mean():+6.2f} {ci(b.R.values):>13}")


def main():
    folder = sys.argv[1]
    data = load(folder)
    r = collect(data)
    allr, v3 = r[r.set == "전체"], r[r.set == "v3"]
    print(f"종목 {len(data)}개 | 전체 매매 {len(allr)} | v3 BUY 매매 {len(v3)}")
    # 승·패 평균 특징 비교
    print("\n=== 승·패 매매의 진입 특징 평균 (전체 매매, 전 기간) ===")
    cmp = allr.groupby("win")[FEATS].median().T
    cmp.columns = ["패(중앙값)", "승(중앙값)"]
    print(cmp.round(2).to_string())
    quintiles(allr[allr.train], "전체 매매")
    quintiles(v3[v3.train], "v3 BUY")
    rules = [
        ("기준(조건 없음)", lambda x: x.R.notna()),
        ("시장 폭 ≥ 60%", lambda x: x.mkt_a50 >= 60),
        ("시장 폭 ≤ 40% (역발상)", lambda x: x.mkt_a50 <= 40),
        ("시장 20일 중앙수익률 > 0", lambda x: x.mkt_ret20 > 0),
        ("52주 고점 대비 −10% 이내", lambda x: x.hi_dd >= -10),
        ("52주 고점 대비 −20% 이하", lambda x: x.hi_dd <= -20),
        ("ADX ≥ 25", lambda x: x.adx >= 25),
        ("50일선 0~5% 위", lambda x: x.dist50.between(0, 5)),
        ("50일선 +10% 이상 위", lambda x: x.dist50 >= 10),
        ("63일 수익률 ≥ 15%", lambda x: x.ret63 >= 15),
        ("63일 수익률 < 0", lambda x: x.ret63 < 0),
        ("CHOP < 50", lambda x: x.chop < 50),
        ("거래량 ≥ 1.5배", lambda x: x.relvol >= 1.5),
        ("ATR% ≤ 2", lambda x: x.atr_pct <= 2),
        ("ATR% ≥ 3", lambda x: x.atr_pct >= 3),
    ]
    evaluate(allr, rules, "전체 매매")
    evaluate(v3, rules, "v3 BUY")
    r.to_csv(os.path.join(folder, "..", "feature_bt_trades.csv"), index=False)


if __name__ == "__main__":
    main()
