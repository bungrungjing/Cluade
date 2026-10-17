"""수익추종형 청산 비교 — v3 BUY 진입 고정, 3R 목표 없이 추세를 따라 보유하는 방식 검증

사용법: python trend_bt.py <일봉 폴더>
 [기준] 3R 목표 + +1R 본전 손절 + 60일 (현재 방어형)
 [추적] 최고가 − k×ATR 추적 손절 (k=3,4,5), 목표 없음, 최대 250거래일
 [50일선] 종가가 50일선 아래로 내려가면 청산, 초기 손절 max(7%,2ATR)
 갭 하락은 시가 체결로 처리, 수수료 왕복 0.2%
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators, trades_for  # noqa: E402

FEE, SPLIT = 0.002, pd.Timestamp("2025-01-01")


def trend_trades(d, mask, mode, k=3.0, maxhold=250):
    out, nxt, n = [], 0, len(d)
    for i in np.where(mask.values)[0]:
        if i < nxt or i >= n - 1:
            continue
        e = d.close.iloc[i]
        risk = max(0.07 * e, 2 * d.atr.iloc[i])
        stop, hh, px, why, j = e - risk, e, None, None, i
        for j in range(i + 1, min(n, i + 1 + maxhold)):
            if d.low.iloc[j] <= stop:
                px, why = min(stop, d.open.iloc[j]), "손절"; break
            if mode == "sma50" and d.close.iloc[j] < d.sma50.iloc[j]:
                px, why = d.close.iloc[j], "50일선"; break
            hh = max(hh, d.high.iloc[j])
            if mode == "trail":
                stop = max(stop, hh - k * d.atr.iloc[j])
        if px is None:
            px, why = d.close.iloc[j], "기간/미결"
        out.append(dict(date=d.index[i], R=(px - e) / risk - FEE * e / risk,
                        ret=px / e - 1 - FEE, hold=j - i, why=why))
        nxt = j + 1
    return out


def ci(x):
    bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(1000)]
    return f"{np.percentile(bs, 2.5):+.2f}~{np.percentile(bs, 97.5):+.2f}"


def main():
    folder = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < 260:
            continue
        d = indicators(df)
        d["relvol"] = d.volume / d.volume.rolling(20).mean()
        d = d.iloc[200:]
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        buy = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
        t = os.path.basename(f)[:-4]
        sets = {"기준 3R+본전": trades_for(d, buy),
                "추적 k=3": trend_trades(d, buy, "trail", 3),
                "추적 k=4": trend_trades(d, buy, "trail", 4),
                "추적 k=5": trend_trades(d, buy, "trail", 5),
                "50일선 이탈": trend_trades(d, buy, "sma50")}
        for name, trs in sets.items():
            for tr in trs:
                tr.update(model=name, ticker=t); rows.append(tr)
    r = pd.DataFrame(rows)
    r["win"] = r.ret > 0
    order = list(dict.fromkeys(r.model))
    print(f"종목 {r.ticker.nunique()}개")
    for label, sub in (("전 기간", r), ("학습 2022-10~2024-12", r[r.date < SPLIT]), ("검증 2025~2026", r[r.date >= SPLIT])):
        print(f"\n=== {label} ===")
        g = sub.groupby("model").agg(매매=("R", "size"), 승률=("win", "mean"), 평균R=("R", "mean"),
                                    중앙R=("R", "median"), 평균수익률=("ret", "mean"), 평균보유=("hold", "mean"),
                                    최대이익R=("R", "max"), 최악R=("R", "min"))
        g["승률"] *= 100; g["평균수익률"] *= 100
        g["평균R 95%구간"] = sub.groupby("model").R.apply(lambda x: ci(x.values) if len(x) > 10 else "-")
        print(g.reindex(order).round(2).to_string())
    r.to_csv(os.path.join(folder, "..", "trend_bt_trades.csv"), index=False)


if __name__ == "__main__":
    main()
