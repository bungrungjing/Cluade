"""승률 개선안 비교 — v2 BUY 진입 고정, 청산 방식·추가 필터를 바꿔 승률과 평균 R을 함께 비교

사용법: python winrate_bt.py <일봉 폴더>
비교 항목
 [청산] 목표 1.5R / 2R / 3R, 절반 익절(+1R에 절반 매도, 나머지 3R·본전 손절), 본전 손절 유무
 [필터] 시장 폭(표본 종목 중 50일선 위 비율 ≥ 50%), 거래량(신호일 거래량 ≥ 20일 평균 1.2배),
        상대강도(63일 수익률이 표본 중앙값 이상), 200일선 위
 승률만 올리면 평균 R이 떨어질 수 있으므로 반드시 평균 R(기대값)과 같이 본다.
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators  # noqa: E402

FEE, MAXHOLD = 0.002, 60


def sim(d, mask, tgt_r=3.0, be=True, half=False):
    out, nxt, n = [], 0, len(d)
    for i in np.where(mask.values)[0]:
        if i < nxt or i >= n - 1:
            continue
        e = d.close.iloc[i]
        risk = max(0.07 * e, 2 * d.atr.iloc[i])
        stop, tgt, moved, banked, size = e - risk, e + tgt_r * risk, False, 0.0, 1.0
        res, j = None, i
        for j in range(i + 1, min(n, i + 1 + MAXHOLD)):
            if d.low.iloc[j] <= stop:
                res = banked + size * (stop - e) / risk; break
            if d.high.iloc[j] >= tgt:
                res = banked + size * tgt_r; break
            if d.high.iloc[j] >= e + risk:
                if half and size == 1.0:
                    banked, size = 0.5 * 1.0, 0.5
                if be and not moved:
                    stop, moved = e, True
        if res is None:
            res = banked + size * (d.close.iloc[j] - e) / risk
        res -= FEE * e / risk
        out.append(res)
        nxt = j + 1
    return out


def main():
    folder = sys.argv[1]
    data = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < 260:
            continue
        d = indicators(df)
        d["sma200"] = d.close.rolling(200).mean()
        d["ret63"] = d.close.pct_change(63)
        d["relvol"] = d.volume / d.volume.rolling(20).mean()
        data[os.path.basename(f)[:-4]] = d.iloc[200:]
    a50 = pd.DataFrame({t: d.above50 for t, d in data.items()}).mean(axis=1)
    rs_med = pd.DataFrame({t: d.ret63 for t, d in data.items()}).median(axis=1)
    rows = []
    for t, d in data.items():
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        buy = gate & (d.signal == "BUY")
        breadth = a50.reindex(d.index) >= 0.5
        vol = d.relvol >= 1.2
        rs = d.ret63 >= rs_med.reindex(d.index)
        up200 = d.close > d.sma200
        tests = [
            ("기준: v2 BUY, 3R, 본전손절", buy, dict()),
            ("[청산] 목표 2R", buy, dict(tgt_r=2.0)),
            ("[청산] 목표 1.5R", buy, dict(tgt_r=1.5)),
            ("[청산] +1R 절반 익절 + 3R", buy, dict(half=True)),
            ("[청산] 본전손절 없음, 3R", buy, dict(be=False)),
            ("[필터] 시장 폭 50%↑", buy & breadth, dict()),
            ("[필터] 거래량 1.2배↑", buy & vol, dict()),
            ("[필터] 상대강도 상위 절반", buy & rs, dict()),
            ("[필터] 200일선 위", buy & up200, dict()),
            ("[조합] 시장폭+상대강도, 절반익절", buy & breadth & rs, dict(half=True)),
            ("[조합] 시장폭+상대강도, 2R", buy & breadth & rs, dict(tgt_r=2.0)),
        ]
        for name, m, kw in tests:
            for r in sim(d, m, **kw):
                rows.append(dict(test=name, R=r, ticker=t))
    r = pd.DataFrame(rows)
    order = list(dict.fromkeys(r.test))
    s = r.groupby("test").agg(매매=("R", "size"), 승률=("R", lambda x: (x > 0).mean() * 100),
                              평균R=("R", "mean"), 중앙R=("R", "median"))
    def ci(x):
        bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(1000)]
        return f"{np.percentile(bs, 2.5):+.2f}~{np.percentile(bs, 97.5):+.2f}"
    s["평균R 95%구간"] = r.groupby("test").R.apply(lambda x: ci(x.values))
    print(f"종목 {len(data)}개")
    print(s.reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
