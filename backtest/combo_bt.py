"""종합 초안 백테스트 — 방어형(40%) + 수익추종형(60%) 두 슬롯, 그리고 승률 개선안 비교

사용법: python combo_bt.py <일봉 폴더>
 공통 진입: v3 BUY (v2 + 신호일 거래량 ≥ 20일 평균 1.2배), 종가 진입, 통과 조건(5달러·ATR 1~4%·거래대금 4,000만)
 방어형:   손절 max(7%,2ATR), 목표 3R, +1R 본전, 최대 60일
 수익추종: 손절 max(7%,2ATR), 목표 없음, +1R 본전 → 이후 최고가 − 4ATR 추적, 최대 250일
 승리 = 수수료(왕복 0.2%) 뺀 수익률 > 0
 개선안은 2022-10~2024-12(학습)과 2025~2026(검증)에서 같이 확인
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators  # noqa: E402

FEE, SPLIT = 0.002, pd.Timestamp("2025-01-01")


def sim(d, mask, tgt=3.0, be_at=1.0, lock=0.0, trail=None, half_at=None, maxhold=60, time_stop=None):
    """tgt: 목표 R(None이면 없음), be_at: 손절을 올리는 R, lock: 그때 올릴 손절 위치(R),
    trail: 손절 이동 뒤 최고가 − trail×ATR, half_at: 이 R에서 절반 익절, time_stop: (일, R) 이 기간 안에 R 미도달 시 청산"""
    out, nxt, n = [], 0, len(d)
    for i in np.where(mask.values)[0]:
        if i < nxt or i >= n - 1:
            continue
        e = d.close.iloc[i]
        risk = max(0.07 * e, 2 * d.atr.iloc[i])
        stop, hh, moved, banked, size, res, j = e - risk, e, False, 0.0, 1.0, None, i
        for j in range(i + 1, min(n, i + 1 + maxhold)):
            o, lo, hi = d.open.iloc[j], d.low.iloc[j], d.high.iloc[j]
            if lo <= stop:
                res = banked + size * (min(stop, o) - e) / risk; break
            if tgt is not None and hi >= e + tgt * risk:
                res = banked + size * tgt; break
            hh = max(hh, hi)
            if half_at is not None and size == 1.0 and hh >= e + half_at * risk:
                banked, size = 0.5 * half_at, 0.5
            if not moved and hh >= e + be_at * risk:
                stop, moved = max(stop, e + lock * risk), True
            if moved and trail is not None:
                stop = max(stop, hh - trail * d.atr.iloc[j])
            if time_stop and j - i == time_stop[0] and hh < e + time_stop[1] * risk:
                res = banked + size * (d.close.iloc[j] - e) / risk; break
        if res is None:
            res = banked + size * (d.close.iloc[j] - e) / risk
        res -= FEE * e / risk
        out.append(dict(date=d.index[i], R=res, ret=res * risk / e, hold=j - i))
        nxt = j + 1
    return out


DEF = dict(tgt=3.0, maxhold=60)
TRD = dict(tgt=None, trail=4.0, maxhold=250)


def main():
    folder = sys.argv[1]
    data = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < 260:
            continue
        d = indicators(df)
        d["relvol"] = d.volume / d.volume.rolling(20).mean()
        data[os.path.basename(f)[:-4]] = d.iloc[200:]
    a50 = pd.DataFrame({t: d.above50 for t, d in data.items()}).mean(axis=1) * 100
    rows = []
    for t, d in data.items():
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        v3 = gate & (d.signal == "BUY") & (d.relvol >= 1.2)
        v2 = gate & (d.signal == "BUY")
        br = a50.reindex(d.index)
        tests = []
        for model, base in (("방어형", DEF), ("수익추종", TRD)):
            tests += [
                (model, "① 초안 그대로", v3, base),
                (model, "② +0.5R에서 본전", v3, {**base, "be_at": 0.5}),
                (model, "③ +1R에서 +0.3R로 손절 상향", v3, {**base, "lock": 0.3}),
                (model, "④ +1R 절반 익절", v3, {**base, "half_at": 1.0}),
                (model, "⑤ 10일 내 +0.5R 미도달 시 청산", v3, {**base, "time_stop": (10, 0.5)}),
                (model, "⑥ 진입: 거래량 1.5배↑", v3 & (d.relvol >= 1.5), base),
                (model, "⑦ 진입: 시장 폭 ≤ 50%", v3 & (br <= 50), base),
                (model, "⑧ 진입: v2(거래량 조건 뺌)", v2, base),
                (model, "⑨ ②+④ 조합", v3, {**base, "be_at": 0.5, "half_at": 1.0}),
            ]
        for model, name, m, kw in tests:
            for tr in sim(d, m, **kw):
                tr.update(model=model, test=name, ticker=t)
                rows.append(tr)
    r = pd.DataFrame(rows)
    r["win"] = r.ret > 0
    r.to_csv(os.path.join(folder, "..", "combo_bt_trades.csv"), index=False)

    def ci(x):
        bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(1000)]
        return f"{np.percentile(bs, 2.5) * 100:+.1f}~{np.percentile(bs, 97.5) * 100:+.1f}"

    for model in ("방어형", "수익추종"):
        s = r[r.model == model]
        order = list(dict.fromkeys(s.test))
        out = []
        for name in order:
            x = s[s.test == name]
            tr, te = x[x.date < SPLIT], x[x.date >= SPLIT]
            out.append(dict(조건=name, 매매=len(x), 승률=x.win.mean() * 100, 평균수익률=x.ret.mean() * 100,
                            평균R=x.R.mean(), 구간=ci(x.ret.values), 학습승률=tr.win.mean() * 100,
                            학습수익률=tr.ret.mean() * 100, 검증승률=te.win.mean() * 100,
                            검증수익률=te.ret.mean() * 100, 보유=x.hold.mean()))
        print(f"\n=== {model} ===")
        print(pd.DataFrame(out).set_index("조건").round(2).to_string())


if __name__ == "__main__":
    main()
