"""전략 토너먼트 — 문헌에 있는 대표 전략을 결과 보기 전에 한꺼번에 등록하고 한 번만 시험 (2026-11-01)

사용법: python tournament_bt.py <일봉 폴더> <결과 md>
공통: 1단계 관문(주가 5달러↑, ATR% 1~4 대신 '거래대금 4,000만 달러↑'만 — 전략마다 변동성 성격이 달라서), 동시 5종목 균등(20%씩)
 S1 횡단면 모멘텀(Jegadeesh·Titman 1993): 월말, 200일선 위 종목 중 12-1개월 수익률 상위 5 → 한 달 보유
 S2 52주 고점 근접(George·Hwang 2004): 월말, 종가/52주 고점 상위 5(200일선 위) → 한 달 보유
 S3 RSI(2) 되돌림(Connors·Alvarez 2009): 종가>200일선 & RSI(2)<10 → 종가 매수, 종가>5일선이면 청산(최대 10일), 같은 날 여럿이면 RSI 낮은 순
 S4 단기 되돌림(Lehmann 1990 계열): 주말, 200일선 위 종목 중 5일 수익률 하위 5 → 1주 보유
 S5 모멘텀 + 시장 필터: S1과 같되 SPY 200일선 아래 달은 현금
비용: 매매 금액의 0.2%(왕복). 설계 2011~2021 / 검증 2022~
통과 기준: 검증 구간 연복리가 QQQ 보유보다 높고, 최대 낙폭이 QQQ의 1.5배 이내
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import load_csv  # noqa: E402
from sector_rsi_bt import rsi  # noqa: E402

FEE, N = 0.002, 5
SPLIT = "2022-01-01"


def perf(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    m = eq.resample("ME").last().pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    return {"연복리": f"{100 * cagr:.1f}%", "월복리": f"{100 * ((1 + cagr) ** (1 / 12) - 1):.2f}%", "최대낙폭": f"{100 * dd:.1f}%",
            "월+5%이상": f"{100 * (m >= 0.05).mean():.0f}%", "최악달": f"{100 * m.min():.1f}%", "_cagr": cagr, "_dd": dd}


def main():
    folder, outp = sys.argv[1], sys.argv[2]
    C, H, V = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        if t == "_VIX":   # VIX는 미국 휴장일에도 값이 있어 다른 종목에 빈 날짜를 만듦 → 제외
            continue
        df = load_csv(f)
        if len(df) < 300:
            continue
        C[t], H[t], V[t] = df.close, df.high, df.volume
    C, H, V = pd.DataFrame(C)["2009-01-01":], pd.DataFrame(H)["2009-01-01":], pd.DataFrame(V)["2009-01-01":]
    spy, qqq = C.pop("SPY"), C.pop("QQQ")
    H, V = H[C.columns], V[C.columns]
    ok = (C >= 5) & ((C * V).rolling(20).mean() >= 40e6) & (C > C.rolling(200).mean())
    ret = C.pct_change()

    def monthly(score, filt=None):
        me = C.resample("ME").last().index
        days = C.index
        eq, curve, held = 1.0, {}, []
        for a, b in zip(me[:-1], me[1:]):
            da = days[days <= a][-1]
            s = score.loc[da].where(ok.loc[da]).dropna()
            if filt is not None and not filt.get(da, True):
                pick = []
            else:
                pick = list(s.nlargest(N).index)
            period = ret.loc[(ret.index > da) & (ret.index <= b), pick] if pick else None
            turn = len(set(pick) ^ set(held)) / (2 * N) if pick or held else 0
            eq *= 1 - FEE * turn
            if period is not None and len(period):
                path = (1 + period.fillna(0)).cumprod().mean(axis=1)
                for d, v in path.items():
                    curve[d] = eq * v
                eq *= path.iloc[-1]
            else:
                for d in days[(days > da) & (days <= b)]:
                    curve[d] = eq
            held = pick
        return pd.Series(curve)

    def events(entry, exit_rule, rank, maxhold):
        slots, cash_w = [], 1.0
        days = C.index
        vals = {}
        for i, d in enumerate(days):
            if i == 0:
                continue
            # 보유 종목 하루 수익
            keep = []
            for p in slots:
                r = ret.at[d, p["t"]]
                r = 0 if np.isnan(r) else r
                p["v"] *= 1 + r
                p["n"] += 1
                if exit_rule(p["t"], d) or p["n"] >= maxhold:
                    cash_w += p["v"] * (1 - FEE)
                else:
                    keep.append(p)
            slots = keep
            total = cash_w + sum(p["v"] for p in slots)
            free = N - len(slots)
            if free > 0:
                cand = entry.loc[d]
                cand = cand[cand].index.difference([p["t"] for p in slots])
                if len(cand):
                    cand = rank.loc[d, cand].sort_values().index[:free]
                    for t in cand:
                        w = min(total / N, cash_w)
                        if w <= 0:
                            break
                        cash_w -= w
                        slots.append(dict(t=t, v=w, n=0))
            vals[d] = cash_w + sum(p["v"] for p in slots)
        return pd.Series(vals)

    r12 = C.shift(21) / C.shift(252) - 1
    hi52 = C / H.rolling(252).max()
    spy_ok = (spy > spy.rolling(200).mean())
    R2 = pd.DataFrame({t: rsi(C[t], 2) for t in C.columns})
    sma5 = C.rolling(5).mean()
    r5 = C / C.shift(5) - 1

    res = {}
    res["S1 모멘텀 12-1"] = monthly(r12)
    res["S2 52주 고점 근접"] = monthly(hi52)
    res["S5 모멘텀 + SPY 200일선"] = monthly(r12, spy_ok)
    s3_entry = ok & (R2 < 10)
    res["S3 RSI(2) 되돌림"] = events(s3_entry, lambda t, d: C.at[d, t] > sma5.at[d, t], R2, 10)
    wk = C.resample("W-FRI").last().index
    s4_entry = pd.DataFrame(False, index=C.index, columns=C.columns)
    fri = C.index[C.index.isin(wk)]
    s4_entry.loc[fri] = ok.loc[fri] & (r5.loc[fri].where(ok.loc[fri]).rank(axis=1) <= N)
    res["S4 단기 되돌림(주간)"] = events(s4_entry, lambda t, d: False, r5, 5)
    rq = qqq.pct_change().fillna(0)
    res["기준: QQQ 보유"] = (1 + rq).cumprod()

    rows = {}
    for k, eq in res.items():
        eq = eq["2011-01-01":]
        a, b, c = perf(eq), perf(eq[:"2021-12-31"]), perf(eq[SPLIT:])
        rows[k] = {"연복리": a["연복리"], "월복리": a["월복리"], "최대낙폭": a["최대낙폭"], "월+5%이상": a["월+5%이상"],
                   "최악달": a["최악달"], "설계 연복리": b["연복리"], "검증 연복리": c["연복리"], "검증 낙폭": c["최대낙폭"],
                   "_vc": c["_cagr"], "_vd": c["_dd"]}
    q = rows["기준: QQQ 보유"]
    for k, v in rows.items():
        v["판정"] = "-" if k.startswith("기준") else ("통과" if v["_vc"] > q["_vc"] and v["_vd"] >= 1.5 * q["_vd"] else "미통과")
    df = pd.DataFrame(rows).T.drop(columns=["_vc", "_vd"])
    txt = "\n".join([f"# 전략 토너먼트 ({pd.Timestamp.now():%Y-%m-%d})", "", "등록 전략 5개, 한 번만 실행. 2011-01 ~ 2026-10, 비용 0.2%, 동시 5종목 균등.", "",
                     df.to_markdown(), "", "- 통과 기준: 검증(2022~) 연복리 > QQQ, 검증 최대 낙폭이 QQQ의 1.5배 이내",
                     "- 생존 편향 있음(지금 살아 있는 종목만) — 특히 모멘텀·되돌림 결과를 부풀릴 수 있음", ""])
    print(txt)
    open(outp, "w").write(txt)


if __name__ == "__main__":
    main()
