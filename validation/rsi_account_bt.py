"""RSI 필터 계좌 시뮬레이션 + 개별 종목 RSI 버전 (2026-11-01, 사용자 질문: "월 5%가 나와?", "개별주로 한다면?")

사용법: python rsi_account_bt.py <일봉 폴더> <결과 md>
진입 조건(1단계 관문 통과일 중): 필터 없음 / 섹터 평균 RSI ≤ 40 / 개별 종목 RSI ≤ 40 / 개별 종목 RSI ≥ 60
청산: LIVE(실전 손절 규칙) / HOLD60(손절 없이 60거래일)
계좌: 1만 달러, 동시 최대 5종목, 같은 날 신호는 무작위 순서(5회)
 - LIVE: 매매당 위험 2%(손절폭 기준 수량), 종목당 상한 계좌의 30%
 - HOLD60: 종목당 계좌의 20%씩
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE  # noqa: E402
from full_bt import stats  # noqa: E402
from sector_rsi_bt import rsi  # noqa: E402
from exposure_bt import perf  # noqa: E402

SPLIT = pd.Timestamp("2022-01-01")


def account(tr, mode, seed, cap=10000.0, slots=5):
    tr = tr.sample(frac=1, random_state=seed).sort_values("date", kind="stable")
    eq, open_, curve = cap, [], {tr.date.min(): cap}
    for x in tr.itertuples():
        keep = []
        for p in open_:
            if p[0] <= x.date:
                eq += p[1]; curve[p[0]] = eq
            else:
                keep.append(p)
        open_ = keep
        if len(open_) >= slots or any(p[2] == x.ticker for p in open_):
            continue
        if mode == "LIVE":
            size = min(eq * 0.02 / (x.risk / x.entry), eq * 0.30)
        else:
            size = eq / slots
        open_.append((x.exit_date, size * x.ret, x.ticker))
    for p in sorted(open_):
        eq += p[1]; curve[p[0]] = eq
    c = pd.Series(curve).sort_index().groupby(level=0).last()
    return c.reindex(pd.date_range(c.index[0], c.index[-1], freq="D")).ffill()


def main():
    folder, outp = sys.argv[1], sys.argv[2]
    uni = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.csv")).set_index("ticker").sector.fillna("Unknown")
    ind = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        if t not in uni.index:
            continue
        df = load_csv(f)
        if len(df) < 260:
            continue
        d = indicators(df)
        d["rsi"] = rsi(df.close)
        ind[t] = d.iloc[200:]
    R = pd.DataFrame({t: d.rsi for t, d in ind.items()})
    sec = {s: R[[t for t in R.columns if uni[t] == s]].mean(axis=1) for s in set(uni[t] for t in R.columns)}
    rows = []
    for t, d in ind.items():
        sr = sec[uni[t]].reindex(d.index)
        masks = {"필터 없음": d.gate, "섹터 RSI ≤ 40": d.gate & (sr <= 40),
                 "개별 RSI ≤ 40": d.gate & (d.rsi <= 40), "개별 RSI ≥ 60": d.gate & (d.rsi >= 60)}
        c, idx = d.close.values, d.index
        for name, m in masks.items():
            for tr in sim(d, m, **LIVE):
                if tr["status"] == "closed":
                    tr.update(test=name, ticker=t, model="LIVE"); rows.append(tr)
            nxt = 0
            for i in np.where(m.values)[0]:
                if i < nxt or i + 60 >= len(c):
                    continue
                rows.append(dict(date=idx[i], exit_date=idx[i + 60], entry=c[i], risk=np.nan,
                                 ret=c[i + 60] / c[i] - 1 - 0.002, R=np.nan, test=name, ticker=t, model="HOLD60"))
                nxt = i + 61
    r = pd.DataFrame(rows)
    order = ["필터 없음", "섹터 RSI ≤ 40", "개별 RSI ≤ 40", "개별 RSI ≥ 60"]
    L = [f"# RSI 필터 — 계좌 시뮬레이션과 개별 종목 RSI ({pd.Timestamp.now():%Y-%m-%d})", ""]
    x = r[r.model == "LIVE"]
    tab = {}
    for k in order:
        a, b = stats(x[x.test == k]), stats(x[(x.test == k) & (x.date >= SPLIT)])
        tab[k] = dict(매매=a["매매"], 승률=a["승률"], 평균R=a["평균R"], R구간95=a["R구간95"], 검증_평균R=b["평균R"], 검증_R구간95=b["R구간95"])
    L += ["## 1. 매매 한 건당 (실전 손절 규칙)", "", pd.DataFrame(tab).T.to_markdown(), ""]
    h = r[r.model == "HOLD60"]
    tab = {k: dict(매매=len(h[h.test == k]), 승률=round(100 * (h[h.test == k].ret > 0).mean(), 1),
                   평균수익률=round(100 * h[h.test == k].ret.mean(), 2),
                   검증_평균수익률=round(100 * h[(h.test == k) & (h.date >= SPLIT)].ret.mean(), 2)) for k in order}
    L += ["## 2. 매매 한 건당 (손절 없이 60거래일)", "", pd.DataFrame(tab).T.to_markdown(), ""]
    acc = {}
    for mode in ("LIVE", "HOLD60"):
        for k in order:
            y = r[(r.model == mode) & (r.test == k)]
            ps = [perf(account(y, mode, s)) for s in range(5)]
            df = pd.DataFrame(ps)
            rec = df.iloc[0].to_dict()
            rec["연복리"] = " / ".join(df["연복리"])
            acc[f"{k} / {'실전 손절(위험 2%)' if mode == 'LIVE' else '60일 보유(20%씩)'}"] = rec
    L += ["## 3. 계좌 시뮬레이션 (1만 달러, 동시 5종목, 2010~, 연복리는 무작위 순서 5회)", "",
          pd.DataFrame(acc).T.to_markdown(), "",
          "- 계좌 손익은 청산일에 반영(보유 중 평가손익 미반영) → 실제 낙폭은 더 클 수 있음. 생존 편향 있음.", ""]
    txt = "\n".join(L)
    print(txt)
    open(outp, "w").write(txt)


if __name__ == "__main__":
    main()
