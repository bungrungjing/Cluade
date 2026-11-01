"""섹터 평균 RSI 매수 필터 — 사용자 가설, 사전 등록 한 번만 시험 (2026-11-01)

가설: 섹터 평균 RSI(14)가 40 이하면 매수, 60 이상이면 매수하지 않는다.
섹터 평균 RSI = 그날 같은 섹터(universe.csv) 종목들의 RSI(14, 와일더) 단순 평균.

1) 종목 매매: 1단계 관문 통과일 중
   - 섹터 RSI ≤ 40 인 날만 진입 / 40~60 / ≥ 60 / 전체(필터 없음) 비교
   - 청산: 실전 규칙(LIVE) 과 손절 없이 60거래일 보유 두 가지
2) 지수 매매(기술 섹터 평균 RSI 사용): RSI ≤ 40 이 되면 다음 날 매수, RSI ≥ 60 이 되면 다음 날 현금
   - QQQ 1배 / 3배 근사(보수·차입·전환 비용 반영)
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE  # noqa: E402
from full_bt import stats  # noqa: E402
from public_models_bt import run, perf  # noqa: E402

SPLIT = pd.Timestamp("2022-01-01")


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def main():
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    uni = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.csv")).set_index("ticker").sector.fillna("Unknown")
    ind, rs = {}, {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        if t not in uni.index:
            continue
        df = load_csv(f)
        if len(df) < 260:
            continue
        d = indicators(df).iloc[200:]
        d["rsi"] = rsi(df.close).reindex(d.index)
        ind[t] = d
        rs[t] = d.rsi
    R = pd.DataFrame(rs)
    sec = {s: R[[t for t in R.columns if uni[t] == s]].mean(axis=1) for s in uni.unique() if any(uni[t] == s for t in R.columns)}
    cnt = {s: sum(uni[t] == s for t in R.columns) for s in sec}

    rows = []
    for t, d in ind.items():
        sr = sec[uni[t]].reindex(d.index)
        masks = {"전체(필터 없음)": d.gate, "섹터 RSI ≤ 40": d.gate & (sr <= 40),
                 "섹터 RSI 40~60": d.gate & sr.between(40, 60, inclusive="neither"), "섹터 RSI ≥ 60": d.gate & (sr >= 60)}
        c = d.close.values
        for name, m in masks.items():
            for tr in sim(d, m, **LIVE):
                tr.update(test=name, ticker=t, model="LIVE"); rows.append(tr)
            idx = np.where(m.values)[0]
            nxt = 0
            for i in idx:  # 손절 없이 60거래일 보유
                if i < nxt or i + 60 >= len(c):
                    continue
                rows.append(dict(date=d.index[i], ret=c[i + 60] / c[i] - 1 - 0.002, R=np.nan, hold=60,
                                 status="closed", test=name, ticker=t, model="HOLD60"))
                nxt = i + 61
    r = pd.DataFrame(rows)
    r = r[r.status == "closed"]
    order = ["전체(필터 없음)", "섹터 RSI ≤ 40", "섹터 RSI 40~60", "섹터 RSI ≥ 60"]
    L = [f"# 섹터 평균 RSI 매수 필터 시험 ({pd.Timestamp.now():%Y-%m-%d})", "",
         "섹터별 종목 수: " + ", ".join(f"{s} {n}" for s, n in sorted(cnt.items(), key=lambda x: -x[1])), ""]
    for model, title in (("LIVE", "실전 손절 규칙"), ("HOLD60", "손절 없이 60거래일 보유")):
        x = r[r.model == model]
        tab = {}
        for k in order:
            y = x[x.test == k]
            if model == "LIVE":
                s_all, s_te = stats(y), stats(y[y.date >= SPLIT])
                tab[k] = dict(매매=s_all["매매"], 승률=s_all.get("승률"), 평균R=s_all.get("평균R"), R구간95=s_all.get("R구간95"),
                              평균수익률=s_all.get("평균수익률"), 검증_평균R=s_te.get("평균R"), 검증_매매=s_te["매매"])
            else:
                te = y[y.date >= SPLIT]
                tab[k] = dict(매매=len(y), 승률=round(100 * (y.ret > 0).mean(), 1), 평균수익률=round(100 * y.ret.mean(), 2),
                              하위5퍼=round(100 * y.ret.quantile(.05), 1), 검증_평균수익률=round(100 * te.ret.mean(), 2), 검증_매매=len(te))
        L += [f"## 1. 종목 매매 — {title}", "", pd.DataFrame(tab).T.to_markdown(), ""]

    # 2) 지수 매매 — 기술 섹터 평균 RSI
    q = load_csv(os.path.join(folder, "QQQ.csv.gz")).close
    tech = sec.get("Technology").reindex(q.index)
    pos, inn = [], False
    for v in tech.values:
        if not inn and v <= 40:
            inn = True
        elif inn and v >= 60:
            inn = False
        pos.append(inn)
    pos = pd.Series(pos, index=q.index).shift(1, fill_value=False)
    rq = q.pct_change().fillna(0)
    one = pd.Series(1.0, index=q.index)
    st = "2011-01-01"
    M = {"QQQ 보유": one, "QQQ: 기술 RSI ≤40 매수 → ≥60 현금": one.where(pos, 0),
         "3배 보유": one * 3, "3배: 기술 RSI ≤40 매수 → ≥60 현금": (one * 3).where(pos, 0)}
    t2 = pd.DataFrame({k: perf(run(rq[st:], v[st:])) for k, v in M.items()}).T
    L += ["## 2. 지수 매매 — 기술 섹터 평균 RSI (2011~, 비용 반영)", "", t2.to_markdown(), "",
          f"- 보유 비율 {100 * pos[st:].mean():.0f}%, 매수 횟수 {(pos[st:].astype(int).diff() == 1).sum()}",
          f"- 현재(데이터 마지막 날 {tech.dropna().index[-1]:%Y-%m-%d}) 기술 섹터 평균 RSI {tech.dropna().iloc[-1]:.1f}", ""]
    now = {s: round(v.dropna().iloc[-1], 1) for s, v in sec.items()}
    L += ["## 현재 섹터 평균 RSI", "", ", ".join(f"{s} {v}" for s, v in sorted(now.items(), key=lambda x: x[1])), ""]
    txt = "\n".join(L)
    print(txt)
    if out:
        open(out, "w").write(txt)


if __name__ == "__main__":
    main()
