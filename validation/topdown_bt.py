"""사용자 매매 방식(하향식) 시험 — 사전 등록, 한 번만 (2026-11-02, 사용자 설명 그대로 옮김)

사용자: "1. 뉴스 분위기  2. 미래에 무조건 올라갈 테마  3. 눌렸을 때 샀다"
과거 데이터로 옮길 수 있는 형태(미래를 미리 알 수 없으므로 '지금 보이는 것'으로 근사):
 1) 큰 흐름(뉴스 분위기 근사): SPY 종가 > 200일선 그리고 VIX < 25   ※ 과거 뉴스 감정 데이터는 없어 시장 상태로 대신함
 2) 중간 흐름(오를 테마 근사): 업종(industry, 5종목 이상) 중 최근 6개월 수익률 중앙값 상위 3개 업종
    ※ '미래에 오를 테마'는 미리 알 수 없으므로 '지금 오르고 있는 테마'(테마 모멘텀)로 근사
 3) 작은 흐름(눌림): 종목 종가 > 50일선 이고, 오늘 저가가 20일선 +1% 이내로 닿고 양봉 마감
 공통 관문: 주가 5달러↑, 20일 거래대금 4,000만 달러↑
청산: LIVE(실전 손절 규칙) / HOLD60(손절 없이 60거래일). 계좌: 1만 달러, 동시 5종목.
비교: 같은 관문의 아무 날 / 2)+3)만(큰 흐름 없이) / 1)+3)만(테마 없이)
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE  # noqa: E402
from full_bt import stats  # noqa: E402
from rsi_account_bt import account  # noqa: E402
from exposure_bt import perf  # noqa: E402

SPLIT = pd.Timestamp("2022-01-01")


def main():
    folder, outp = sys.argv[1], sys.argv[2]
    uni = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.csv")).set_index("ticker")
    ind_name = uni.industry.fillna("Unknown")
    spy = load_csv(os.path.join(folder, "SPY.csv.gz")).close
    vix = load_csv(os.path.join(folder, "_VIX.csv.gz")).close
    big = ((spy > spy.rolling(200).mean()) & (vix.reindex(spy.index).ffill() < 25))
    D = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        if t not in uni.index:
            continue
        df = load_csv(f)
        if len(df) < 300:
            continue
        D[t] = indicators(df).iloc[200:]
    C = pd.DataFrame({t: d.close for t, d in D.items()})
    r6 = C / C.shift(126) - 1
    groups = ind_name[C.columns]
    big_groups = groups.value_counts()
    big_groups = big_groups[big_groups >= 5].index.drop("Unknown", errors="ignore")
    med = pd.DataFrame({g: r6[groups.index[groups == g]].median(axis=1) for g in big_groups})
    rank = med.rank(axis=1, ascending=False)
    top3 = {g: (rank[g] <= 3) for g in big_groups}

    rows = []
    for t, d in D.items():
        g = groups[t]
        theme = top3[g].reindex(d.index).fillna(False) if g in top3 else pd.Series(False, index=d.index)
        mk = big.reindex(d.index).fillna(False)
        gate = (d.close >= 5) & (d.dvol20 >= 40)
        pull = (d.close > d.sma50) & (d.low <= d.sma20 * 1.01) & (d.close >= d.sma20 * 0.99) & (d.close > d.open)
        masks = {"아무 날": gate, "하향식 전체(1+2+3)": gate & mk & theme & pull,
                 "테마+눌림(2+3, 큰 흐름 없이)": gate & theme & pull, "큰 흐름+눌림(1+3, 테마 없이)": gate & mk & pull}
        c, idx = d.close.values, d.index
        for name, m in masks.items():
            for tr in sim(d, m.astype(bool), **LIVE):
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
    order = ["아무 날", "하향식 전체(1+2+3)", "테마+눌림(2+3, 큰 흐름 없이)", "큰 흐름+눌림(1+3, 테마 없이)"]
    L = [f"# 사용자 하향식 매매 방식 시험 ({pd.Timestamp.now():%Y-%m-%d})", ""]
    x = r[r.model == "LIVE"]
    tab = {}
    for k in order:
        a, b = stats(x[x.test == k]), stats(x[(x.test == k) & (x.date >= SPLIT)])
        tab[k] = dict(매매=a["매매"], 승률=a.get("승률"), 평균R=a.get("평균R"), R구간95=a.get("R구간95"),
                      검증_매매=b["매매"], 검증_평균R=b.get("평균R"), 검증_R구간95=b.get("R구간95"))
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
            ps = pd.DataFrame([perf(account(y, mode, s)) for s in range(5)])
            rec = ps.iloc[0].to_dict(); rec["연복리"] = " / ".join(ps["연복리"])
            acc[f"{k} / {'실전 손절(위험 2%)' if mode == 'LIVE' else '60일 보유(20%씩)'}"] = rec
    L += ["## 3. 계좌 시뮬레이션 (1만 달러, 동시 5종목, 무작위 순서 5회)", "", pd.DataFrame(acc).T.to_markdown(), "",
          "- 한계: 뉴스 분위기는 시장 상태(SPY 200일선·VIX)로, '미래에 오를 테마'는 '지금 오르는 테마'로 근사. 생존 편향 있음.", ""]
    last = rank.dropna(how="all").iloc[-1].sort_values().head(5)
    L += [f"## 현재({rank.dropna(how='all').index[-1]:%Y-%m-%d}) 6개월 수익률 상위 업종", "",
          ", ".join(f"{g} ({100 * med[g].dropna().iloc[-1]:+.0f}%)" for g in last.index), ""]
    txt = "\n".join(L)
    print(txt)
    open(outp, "w").write(txt)


if __name__ == "__main__":
    main()
