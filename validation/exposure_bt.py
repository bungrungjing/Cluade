"""시장 노출·비중·손절 시험 — 사전 등록, 한 번씩만 실행 (2026-11-01 사용자 승인)

사용법: python exposure_bt.py <일봉 폴더> <full_bt 결과 trades.csv.gz> <결과 폴더>

A. 시장 노출 (2011-01 ~ 현재, 일 단위, 비용 무시):
   QQQ 보유 / QQQ 200일선 필터(종가>200일선이면 다음 날 보유, 아니면 현금) / SPY 보유
   / 3배(TQQQ 근사 = QQQ 일간 수익률×3 − 연 0.95%) 보유 / 3배 + 200일선 필터
B. 손절 효과 (아무 날 진입, 같은 진입점): 실전 손절(LIVE) vs 손절 없이 60거래일 보유
C. 계좌 시뮬레이션 (계좌 1만 달러, 동시 최대 5종목, 매매당 위험 = 계좌의 1% 또는 2%,
   종목당 투자 상한 = 계좌의 1/5 × 1.5, 진입 순서는 날짜순·같은 날은 무작위):
   v3 신호 매매 / 아무 날 매매 → 월 수익률 분포, 최대 낙폭, 월 5% 이상인 달 비율
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import load_csv  # noqa: E402


def perf(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    m = eq.resample("ME").last().pct_change().dropna()
    y = eq.resample("YE").last().pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    return dict(연복리=f"{100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1):.1f}%", 최대낙폭=f"{100 * dd:.1f}%",
                최악연도=f"{100 * y.min():.1f}%", 월평균=f"{100 * m.mean():.2f}%", 플러스달=f"{100 * (m > 0).mean():.0f}%",
                월5퍼이상=f"{100 * (m >= 0.05).mean():.0f}%", 최악달=f"{100 * m.min():.1f}%")


def part_a(folder):
    q = load_csv(os.path.join(folder, "QQQ.csv.gz")).close
    s = load_csv(os.path.join(folder, "SPY.csv.gz")).close
    q, s = q["2010-01-01":], s["2010-01-01":]
    rq, rs = q.pct_change().fillna(0), s.pct_change().fillna(0)
    on = (q > q.rolling(200).mean()).shift(1, fill_value=False)
    r3 = 3 * rq - 0.0095 / 252
    start = "2011-01-01"
    strat = {"QQQ 보유": rq, "QQQ 200일선 필터": rq.where(on, 0), "SPY 보유": rs,
             "3배 보유(TQQQ 근사)": r3, "3배 + 200일선 필터": r3.where(on, 0)}
    return pd.DataFrame({k: perf((1 + v[start:]).cumprod()) for k, v in strat.items()}).T


def part_b(trades):
    return None  # B는 아래 main에서 일봉으로 직접 계산


def portfolio(tr, risk_pct, seed=0, cap=10000.0, slots=5):
    tr = tr.sample(frac=1, random_state=seed).sort_values("date", kind="stable")
    eq, open_, curve = cap, [], {}
    events = []
    for _, x in tr.iterrows():
        d = x.date
        # 만기된 포지션 정리
        still = []
        for p in open_:
            if p["exit"] <= d:
                eq += p["pnl"]; curve[p["exit"]] = eq
            else:
                still.append(p)
        open_ = still
        if len(open_) >= slots:
            continue
        if any(p["ticker"] == x.ticker for p in open_):
            continue
        stop_pct = x.risk / x.entry
        size = min(eq * risk_pct / stop_pct, eq / slots * 1.5)
        open_.append(dict(exit=x.exit_date, pnl=size * x.ret, ticker=x.ticker))
        events.append(d)
    for p in sorted(open_, key=lambda p: p["exit"]):
        eq += p["pnl"]; curve[p["exit"]] = eq
    c = pd.Series(curve).sort_index()
    c = pd.concat([pd.Series({tr.date.min(): cap}), c]).groupby(level=0).last()
    daily = c.reindex(pd.date_range(c.index[0], c.index[-1], freq="D")).ffill()
    return daily, len(events)


def main():
    folder, tpath, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    L = [f"# 시장 노출·손절·비중 시험 ({pd.Timestamp.now():%Y-%m-%d})", ""]
    a = part_a(folder)
    L += ["## A. 시장 노출 (2011-01 ~ 현재, 비용·세금 무시)", "", a.to_markdown(), ""]

    tr = pd.read_csv(tpath, parse_dates=["date", "exit_date"])
    tr = tr[(tr.model == "LIVE") & tr.test.isin(["v3", "아무 날"])]

    # B: 손절 효과 — 아무 날 진입점에서 60거래일 그냥 보유했을 때와 비교
    rows = []
    any_ = tr[tr.test == "아무 날"]
    for t, g in any_.groupby("ticker"):
        f = os.path.join(folder, t + ".csv.gz")
        if not os.path.exists(f):
            continue
        c = load_csv(f).close
        pos = c.index.get_indexer(g.date)
        ok = (pos >= 0) & (pos + 60 < len(c))
        for p, (_, x) in zip(pos[ok], g[ok].iterrows()):
            path = c.iloc[p:p + 61].values / c.iloc[p]
            rows.append(dict(live=x.ret, hold60=path[-1] - 1 - 0.002, mdd60=path.min() - 1))
    b = pd.DataFrame(rows)
    if len(b):
        q = lambda s: f"{100 * s.mean():+.2f}% / 하위5% {100 * s.quantile(.05):+.1f}% / 상위5% {100 * s.quantile(.95):+.1f}%"
        L += ["## B. 손절 효과 (아무 날 진입 %d건, 같은 진입점)" % len(b), "",
              f"- 실전 손절 규칙: 평균 {q(b.live)}",
              f"- 손절 없이 60거래일 보유: 평균 {q(b.hold60)}",
              f"- 손절 없이 보유 중 겪는 최대 하락: 평균 {100 * b.mdd60.mean():.1f}%, −20% 이상 하락 겪은 비율 {100 * (b.mdd60 <= -0.2).mean():.1f}%",
              ""]

    # C: 계좌 시뮬레이션
    res = {}
    for test in ("v3", "아무 날"):
        x = tr[tr.test == test]
        for rp in (0.01, 0.02):
            vals = []
            for seed in range(5):
                daily, n = portfolio(x, rp, seed)
                vals.append(perf(daily))
            df = pd.DataFrame(vals)
            res[f"{test} / 매매당 위험 {int(rp * 100)}%"] = {k: f"{df[k].iloc[0]} (5회: {', '.join(df[k])})" if k == "연복리" else df[k].iloc[0] for k in df}
    L += ["## C. 계좌 시뮬레이션 (1만 달러, 동시 최대 5종목, 2010~)", "",
          "같은 날 신호가 여럿이면 무작위 순서 — 5번 돌려 연복리 범위를 함께 표시", "",
          pd.DataFrame(res).T.to_markdown(), ""]
    L += ["## 한계", "", "- 생존 편향(지금 살아 있는 종목만), 세금·환율 무시, 3배 ETF는 근사치(실제 TQQQ는 2010-02 상장).",
          "- 계좌 시뮬레이션은 청산일에 손익을 한꺼번에 반영(보유 중 평가손익 미반영) → 실제 낙폭은 더 클 수 있음.", ""]
    with open(os.path.join(outdir, "exposure_summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
