"""약세 구간 지지선 지정가 매수 — 사전에 정한 규칙 그대로 한 번만 시험 (2026-11-01 사용자 승인)

사용법: python weak_bt.py <일봉 폴더(*.csv.gz)> <결과 폴더>

규칙(결과를 보기 전에 고정, 바꾸지 않음):
 - 언제: 시장 폭(유니버스 중 50일선 위 비율) ≤ 40%  또는  SPY 종가 < SPY 200일선
 - 무엇: 1단계 관문(주가 5달러↑, ATR% 1~4, 20일 거래대금 4,000만 달러↑) + 종가 > 200일선
 - 주문: 오늘 종가 기준 지정가 L = min(20일 VWAP, 50일선). 오늘 종가가 L 위일 때만 다음 날 주문
         다음 날 저가 ≤ L 이면 체결(시가가 L 아래로 열리면 시가 체결), 아니면 취소(매일 다시 계산)
 - 손절: 체결가 − max(7%, 2ATR) / 체결 당일 저가가 손절 아래면 손절로 처리(보수적)
 - 청산 A: 실전 규칙(+1R 도달 시 손절을 체결가+0.3R로, 목표 3R, 최대 60거래일)
 - 청산 B: A와 같되 목표 = 신호일 60일 고점(체결가+1R보다 낮으면 +1R)
 - 종목당 한 번에 한 매매, 왕복 비용 0.2%
통과 기준: 같은 약세 구간 '아무 날 진입'(LIVE, 신호일 종가 진입)보다 평균 R이 높고 95% 구간 하한도 그 평균보다 높을 것.
          검증 구간(2022~)에서도 같은 결론일 것.
"""
import glob, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE, FEE  # noqa: E402
from full_bt import stats, ci  # noqa: E402

SPLIT = pd.Timestamp("2022-01-01")


def sim_limit(d, setup, limit, tgt_price=None, be_at=1.0, lock=0.3, maxhold=60):
    o_, h_, l_, c_, a_ = (d[x].values for x in ("open", "high", "low", "close", "atr"))
    n, out, nxt = len(d), [], 0
    for i in np.where(setup.values)[0]:
        f = i + 1
        if i < nxt or f >= n:
            continue
        L = limit[i]
        if not (l_[f] <= L):
            continue                     # 미체결 → 취소
        e = min(o_[f], L)
        risk = max(0.07 * e, 2 * a_[i])
        stop = e - risk
        if tgt_price is None:
            tgt = e + 3 * risk
        else:
            tgt = max(tgt_price[i], e + risk)
        moved, res, why, j = False, None, "", f
        if l_[f] <= stop:
            res, why = -1.0, "손절(체결일)"
        else:
            hh = max(e, c_[f])
            for j in range(f + 1, min(n, f + 1 + maxhold)):
                if l_[j] <= stop:
                    res, why = (min(stop, o_[j]) - e) / risk, ("이익보호" if moved else "손절"); break
                if h_[j] >= tgt:
                    res, why = (max(tgt, o_[j]) - e) / risk, "목표"; break
                hh = max(hh, h_[j])
                if not moved and hh >= e + be_at * risk:
                    stop, moved = max(stop, e + lock * risk), True
        status = "closed"
        if res is None:
            res = (c_[j] - e) / risk
            if j >= n - 1 and j - f < maxhold:
                status, why = "open", "보유중"
            else:
                why = "기간만료"
        res -= FEE * e / risk
        out.append(dict(date=d.index[i], R=res, ret=res * risk / e, why=why, hold=j - f, status=status,
                        tgtR=(tgt - e) / risk))
        nxt = j + 1
    return out


def main():
    folder, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    raw = {os.path.basename(f)[:-7]: load_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz")))}
    spy = raw.pop("SPY"); raw.pop("QQQ", None); raw.pop("_VIX", None)
    ind = {}
    for t, df in raw.items():
        if len(df) < 260:
            continue
        d = indicators(df)
        d["sma200"] = d.close.rolling(200).mean()
        d["hi60"] = d.high.rolling(60).max()
        d["limit"] = np.minimum(d.vwap20, d.sma50)
        ind[t] = d.iloc[200:]
    breadth = pd.DataFrame({t: d.above50 for t, d in ind.items()}).mean(axis=1) * 100
    spy_below = spy.close < spy.close.rolling(200).mean()
    print(f"지표 {len(ind)}종목 {time.time() - t0:.0f}초", flush=True)

    rows = []
    for t, d in ind.items():
        weak = ((breadth.reindex(d.index) <= 40) | spy_below.reindex(d.index).fillna(False)).astype(bool)
        base = weak & d.gate
        setup = base & (d.close > d.sma200) & (d.close > d.limit)
        for name, kw in (("지지선 지정가 / 청산 A(3R)", {}), ("지지선 지정가 / 청산 B(60일 고점)", {"tgt_price": d.hi60.values})):
            for tr in sim_limit(d, setup, d.limit.values, **kw):
                tr.update(ticker=t, test=name); rows.append(tr)
        for name, m in (("약세 구간 아무 날", base), ("약세 구간 아무 날 + 200일선 위", base & (d.close > d.sma200))):
            for tr in sim(d, m, **LIVE):
                tr.update(ticker=t, test=name); rows.append(tr)
    r = pd.DataFrame(rows)
    r = r[r.status == "closed"]
    print(f"시뮬레이션 {len(r)}건 {time.time() - t0:.0f}초", flush=True)

    order = ["지지선 지정가 / 청산 A(3R)", "지지선 지정가 / 청산 B(60일 고점)", "약세 구간 아무 날", "약세 구간 아무 날 + 200일선 위"]
    L = [f"# 약세 구간 지지선 지정가 매수 — 사전 등록 시험 ({pd.Timestamp.now():%Y-%m-%d})", "",
         f"- 종목 {len(ind)}개, 약세 구간 거래일 비율 {100 * ((breadth <= 40) | spy_below.reindex(breadth.index).fillna(False)).mean():.0f}%", ""]
    for title, sel in (("전체", r), ("설계 구간(2011~2021)", r[r.date < SPLIT]), ("검증 구간(2022~)", r[r.date >= SPLIT])):
        L += [f"## {title}", "", pd.DataFrame({k: stats(sel[sel.test == k]) for k in order}).T.to_markdown(), ""]
    for k in order[:2]:
        x = r[r.test == k]
        L += [f"### {k} 청산 사유", "", x.why.value_counts(normalize=True).mul(100).round(1).to_frame("비율%").to_markdown(), ""]
    b = r[r.test == "지지선 지정가 / 청산 B(60일 고점)"]
    if len(b):
        L += [f"- 청산 B 목표의 평균 크기: {b.tgtR.mean():.2f}R (중앙값 {b.tgtR.median():.2f}R)", ""]
    # 판정
    L += ["## 판정 (통과 기준: 아무 날 평균R보다 높고, 95% 구간 하한도 그 평균보다 높을 것 — 전체·검증 둘 다)", ""]
    for k in order[:2]:
        verdict = []
        for title, sel in (("전체", r), ("검증", r[r.date >= SPLIT])):
            x, base_ = sel[sel.test == k].R.values, sel[sel.test == "약세 구간 아무 날"].R.mean()
            lo, _ = ci(x)
            ok = len(x) >= 20 and x.mean() > base_ and lo > base_
            verdict.append(f"{title}: 평균 {x.mean():+.3f}R, 하한 {lo:+.3f} vs 기준 {base_:+.3f} → {'통과' if ok else '미통과'}")
        L.append(f"- {k}: " + " / ".join(verdict))
    L += ["", "## 한계", "",
          "- 생존 편향: 지금 살아 있는 종목만 포함. 약세 구간에서 가장 크게 작용(그때 무너져 사라진 종목이 빠짐).",
          "- 지정가 체결은 일봉 저가로 판정 — 실제로는 저가에 딱 닿을 때 체결이 안 될 수도 있다.", ""]
    with open(os.path.join(outdir, "weak_summary.md"), "w") as f:
        f.write("\n".join(L))
    r.to_csv(os.path.join(outdir, "weak_trades.csv.gz"), index=False, compression="gzip")
    print("\n".join(L))


if __name__ == "__main__":
    main()
