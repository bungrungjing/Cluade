"""우위 찾기 2차 — 사전 등록 시험 (2026-11-08, 실행 전에 가설·합격 기준을 적음, 한 번만 실행)

이미 시험한 것(제외): 12-1 모멘텀·52주 고점·RSI2·주간 되돌림(tournament), 200일선·변동성 목표(public_models),
                     지지선 지정가(weak), 하향식(topdown), 차트 등급, 시장 폭 공포 등급(fear_grade_oos)
데이터: data 갈래 daily/ 약 1,090종목 + QQQ·SPY·VIX, 2010~2026-11. 생존 편향 있음(현재 상장 종목만).
구간 재현: 2010~17 / 2018~26 두 구간 모두 같은 방향이어야 함.
관문(개별 종목): 주가 ≥5, 20일 평균 거래대금 ≥4천만 달러

H1 계절성(SPY·QQQ)
  (a) 11~4월 vs 5~10월 월평균 수익률  (b) 월말·월초(마지막 거래일 + 첫 3거래일) vs 나머지 날 일평균
  합격: 차이의 95% 구간이 0 위 + 두 구간 모두 같은 방향 (SPY·QQQ 둘 다)
H2 VIX 급등 후 QQQ: VIX 종가가 30 이상이 된 첫날(직전 20일 30 미만) 매수, 20·60거래일 뒤 수익률 vs 전체 평균
  합격: 두 구간 모두 전체 평균보다 높음 (사건 수 함께 적음, 통계 검정은 사건 수가 적어 참고만)
H3 거래량 동반 갭 상승(실적 발표 대리): 시가 ≥ 전일 종가 ×1.05, 거래량 ≥ 20일 평균 ×3, 종가가 당일 범위 위쪽 절반
  다음 날 시가 매수 → 20·60거래일 뒤 수익률 − 같은 기간 SPY 수익률(초과수익)
  합격: 60일 초과수익 95% 구간이 0 위 + 두 구간 모두 플러스
H3b 거래량 동반 갭 하락(시가 ≤ ×0.95, 거래량 ×3, 종가 아래쪽 절반): 초과수익 95% 구간이 0 아래 + 두 구간 모두 마이너스면
  '보유 종목 매도 경고' 합격
H4 저변동성: 매월 말 관문 통과 종목 중 60일 변동성 하위 20% 균등 보유 1개월 vs 관문 통과 전체 균등
  합격: 샤프 비율이 두 구간 모두 높고 전체 최대 낙폭이 더 작음
H5 업종 모멘텀: 매월 말 업종(industry, 관문 통과 5종목 이상) 평균 6개월 수익률 상위 3개 업종의 종목 균등 보유 1개월 vs 전체 균등
  합격: 연복리가 두 구간 모두 높고 최대 낙폭이 전체의 1.5배 이내
H6 계좌 전략(사용자 맞춤, QQQ 일간 기준, 3배는 QQQ ×3 − 연 3% 비용 근사)
  A QQQ 보유 / B 3배 보유 / C 3배 + QQQ 200일선 필터(아래면 현금)
  D QQQ 보유 + 시장 폭 ≤30%가 된 날 다음 날부터 60거래일 3배로 전환 후 QQQ 복귀
  E QQQ 보유 + VIX 30 이상 첫날 다음 날부터 60거래일 3배 후 QQQ 복귀
  합격(D·E): 연복리가 두 구간 모두 A보다 높고 전체 최대 낙폭이 A의 1.5배 이내
"""
import os, subprocess, io
import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "edge_hunt_2026-11-08.md")
lines = []


def say(s=""):
    print(s); lines.append(s)


def rd(p):
    raw = subprocess.run(["git", "-C", REPO, "show", f"origin/data:{p}"], capture_output=True).stdout
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def half(idx):
    return np.where(pd.DatetimeIndex(idx).year <= 2017, "2010~17", "2018~26")


def ci(x):
    x = pd.Series(x).dropna(); m = x.mean(); se = x.std() / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se, len(x)


def stats(r, ppy):
    eq = (1 + r).cumprod(); yrs = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; mdd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(ppy)
    return cagr, mdd, sh


def main():
    names = [n for n in subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", "origin/data"],
                                       capture_output=True, text=True).stdout.split() if n.startswith("daily/")]
    uni = pd.read_csv(os.path.join(REPO, "validation", "universe.csv")).set_index("ticker")
    spy, qqq, vix = rd("daily/SPY.csv.gz"), rd("daily/QQQ.csv.gz"), rd("daily/_VIX.csv.gz")
    stocks = {}
    for p in names:
        t = os.path.basename(p).split(".")[0]
        if t in ("SPY", "QQQ", "_VIX"):
            continue
        try:
            df = rd(p)
        except Exception:
            continue
        if len(df) >= 300:
            stocks[t] = df
    say(f"# 우위 찾기 2차 — 사전 등록 시험 (2026-11-08)\n\n종목 {len(stocks)}개 + SPY·QQQ·VIX, 생존 편향 있음. 가설·합격 기준은 validation/edge_hunt.py 맨 위.\n")

    # H1
    say("## H1 계절성")
    for nm, d in (("SPY", spy), ("QQQ", qqq)):
        m = d.close.resample("ME").last().pct_change().dropna()
        win = m.index.month.isin([11, 12, 1, 2, 3, 4])
        a, b = m[win], m[~win]
        diff = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        hv = [a[half(a.index) == h].mean() - b[half(b.index) == h].mean() for h in ("2010~17", "2018~26")]
        say(f"- {nm} (a) 11~4월 {a.mean()*100:.2f}% vs 5~10월 {b.mean()*100:.2f}% / 차이 {diff*100:+.2f}%p (95% {(diff-1.96*se)*100:+.2f}~{(diff+1.96*se)*100:+.2f}) / 구간별 {hv[0]*100:+.2f}, {hv[1]*100:+.2f}")
        r = d.close.pct_change().dropna()
        mo = r.index.to_period("M")
        pos = r.groupby(mo).cumcount(); neg = r.groupby(mo).cumcount(ascending=False)
        tom = (pos <= 2) | (neg == 0)
        a, b = r[tom], r[~tom]
        diff = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        hv = [a[half(a.index) == h].mean() - b[half(b.index) == h].mean() for h in ("2010~17", "2018~26")]
        say(f"- {nm} (b) 월말·월초 {a.mean()*100:.3f}%/일 vs 나머지 {b.mean()*100:.3f}%/일 / 차이 {diff*100:+.3f}%p (95% {(diff-1.96*se)*100:+.3f}~{(diff+1.96*se)*100:+.3f}) / 구간별 {hv[0]*100:+.3f}, {hv[1]*100:+.3f}")

    # H2
    say("\n## H2 VIX 30 이상 첫날 → QQQ")
    v = vix.close.reindex(qqq.index).ffill()
    hi = v >= 30
    first = hi & ~hi.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    q = qqq.close
    f20, f60 = q.shift(-20) / q - 1, q.shift(-60) / q - 1
    ev = first[first].index
    say(f"사건 {len(ev)}회: " + ", ".join(d.strftime("%Y-%m") for d in ev))
    for h in ("2010~17", "2018~26", "전체"):
        sel = ev if h == "전체" else ev[half(ev) == h]
        base = q.index if h == "전체" else q.index[half(q.index) == h]
        say(f"- {h}: 사건 {len(sel)}회 20일 {f20[sel].mean()*100:+.2f}% / 60일 {f60[sel].mean()*100:+.2f}% (전체 평균 20일 {f20[base].mean()*100:+.2f}% / 60일 {f60[base].mean()*100:+.2f}%), 60일 손실 비율 {(f60[sel]<0).mean()*100:.0f}%")

    # H3
    say("\n## H3 거래량 동반 갭 (다음 날 시가 진입, SPY 대비 초과수익)")
    sc = spy.close
    rows = []
    for t, d in stocks.items():
        c, o, h, l, vol = d.close, d.open, d.high, d.low, d.volume
        v20 = vol.rolling(20).mean().shift(1); dv = (c * vol).rolling(20).mean().shift(1) / 1e6
        gate = (c >= 5) & (dv >= 40)
        mid = (h + l) / 2
        up = gate & (o >= c.shift() * 1.05) & (vol >= 3 * v20) & (c >= mid)
        dn = gate & (o <= c.shift() * 0.95) & (vol >= 3 * v20) & (c <= mid)
        e = o.shift(-1)
        s = sc.reindex(d.index).ffill(); se_ = spy.open.reindex(d.index).ffill().shift(-1)
        for k, n in (("20", 21), ("60", 61)):
            d[f"x{k}"] = (c.shift(-n + 1) / e - 1) - (s.shift(-n + 1) / se_ - 1)
        for kind, m in (("갭상승", up), ("갭하락", dn)):
            x = d.loc[m, ["x20", "x60"]].dropna()
            x["kind"] = kind; rows.append(x)
    g = pd.concat(rows)
    for kind in ("갭상승", "갭하락"):
        x = g[g.kind == kind]
        for k in ("x20", "x60"):
            m_, lo, hi_, n = ci(x[k])
            hv = x.groupby(half(x.index))[k].mean()
            say(f"- {kind} {k[1:]}일 초과수익 {m_*100:+.2f}% (95% {lo*100:+.2f}~{hi_*100:+.2f}, {n}건) / 2010~17 {hv.get('2010~17',np.nan)*100:+.2f}% · 2018~26 {hv.get('2018~26',np.nan)*100:+.2f}% / 승률(SPY 초과) {(x[k]>0).mean()*100:.0f}%")

    # 월간 패널 (H4·H5)
    close = pd.DataFrame({t: d.close for t, d in stocks.items()})
    dvol = pd.DataFrame({t: (d.close * d.volume).rolling(20).mean() / 1e6 for t, d in stocks.items()})
    vol60 = close.pct_change().rolling(60).std()
    ret6 = close / close.shift(126) - 1
    me = close.resample("ME").last().index
    me = [close.index[close.index <= m][-1] for m in me if (close.index <= m).any()]
    me = pd.DatetimeIndex(sorted(set(me)))
    cm = close.loc[me]; fwd = cm.shift(-1) / cm - 1
    gate = (cm >= 5) & (dvol.loc[me] >= 40)
    allr, lowr, indr = [], [], []
    ind = uni.industry.reindex(close.columns)
    for i, dt in enumerate(me[:-1]):
        gk = gate.loc[dt] & fwd.loc[dt].notna()
        if gk.sum() < 100 or dt.year < 2011:
            continue
        f = fwd.loc[dt][gk].clip(-0.9, 3)
        allr.append((dt, f.mean()))
        vv = vol60.loc[dt][gk].dropna()
        low = vv[vv <= vv.quantile(0.2)].index
        lowr.append((dt, f[low].mean()))
        r6 = ret6.loc[dt][gk].dropna()
        grp = r6.groupby(ind[r6.index]).agg(["mean", "size"])
        grp = grp[grp["size"] >= 5].sort_values("mean", ascending=False)
        top = grp.index[:3]
        mem = r6.index[ind[r6.index].isin(top)]
        indr.append((dt, f[mem].mean()))
    def ser(x):
        return pd.Series(dict(x)).sort_index()
    A, L, I = ser(allr), ser(lowr), ser(indr)
    say("\n## H4 저변동성 하위 20% / H5 업종 모멘텀 상위 3 (월 리밸런싱, 비용 미반영)")
    say("| 포트폴리오 | 연복리 | 최대낙폭 | 샤프 | 플러스 달 | 2010~17 연복리/샤프 | 2018~26 연복리/샤프 |")
    say("|---|---|---|---|---|---|---|")
    for nm, s in (("관문 통과 전체 균등", A), ("H4 저변동성 하위 20%", L), ("H5 업종 모멘텀 상위 3", I)):
        c_, m_, sh = stats(s, 12)
        hv = [stats(s[half(s.index) == h], 12) for h in ("2010~17", "2018~26")]
        say(f"| {nm} | {c_*100:.1f}% | {m_*100:.1f}% | {sh:.2f} | {(s>0).mean()*100:.0f}% | {hv[0][0]*100:.1f}% / {hv[0][2]:.2f} | {hv[1][0]*100:.1f}% / {hv[1][2]:.2f} |")

    # H6
    say("\n## H6 계좌 전략 (QQQ 일간, 3배 = ×3 − 연 3%)")
    above = pd.DataFrame({t: (d.close > d.close.rolling(50).mean()).where(d.close.rolling(50).count() >= 50) for t, d in stocks.items()})
    br = above.mean(axis=1)[above.count(axis=1) >= 200].reindex(qqq.index).ffill()
    qr = qqq.close.pct_change().fillna(0); l3 = 3 * qr - 0.03 / 252
    sma200 = qqq.close.rolling(200).mean()
    def window(trig, n=60):
        on = pd.Series(False, index=qqq.index); idx = np.where(trig.shift(1, fill_value=False).values)[0]
        for i in idx:
            on.iloc[i:i + n] = True
        return on
    f30 = br <= 0.30
    brfirst = f30 & ~f30.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    strat = {"A QQQ 보유": qr, "B 3배 보유": l3,
             "C 3배 + 200일선": l3.where((qqq.close > sma200).shift(1, fill_value=False), 0.0),
             "D QQQ + 시장 폭 ≤30% 때 60일 3배": qr.where(~window(brfirst), l3),
             "E QQQ + VIX 30↑ 때 60일 3배": qr.where(~window(first), l3)}
    st = qqq.index >= "2010-04-01"
    say("| 전략 | 연복리 | 최대낙폭 | 최악 연도 | 플러스 달 | 2010~17 연복리 | 2018~26 연복리 |")
    say("|---|---|---|---|---|---|---|")
    for nm, r in strat.items():
        r = r[st]; c_, m_, _ = stats(r, 252)
        yr = (1 + r).groupby(r.index.year).prod() - 1
        mo = (1 + r).groupby(r.index.to_period("M")).prod() - 1
        hv = [stats(r[half(r.index) == h], 252)[0] for h in ("2010~17", "2018~26")]
        say(f"| {nm} | {c_*100:.1f}% | {m_*100:.1f}% | {yr.min()*100:.1f}% | {(mo>0).mean()*100:.0f}% | {hv[0]*100:.1f}% | {hv[1]*100:.1f}% |")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
