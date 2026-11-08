"""전략 실험실 1차 — 수익률을 높일 새 방법 일괄 사전 등록 시험 (2026-11-08, 실행 전 작성, 한 번만 실행)

데이터: data 갈래 1,087종목 + QQQ·SPY. 생존 편향 큼(현재 상장 종목만 → 특히 종목 모멘텀 결과가 부풀 수 있음).
구간: 설계 2011~2018 / 검증 2019~2026-11. 레버리지는 QQQ 일간 × 배수 − 연 (배수−1)% 비용 근사, 현금 이자 0.
종목 포트폴리오: 매월 말 선정, 다음 달 균등 보유, 교체분에 편도 0.2% 비용. 관문: 주가 ≥5, 20일 평균 거래대금 ≥4천만 달러.

후보 (총 14개)
 [종목 모멘텀]
  P1  12-1 모멘텀 상위 10
  P2  12-1 모멘텀 상위 10 + SPY 200일선 필터(아래면 현금)
  P3  변동성 조정 모멘텀(12-1 수익 / 126일 변동성) 상위 10 + SPY 200일선
  P4  추세 템플릿(종가>50>150>200일선, 200일선 상승, 52주 고점 −25% 이내, 52주 저점 +30% 이상) 통과 종목 중 12개월 수익 상위 10 + SPY 200일선
  P5  12-1 모멘텀 상위 5 + SPY 200일선 (집중)
  P6  6개월 모멘텀 상위 10 + SPY 200일선
 [지수]
  I1  QQQ 2배 절대 모멘텀: 매월 말 QQQ 12개월 수익 > 0 이면 다음 달 2배, 아니면 현금
  I2  QQQ 1배 기본 + RSI(2)<10 & 200일선 위면 2배로 올리고 종가 > 5일선이면 1배 복귀 (지수 단기 되돌림)
  I3  QQQ 1배 기본 + 시장 폭 급반등(20거래일 안에 ≤30% → ≥60%)이면 다음 날부터 120거래일 2배
  I4  QQQ 2배 + 200일선 필터 + RSI(2)<10 되돌림 예외(200일선 아래여도 RSI(2)<10이면 5일선 회복까지 2배)
 [혼합]
  C1  50% H(2배+200일선+공포 60일 예외) + 50% P2, 매월 재조정
  C2  50% QQQ 2배 보유 + 50% P2, 매월 재조정
  C3  50% QQQ 1배 + 50% P2, 매월 재조정
  C4  P2인데 SPY 200일선 아래 구간에는 현금 대신 QQQ 1배(공포 때 지수 보유)
비교 기준: QQQ 보유 / QQQ 2배 보유 / H

합격 A(실전 후보): 설계·검증 두 구간 모두 연복리 > QQQ, 전체 최대 낙폭 ≥ −52.7%(QQQ의 1.5배)
합격 B(고수익): 두 구간 모두 연복리 > QQQ 2배 보유, 최대 낙폭이 QQQ 2배 보유보다 나쁘지 않음
14개를 시험하므로 하나쯤은 우연히 합격할 수 있음 — 합격해도 '전향 검증(앞으로의 실제 기록)' 전까지 확정 아님
"""
import os, subprocess, io
import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "strat_lab_2026-11-08.md")


def rd(p):
    raw = subprocess.run(["git", "-C", REPO, "show", f"origin/data:{p}"], capture_output=True).stdout
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def per(idx):
    y = pd.DatetimeIndex(idx).year
    return np.where(y <= 2018, "설계", "검증")


def stat_daily(r):
    r = r.dropna(); eq = (1 + r).cumprod(); n = len(r)
    cagr = eq.iloc[-1] ** (252 / n) - 1 if eq.iloc[-1] > 0 else -1
    mdd = (eq / eq.cummax() - 1).min()
    yr = (1 + r).groupby(r.index.year).prod() - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    return cagr, mdd, yr.min(), sh


def main():
    names = [n for n in subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", "origin/data"],
                                       capture_output=True, text=True).stdout.split() if n.startswith("daily/")]
    qqq, spy = rd("daily/QQQ.csv.gz"), rd("daily/SPY.csv.gz")
    C, V = {}, {}
    for p in names:
        t = os.path.basename(p).split(".")[0]
        if t in ("SPY", "QQQ", "_VIX"):
            continue
        try:
            d = rd(p)
        except Exception:
            continue
        if len(d) >= 300:
            C[t] = d.close; V[t] = d.volume
    close = pd.DataFrame(C).reindex(qqq.index); vol = pd.DataFrame(V).reindex(qqq.index)
    dret = close.pct_change(fill_method=None)
    dvol = (close * vol).rolling(20).mean() / 1e6
    idx = qqq.index
    q = qqq.close; qr = q.pct_change().fillna(0)
    lev = lambda k: k * qr - (k - 1) * 0.01 / 252
    s = spy.close.reindex(idx).ffill(); spy_up = s > s.rolling(200).mean()
    q_up = q > q.rolling(200).mean()
    # 시장 폭
    sma50 = close.rolling(50).mean()
    ab = (close > sma50).where(sma50.notna())
    br = ab.mean(axis=1).where(ab.count(axis=1) >= 200)
    # 월말
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last().values
    me = pd.DatetimeIndex(me)
    sma150, sma200 = close.rolling(150).mean(), close.rolling(200).mean()
    hi252, lo252 = close.rolling(252, min_periods=200).max(), close.rolling(252, min_periods=200).min()
    mom121 = close.shift(21) / close.shift(252) - 1
    mom12 = close / close.shift(252) - 1
    mom6 = close / close.shift(126) - 1
    vol126 = dret.rolling(126).std()
    tmpl = (close > sma50) & (sma50 > sma150) & (sma150 > sma200) & (sma200 > sma200.shift(21)) & (close >= 0.75 * hi252) & (close >= 1.3 * lo252)
    gate = (close >= 5) & (dvol >= 40)

    def stock_port(score, n, filt=None, alt=None):
        """매월 말 score 상위 n 종목 → 다음 달 일간 균등(매일 재조정 근사 대신 월초 균등 후 보유: 일간 수익 = 종목 누적가치 가중)"""
        out = pd.Series(0.0, index=idx); prev = set()
        for i in range(len(me) - 1):
            t0, t1 = me[i], me[i + 1]
            days = idx[(idx > t0) & (idx <= t1)]
            if t0.year < 2010 or len(days) == 0:
                continue
            if filt is not None and not bool(filt.loc[t0]):
                if alt is not None:
                    out.loc[days] = alt.loc[days]
                if prev:
                    out.loc[days[0]] -= 0.002  # 전량 매도 비용
                prev = set(); continue
            sc = score.loc[t0][gate.loc[t0].fillna(False)].dropna()
            sc = sc[np.isfinite(sc)]
            if len(sc) < n:
                continue
            pick = list(sc.sort_values(ascending=False).index[:n])
            rr = dret.loc[days, pick].fillna(0)
            val = (1 + rr).cumprod()
            port = val.mean(axis=1)
            dr = port.pct_change(); dr.iloc[0] = port.iloc[0] - 1
            turn = len(set(pick) - prev) / n if prev else 1.0
            dr.iloc[0] -= 0.002 * turn * (2 if prev else 1)
            out.loc[days] = dr.values; prev = set(pick)
        return out

    R = {}
    R["QQQ 보유"] = qr
    R["QQQ 2배 보유"] = lev(2)
    # H
    f30 = br <= 0.30
    ev = f30 & ~f30.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    on = pd.Series(False, index=idx)
    for i in np.where(ev.shift(1, fill_value=False).values)[0]:
        on.iloc[i:i + 60] = True
    up1 = q_up.shift(1, fill_value=False)
    R["H 2배+200일선+공포 예외"] = lev(2).where(up1 | on, 0.0)
    spy_f = spy_up  # 월말 판정
    R["P1 모멘텀12-1 상위10"] = stock_port(mom121, 10)
    R["P2 P1 + SPY 200일선"] = stock_port(mom121, 10, spy_f)
    R["P3 변동성조정 모멘텀 + 필터"] = stock_port(mom121 / (vol126 * np.sqrt(252)), 10, spy_f)
    R["P4 추세 템플릿 상위10 + 필터"] = stock_port(mom12.where(tmpl), 10, spy_f)
    R["P5 모멘텀 상위5 + 필터"] = stock_port(mom121, 5, spy_f)
    R["P6 6개월 모멘텀 상위10 + 필터"] = stock_port(mom6, 10, spy_f)
    # I1
    q12 = q / q.shift(252) - 1
    m_on = pd.Series(np.nan, index=idx); m_on.loc[me] = (q12.loc[me] > 0).astype(float)
    m_on = m_on.ffill().shift(1).fillna(0).astype(bool)
    R["I1 QQQ 2배 절대 모멘텀"] = lev(2).where(m_on, 0.0)
    # RSI2
    d = q.diff(); up_ = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean(); dn_ = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi2 = 100 - 100 / (1 + up_ / dn_)
    sma5 = q.rolling(5).mean()
    def rsi_state(need_up):
        st = np.zeros(len(idx), bool); cur = False
        for i in range(len(idx)):
            if not cur and rsi2.iloc[i] < 10 and (q_up.iloc[i] or not need_up):
                cur = True
            elif cur and q.iloc[i] > sma5.iloc[i]:
                cur = False
            st[i] = cur
        return pd.Series(st, index=idx).shift(1, fill_value=False)
    sig_up = rsi_state(True); sig_any = rsi_state(False)
    R["I2 QQQ + RSI(2) 되돌림 때 2배"] = qr.where(~sig_up, lev(2))
    # I3 breadth thrust
    low20 = br.rolling(20, min_periods=1).min()
    thrust = (br >= 0.60) & (low20 <= 0.30)
    tev = thrust & ~thrust.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    on3 = pd.Series(False, index=idx)
    for i in np.where(tev.shift(1, fill_value=False).values)[0]:
        on3.iloc[i:i + 120] = True
    R["I3 QQQ + 시장 폭 급반등 때 120일 2배"] = qr.where(~on3, lev(2))
    R["I4 2배+200일선+RSI(2) 예외"] = lev(2).where(up1 | sig_any, 0.0)
    # 혼합(월 재조정 근사: 일간 50/50 가중 → 매일 재조정과 같음. 월 재조정과 차이는 작다고 보고 일간 가중 사용)
    R["C1 50% H + 50% P2"] = 0.5 * R["H 2배+200일선+공포 예외"] + 0.5 * R["P2 P1 + SPY 200일선"]
    R["C2 50% QQQ 2배 + 50% P2"] = 0.5 * lev(2) + 0.5 * R["P2 P1 + SPY 200일선"]
    R["C3 50% QQQ + 50% P2"] = 0.5 * qr + 0.5 * R["P2 P1 + SPY 200일선"]
    R["C4 P2, 필터 아래면 QQQ 1배"] = stock_port(mom121, 10, spy_f, alt=qr)

    st = idx >= "2011-01-01"
    rows = []
    base = {}
    for k, r in R.items():
        r = r[st]
        cg, md, wy, sh = stat_daily(r)
        pv = {p: stat_daily(r[per(r.index) == p])[0] for p in ("설계", "검증")}
        rows.append([k, cg, md, wy, sh, pv["설계"], pv["검증"]])
        base[k] = (cg, md, pv)
    Q, Q2 = base["QQQ 보유"], base["QQQ 2배 보유"]
    L = ["# 전략 실험실 1차 (2026-11-08, 사전 등록 14개 + 기준 3개)\n",
         "기간 2011-01 ~ 2026-11-06 / 설계 2011~2018 / 검증 2019~2026. 생존 편향 큼.\n",
         "| 전략 | 연복리 | 최대낙폭 | 최악 연도 | 샤프 | 설계 | 검증 | 합격A | 합격B |", "|---|---|---|---|---|---|---|---|---|"]
    for k, cg, md, wy, sh, a, b in rows:
        if k in ("QQQ 보유", "QQQ 2배 보유", "H 2배+200일선+공포 예외"):
            pa = pb = "기준"
        else:
            pa = "합격" if a > Q[2]["설계"] and b > Q[2]["검증"] and md >= 1.5 * Q[1] else "—"
            pb = "합격" if a > Q2[2]["설계"] and b > Q2[2]["검증"] and md >= Q2[1] else "—"
        L.append(f"| {k} | {cg*100:.1f}% | {md*100:.1f}% | {wy*100:.1f}% | {sh:.2f} | {a*100:.1f}% | {b*100:.1f}% | {pa} | {pb} |")
    # 연도별 (주요)
    keys = ["QQQ 보유", "QQQ 2배 보유", "H 2배+200일선+공포 예외", "P2 P1 + SPY 200일선", "C1 50% H + 50% P2", "C2 50% QQQ 2배 + 50% P2"]
    yr = pd.DataFrame({k: (1 + R[k][st]).groupby(idx[st].year).prod() - 1 for k in keys}) * 100
    L += ["\n연도별 수익률(%)", yr.round(0).astype(int).T.to_markdown()]
    txt = "\n".join(L)
    print(txt)
    open(OUT, "w").write(txt + "\n")
    pd.DataFrame(R).to_csv(os.environ.get("LAB_OUT", "/tmp/strat_lab_daily.csv.gz"))


if __name__ == "__main__":
    main()
