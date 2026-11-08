"""시장 공포 등급 + 관련 후보 일괄 검증 (2026-11-08, 실행 전 가설·합격 기준 사전 등록)

데이터: data 갈래 daily/ 전체(약 1,120종목)로 시장 폭(50일선 위 비율)을 계산.
        매매 시험 종목 = 이전 시험에 쓴 52종목·284종목을 뺀 나머지(처음 보는 종목). 2010~2026-11.
관문: 주가 ≥5, ATR% 1~4, 20일 평균 거래대금 ≥4천만 달러 (기존과 같음)
기본 매매: 종가 진입, 손절 max(7%,2ATR), 목표 3R, +1R 후 본전, 최대 60일, 왕복 0.2%, 종목당 한 포지션
구간: 2010~17 / 2018~26. 95% 구간은 매매 단위(같은 날 진입이 몰려 실제보다 좁을 수 있음 → 공포 '사건' 수도 같이 표시)

H1 공포 등급(매수): 시장 폭 ≤20% S / 20~30 A / 30~40 B / 40~60 C / 60~80 D / >80 F
   합격: S·A 각각 전체 95% 하한 > 기준선 평균, 두 구간 모두 기준선보다 높음, 그리고 S ≥ A ≥ B > C (대체로 단조)
H2 공포일 안에서 종목 고르기(시장 폭 ≤30%일 때만): 아래 하위 그룹 vs 나머지 공포일 매매
   (a) 정배열 50>200 (b) ADX>25 & +DI>−DI (c) 52주 고점 −25% 이내 (d) 50일선 대비 −15% 이하(깊게 눌림) (e) 63일 수익률 하위 25%(그날 관문 통과 종목 중)
   합격: 차이의 95% 구간이 0을 포함하지 않고 두 구간 모두 같은 부호
H3 지수 타이밍(TQQQ 관련): QQQ를 그날 종가에 사서 20·60거래일 뒤 수익률, 시장 폭 등급별. 합성 TQQQ(QQQ 일간 ×3 − 연 0.95%)도 같이.
   합격: ≤30% 날의 60일 뒤 평균이 전체 평균보다 높고 두 구간 모두 같은 방향. 독립 사건 수(공포 구간 시작 횟수)를 함께 적음.
H4 환희(매도·익절 쪽): 시장 폭 >80%일 진입 매매 평균R, QQQ 20·60일 뒤 수익률
   합격: 두 구간 모두 기준선보다 낮고 전체 95% 상한 < 기준선 평균
H5 공포일 청산 방식(시장 폭 ≤30% 진입만): A 현재(3R+본전+60일) / B 목표 없음·본전 없음·60일 보유·손절만 / C 3R·본전 없음·60일 / D 5R+본전+120일
   합격: A보다 평균R이 높고 차이 95% 구간이 0 위, 두 구간 모두 같은 방향 → 원칙 변경 메모 후보
"""
import sys, os, subprocess, io
import numpy as np
import pandas as pd
from numba import njit

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FEE = 0.002


@njit
def sim(c, h, l, atr, mask, tgtR, use_be, maxhold):
    n = len(c); R = np.full(n, np.nan); nxt = 0
    for i in range(n - 1):
        if not mask[i] or i < nxt:
            continue
        e = c[i]; risk = max(0.07 * e, 2 * atr[i])
        if not np.isfinite(risk):
            continue
        stop = e - risk; tgt = e + tgtR * risk if tgtR > 0 else 1e18; be = False; px = -1.0; j = i
        for j in range(i + 1, min(n, i + 1 + maxhold)):
            if l[j] <= stop:
                px = stop; break
            if h[j] >= tgt:
                px = tgt; break
            if use_be and (not be) and h[j] >= e + risk:
                stop = e; be = True
        if px < 0:
            if j < i + maxhold:
                break
            px = c[j]
        R[i] = (px - e) / risk - FEE * e / risk
        nxt = j + 1
    return R


def rd(p):
    raw = subprocess.run(["git", "-C", REPO, "show", f"origin/data:{p}"], capture_output=True).stdout
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def ind(df):
    c, h, l = df.close, df.high, df.low
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / 14, adjust=False).mean()
    return pd.DataFrame({"c": c, "h": h, "l": l, "atr": atr, "sma50": c.rolling(50).mean(), "sma200": c.rolling(200).mean(),
                         "adx": adx, "pdi": pdi, "mdi": mdi, "hi252": h.rolling(252, min_periods=120).max(),
                         "r63": c.pct_change(63), "dvol": (c * df.volume).rolling(20).mean() / 1e6})


def ci(x):
    x = np.asarray(x); m = x.mean(); se = x.std() / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se, len(x)


def tier(b):
    return np.select([b <= .2, b <= .3, b <= .4, b <= .6, b <= .8], ["S(≤20)", "A(20~30)", "B(30~40)", "C(40~60)", "D(60~80)"], "F(>80)")


def main():
    used = set(os.path.basename(f)[:-4] for f in os.listdir(sys.argv[1]))  # 52종목
    used |= set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    names = [n for n in subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", "origin/data"],
                                       capture_output=True, text=True).stdout.split() if n.startswith("daily/")]
    data = {}
    for p in names:
        t = os.path.basename(p).split(".")[0]
        if t in ("_VIX",):
            continue
        try:
            df = rd(p)
        except Exception:
            continue
        if len(df) < 300:
            continue
        data[t] = ind(df)
    qqq = data.pop("QQQ")
    above = pd.DataFrame({t: (d.c > d.sma50).where(d.sma50.notna()) for t, d in data.items()})
    breadth = above.mean(axis=1)[above.count(axis=1) >= 200]
    breadth = breadth[breadth.index >= "2010-03-01"]
    test = [t for t in data if t not in used]
    print(f"시장 폭 계산 종목 {len(data)}개 / 매매 시험(처음 보는) 종목 {len(test)}개 / 기간 {breadth.index[0].date()}~{breadth.index[-1].date()}")
    # 공포 사건 수: 시장 폭이 30% 이하로 처음 내려간 날(직전 20일 동안 30% 초과)
    f30 = breadth <= .3
    ev = f30 & ~f30.rolling(20, min_periods=1).max().shift(fill_value=0).astype(bool)
    print("시장 폭 ≤30% 독립 사건:", int(ev.sum()), "회 —", ", ".join(d.strftime("%Y-%m") for d in breadth.index[ev]))
    rows = []
    exits = {"A 현재(3R+본전+60)": (3.0, True, 60), "B 손절만·60일": (0.0, False, 60), "C 3R·본전없음·60": (3.0, False, 60), "D 5R+본전+120": (5.0, True, 120)}
    for t in test:
        d = data[t].reindex(breadth.index).dropna(subset=["c"])
        if len(d) < 260:
            continue
        b = breadth.reindex(d.index).values
        gate = ((d.c >= 5) & (d.atr / d.c).between(.01, .04) & (d.dvol >= 40) & d.sma200.notna()).values
        arr = (d.c.values, d.h.values, d.l.values, d.atr.values)
        tr = tier(b); fear = gate & (b <= .3)
        groups = {"기준선": gate}
        for g in np.unique(tr):
            groups[g] = gate & (tr == g)
        sub = {"(a)정배열": (d.sma50 > d.sma200).values, "(b)ADX상승": ((d.adx > 25) & (d.pdi > d.mdi)).values,
               "(c)52주고점−25%이내": (d.c >= .75 * d.hi252).values, "(d)50일선−15%이하": (d.c <= .85 * d.sma50).values}
        for k, m in sub.items():
            groups["공포∩" + k] = fear & m; groups["공포∩not" + k] = fear & ~m
        for k, m in groups.items():
            R = sim(*arr, m, 3.0, True, 60)
            ok = ~np.isnan(R)
            rows += [(k, t, dt, r, "") for dt, r in zip(d.index[ok], R[ok])]
        for k, (tg, be, mh) in exits.items():
            R = sim(*arr, fear, tg, be, mh)
            ok = ~np.isnan(R)
            rows += [("청산 " + k, t, dt, r, "") for dt, r in zip(d.index[ok], R[ok])]
        # (e) 63일 수익률 하위 25%는 날짜별 횡단면이 필요 → 아래에서 처리용으로 저장
        d["fear"] = fear; d["t"] = t
        data[t] = d
    r = pd.DataFrame(rows, columns=["g", "t", "dt", "R", "_"])
    r["구간"] = np.where(r.dt.dt.year <= 2017, "2010~17", "2018~26")
    base = r[r.g == "기준선"].R.mean()

    def table(keys, title):
        out = []
        for k in keys:
            x = r[r.g == k]
            if len(x) == 0:
                continue
            m, lo, hi, n = ci(x.R)
            hv = x.groupby("구간").R.mean()
            out.append([k, n, x.dt.nunique(), m, lo, hi, (x.R > 0).mean() * 100, hv.get("2010~17", np.nan), hv.get("2018~26", np.nan)])
        print(f"\n[{title}]")
        print(pd.DataFrame(out, columns=["그룹", "매매", "진입일수", "평균R", "하한", "상한", "승률%", "2010~17", "2018~26"]).round(3).to_string(index=False))

    table(["S(≤20)", "A(20~30)", "B(30~40)", "C(40~60)", "D(60~80)", "F(>80)", "기준선"], "H1 공포 등급 / H4 환희(F)")
    print("\n[H2 공포일 안에서 종목 고르기 — 차이(그룹 − 나머지)]")
    for k in ["(a)정배열", "(b)ADX상승", "(c)52주고점−25%이내", "(d)50일선−15%이하"]:
        x, y = r[r.g == "공포∩" + k], r[r.g == "공포∩not" + k]
        diff = x.R.mean() - y.R.mean(); se = np.sqrt(x.R.var() / len(x) + y.R.var() / len(y))
        hv = [x[x.구간 == h].R.mean() - y[y.구간 == h].R.mean() for h in ("2010~17", "2018~26")]
        print(f"{k}: 그룹 {len(x)}건 {x.R.mean():.3f}R / 나머지 {len(y)}건 {y.R.mean():.3f}R / 차이 {diff:+.3f} (95% {diff-1.96*se:+.3f}~{diff+1.96*se:+.3f}) / 구간별 {hv[0]:+.3f}, {hv[1]:+.3f}")
    # (e) 63일 수익률 하위 25% (공포일 횡단면)
    cs = pd.concat([data[t][["r63", "fear"]].assign(t=t) for t in test if isinstance(data[t], pd.DataFrame) and "fear" in data[t]])
    cs = cs[cs.fear]
    cs["q"] = cs.groupby(level=0).r63.rank(pct=True)
    low = set(zip(cs.index[cs.q <= .25], cs.t[cs.q <= .25]))
    fr = pd.concat([r[r.g == "공포∩(a)정배열"], r[r.g == "공포∩not(a)정배열"]])
    fr = fr.assign(low=[(dt, t) in low for dt, t in zip(fr.dt, fr.t)])
    x, y = fr[fr.low], fr[~fr.low]
    diff = x.R.mean() - y.R.mean(); se = np.sqrt(x.R.var() / len(x) + y.R.var() / len(y))
    hv = [x[x.구간 == h].R.mean() - y[y.구간 == h].R.mean() for h in ("2010~17", "2018~26")]
    print(f"(e)63일 하위25%: 그룹 {len(x)}건 {x.R.mean():.3f}R / 나머지 {len(y)}건 {y.R.mean():.3f}R / 차이 {diff:+.3f} (95% {diff-1.96*se:+.3f}~{diff+1.96*se:+.3f}) / 구간별 {hv[0]:+.3f}, {hv[1]:+.3f}")

    table(["청산 " + k for k in exits], "H5 공포일 청산 방식 (같은 진입, 시장 폭 ≤30%)")
    a = r[r.g == "청산 A 현재(3R+본전+60)"].set_index(["t", "dt"]).R
    for k in list(exits)[1:]:
        b2 = r[r.g == "청산 " + k].set_index(["t", "dt"]).R
        j = pd.concat([a, b2], axis=1, join="inner").dropna(); dd = j.iloc[:, 1] - j.iloc[:, 0]
        m, lo, hi, n = ci(dd)
        print(f"  {k} − A: {m:+.3f}R (95% {lo:+.3f}~{hi:+.3f}, 같은 진입 {n}건)")

    # H3 QQQ / 합성 TQQQ
    q = qqq.c.reindex(breadth.index).ffill()
    qr = q.pct_change().fillna(0)
    tq = (1 + 3 * qr - 0.0095 / 252).cumprod()
    df = pd.DataFrame({"b": breadth, "q20": q.shift(-20) / q - 1, "q60": q.shift(-60) / q - 1,
                       "t20": tq.shift(-20) / tq - 1, "t60": tq.shift(-60) / tq - 1})
    df["tier"] = tier(df.b.values); df["구간"] = np.where(df.index.year <= 2017, "2010~17", "2018~26")
    print("\n[H3 지수 타이밍 — 그날 매수 후 20·60거래일 뒤 평균 수익률 %, 겹침 포함]")
    g = df.groupby("tier")[["q20", "q60", "t20", "t60"]].mean() * 100
    g["일수"] = df.groupby("tier").size()
    g["TQQQ60 손실확률%"] = df.groupby("tier").t60.apply(lambda x: (x < 0).mean() * 100)
    g.loc["전체"] = list(df[["q20", "q60", "t20", "t60"]].mean() * 100) + [len(df), (df.t60 < 0).mean() * 100]
    print(g.round(2).to_string())
    print(df.assign(f=df.b <= .3).groupby(["f", "구간"])[["q60", "t60"]].mean().mul(100).round(2).unstack().to_string())
    r.to_csv(os.path.join(sys.argv[1], "..", "fear_grade_oos_trades.csv.gz"), index=False)


if __name__ == "__main__":
    main()
