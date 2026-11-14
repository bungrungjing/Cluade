"""기관식 다요인(복합 점수) 사전 등록 시험 (2026-11-14, 결과 보기 전에 가설·기준 고정, 한 번만 실행)

이전 시험(tournament)은 요인마다 5종목 집중 → 잡음이 큼. 이번에는 기관 방식대로
 (1) 종목 전체의 월간 순위 상관(IC)으로 요인 자체의 힘을 재고
 (2) 약한 요인 여러 개를 '같은 비중'으로 섞은 복합 점수(가중치를 결과로 고르지 않음 → 과최적화 방지)를
 (3) 상위 10% 분산 포트폴리오(약 50~110종목)와 상위 20종목으로 시험한다.

관문(매월 말 시점 기준): 주가 ≥5, 20일 평균 거래대금 ≥4천만 달러, 과거 252일 이상 데이터
요인(값이 클수록 좋게 부호 통일):
 F1 모멘텀 12-1: 252일 전→21일 전 수익률
 F2 52주 고점 근접: 종가 / 252일 최고 종가
 F3 1개월 되돌림: −(21일 수익률)
 F4 저변동성: −(60일 일간 수익률 표준편차)
복합 C = F1~F4 단면 z점수(±3 잘라냄) 단순 평균
시험
 T1 IC: 각 요인·C의 월말 순위 → 다음 달 수익률 스피어만 상관. 평균 IC와 t(월별)
 T2 포트폴리오: C 상위 10% 균등 / C 상위 20종목 균등, 월 리밸런싱, 비용 = 0.2% × 회전율(편도 기준 교체 비중)
     비교: 관문 통과 전체 균등(EW), QQQ
구간: 설계 2011~2017 / 검증 2018~2026 (두 구간 모두 같은 방향이어야 함)
합격(사전 등록)
 T1: C의 평균 IC가 두 구간 모두 >0 이고 전체 t ≥ 2.5 (요인 5개 동시 시험 보정)
 T2: 비용 후 연복리가 두 구간 모두 EW보다 높고, 샤프도 두 구간 모두 EW 이상, 최대 낙폭이 EW의 1.3배 이내
한계: 생존 편향(지금 상장 종목만) — 되돌림·저변동성·모멘텀을 모두 부풀릴 수 있음. 재무(퀄리티·가치) 요인 없음.
python3 factor_blend.py <daily 폴더(*.csv.gz)> [결과 md]
"""
import glob, os, sys
import numpy as np
import pandas as pd

FEE = 0.002


def load(folder):
    C, V = {}, {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        if t.startswith("_"):
            continue
        d = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        C[t] = d["close"].astype(float); V[t] = d["volume"].astype(float)
    return pd.DataFrame(C), pd.DataFrame(V)


def zs(s):
    s = s.dropna()
    if len(s) < 20:
        return s * np.nan
    z = (s - s.mean()) / s.std()
    return z.clip(-3, 3)


def stats(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(12)
    return cagr, mdd, sh


def main(folder, outp=None):
    out = []
    say = lambda s="": (print(s), out.append(s))
    close, vol = load(folder)
    qqq = close.pop("QQQ"); close = close.drop(columns=["SPY"], errors="ignore"); vol = vol[close.columns]
    dv20 = (close * vol).rolling(20, min_periods=15).mean()
    ret1 = close.pct_change(fill_method=None)
    sd60 = ret1.rolling(60, min_periods=50).std()
    hi252 = close.rolling(252, min_periods=240).max()
    me = close.resample("ME").last().index
    me = [d for d in me if d >= pd.Timestamp("2011-01-01")]
    days = close.index
    rows, port = [], []
    prev_w = {"D": pd.Series(dtype=float), "T20": pd.Series(dtype=float)}
    for i in range(len(me) - 1):
        d = days[days <= me[i]][-1]; d2 = days[days <= me[i + 1]][-1]
        p = close.loc[d]
        ok = (p >= 5) & (dv20.loc[d] >= 4e7) & close.loc[:d].iloc[-252:].notna().all()
        u = p[ok].index
        if len(u) < 50:
            continue
        loc = close.index.get_loc(d)
        f1 = close.iloc[loc - 21][u] / close.iloc[loc - 252][u] - 1
        f2 = p[u] / hi252.loc[d][u]
        f3 = -(p[u] / close.iloc[loc - 21][u] - 1)
        f4 = -sd60.loc[d][u]
        F = pd.DataFrame({"F1": f1, "F2": f2, "F3": f3, "F4": f4})
        Z = F.apply(zs)
        F["C"] = Z.mean(axis=1, skipna=False)
        nxt = close.loc[d2][u] / p[u] - 1   # 다음 달 수익(상장폐지 종목은 데이터에 없음 → 생존 편향)
        F["N"] = nxt
        F = F.dropna()
        ic = {k: F[k].rank().corr(F["N"].rank()) for k in ["F1", "F2", "F3", "F4", "C"]}
        ew = F["N"].mean()
        res = {"date": d, "EW": ew, "QQQ": qqq.loc[d2] / qqq.loc[d] - 1, "n": len(F), **{"IC_" + k: v for k, v in ic.items()}}
        s = F.sort_values("C", ascending=False)
        for key, sel in (("D", s.index[: max(1, len(s) // 10)]), ("T20", s.index[:20])):
            w = pd.Series(1 / len(sel), index=sel)
            turn = w.subtract(prev_w[key], fill_value=0).abs().sum() / 2
            res[key] = F.loc[sel, "N"].mean() - FEE * turn
            res[key + "_turn"] = turn
            prev_w[key] = w
        rows.append(res)
    R = pd.DataFrame(rows).set_index("date")
    halves = {"설계 2011~17": R[R.index.year <= 2017], "검증 2018~26": R[R.index.year >= 2018], "전체": R}
    say(f"# 다요인 복합 점수 시험 (2026-11-14) — {R.index[0].date()} ~ {R.index[-1].date()}, 월 {len(R)}개, 평균 관문 통과 {R['n'].mean():.0f}종목")
    say("\n## T1 월간 순위 상관(IC) — 다음 달 수익률")
    say("| 요인 | " + " | ".join(f"{h} 평균IC (t)" for h in halves) + " |")
    say("|---|---|---|---|")
    for k, nm in (("F1", "F1 모멘텀 12-1"), ("F2", "F2 52주 고점 근접"), ("F3", "F3 1개월 되돌림"), ("F4", "F4 저변동성"), ("C", "C 복합")):
        cells = []
        for h, df in halves.items():
            x = df["IC_" + k].dropna(); t = x.mean() / (x.std() / np.sqrt(len(x)))
            cells.append(f"{x.mean():+.3f} ({t:.2f})")
        say(f"| {nm} | " + " | ".join(cells) + " |")
    say("\n## T2 포트폴리오 (비용 0.2% × 회전율 반영, 월 리밸런싱)")
    say("| 포트폴리오 | " + " | ".join(f"{h} 연복리/낙폭/샤프" for h in halves) + " | 월 평균 회전율 |")
    say("|---|---|---|---|---|")
    verdict = {}
    for k, nm in (("D", "C 상위 10%"), ("T20", "C 상위 20종목"), ("EW", "관문 통과 전체 균등"), ("QQQ", "QQQ")):
        cells = []
        for h, df in halves.items():
            c, m, s = stats(df[k]); cells.append(f"{100*c:.1f}% / {100*m:.1f}% / {s:.2f}")
            verdict[(k, h)] = (c, m, s)
        tv = f"{100*R[k+'_turn'].mean():.0f}%" if k + "_turn" in R else "-"
        say(f"| {nm} | " + " | ".join(cells) + f" | {tv} |")
    x = R["IC_C"].dropna(); tC = x.mean() / (x.std() / np.sqrt(len(x)))
    t1 = (halves["설계 2011~17"]["IC_C"].mean() > 0) and (halves["검증 2018~26"]["IC_C"].mean() > 0) and tC >= 2.5
    say("\n## 판정 (사전 등록)")
    say(f"- T1 복합 IC: {'합격' if t1 else '불합격'} (전체 t {tC:.2f})")
    for k, nm in (("D", "C 상위 10%"), ("T20", "C 상위 20종목")):
        ok = all(verdict[(k, h)][0] > verdict[("EW", h)][0] and verdict[(k, h)][2] >= verdict[("EW", h)][2] for h in ("설계 2011~17", "검증 2018~26")) \
            and verdict[(k, "전체")][1] >= 1.3 * verdict[("EW", "전체")][1]
        say(f"- T2 {nm}: {'합격' if ok else '불합격'}")
    last = R.index[-1]
    say(f"\n한계: 생존 편향 큼(상장폐지 종목 없음), 재무 요인 없음, 비용은 0.2% 근사(국내 증권사 해외주식 실제 수수료·환전은 더 클 수 있음).")
    if outp:
        open(outp, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
