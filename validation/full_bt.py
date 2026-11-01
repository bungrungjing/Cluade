"""v3 매수 신호 전체 백테스트 — 일봉 2010~현재, 유니버스 전체

사용법: python full_bt.py <일봉 폴더(*.csv.gz)> <결과 폴더>

- 규칙은 고정(v3 그대로). 숫자를 바꿔 가며 좋은 값을 고르지 않는다(과최적화 방지).
- 청산 모델 3개: LIVE(현재 실전: 3R 목표, +1R에서 손절을 매수가+0.3R로) / DEF(+1R 본전) / TRD(목표 없이 4ATR 추적)
- 비교 기준: '아무 날'(같은 관문을 통과한 모든 날 진입) — v3가 이것보다 나아야 의미가 있다.
- 나누어 보기: 기간(설계 2011~2021 / 검증 2022~), 연도 구간, 시장 폭(50일선 위 비율) 구간, SPY 200일선 위·아래
- 한계(결과에 함께 적음): 생존 편향(지금 살아 있는 종목만), 시총 관문은 과거 값이 없어 미적용
"""
import glob, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE, DEF, TRD  # noqa: E402

SPLIT = pd.Timestamp("2022-01-01")
ERAS = [("2011~2014", "2011", "2014"), ("2015~2018", "2015", "2018"), ("2019~2021", "2019", "2021"),
        ("2022(하락장)", "2022", "2022"), ("2023~", "2023", "2030")]


def ci(x, n=1000):
    if len(x) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def stats(x):
    if len(x) == 0:
        return dict(매매=0)
    lo, hi = ci(x.R.values)
    w, l = x.R[x.R > 0].sum(), -x.R[x.R <= 0].sum()
    return dict(매매=len(x), 승률=round(100 * (x.R > 0).mean(), 1), 평균R=round(x.R.mean(), 3),
                R구간95=f"{lo:+.2f}~{hi:+.2f}" if lo == lo else "-", 평균수익률=round(100 * x.ret.mean(), 2),
                손익비PF=round(w / l, 2) if l > 0 else np.nan, 평균보유일=round(x.hold.mean(), 1))


def main():
    folder, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    data = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        df = load_csv(f)
        if len(df) < 260:
            continue
        data[t] = df
    spy = data.pop("SPY", None); data.pop("QQQ", None); data.pop("_VIX", None)
    ind = {}
    for t, df in data.items():
        d = indicators(df)
        ind[t] = d.iloc[200:]
    print(f"지표 계산 {len(ind)}종목 {time.time() - t0:.0f}초", flush=True)

    breadth = pd.DataFrame({t: d.above50 for t, d in ind.items()}).mean(axis=1) * 100
    spy200 = (spy.close > spy.close.rolling(200).mean()) if spy is not None else None

    rows = []
    for t, d in ind.items():
        tests = {"v3": d.v3, "아무 날": d.gate}
        for mname, kw in (("LIVE", LIVE), ("DEF", DEF), ("TRD", TRD)):
            for tname, m in tests.items():
                if tname == "아무 날" and mname != "LIVE":
                    continue
                for tr in sim(d, m, **kw):
                    tr.update(ticker=t, model=mname, test=tname); rows.append(tr)
        for tr in sim(d, d.v3, entry="open", **LIVE):
            tr.update(ticker=t, model="LIVE", test="v3(다음날 시가)"); rows.append(tr)
    r = pd.DataFrame(rows)
    r = r[r.status == "closed"].copy()
    r["breadth"] = breadth.reindex(r.date).values
    r["regime"] = pd.cut(r.breadth, [-1, 40, 60, 101], labels=["약세(폭≤40)", "중립", "강세(폭≥60)"])
    if spy200 is not None:
        r["spy200"] = spy200.reindex(r.date).map({True: "SPY 200일선 위", False: "SPY 200일선 아래"}).values
    print(f"시뮬레이션 {len(r)}건 {time.time() - t0:.0f}초", flush=True)

    lines = [f"# v3 전체 백테스트 결과 ({pd.Timestamp.now():%Y-%m-%d})", "",
             f"- 종목 {len(ind)}개, 기간 {min(d.index[0] for d in ind.values()):%Y-%m} ~ {max(d.index[-1] for d in ind.values()):%Y-%m}",
             "- 진입: 신호일 종가(별도로 다음 날 시가), 손절 max(7%, 2ATR), 왕복 비용 0.2%, 종목당 한 번에 한 매매",
             "- R = 손절폭 대비 손익 배수. 평균R이 0보다 크고 95% 구간 하한도 0보다 커야 '우위 있음'", ""]

    def table(title, df_):
        lines.append(f"## {title}"); lines.append("")
        lines.append(df_.to_markdown()); lines.append("")

    main_t = pd.DataFrame({f"{tn} / {mn}": stats(r[(r.test == tn) & (r.model == mn)])
                           for tn, mn in [("v3", "LIVE"), ("v3", "DEF"), ("v3", "TRD"), ("v3(다음날 시가)", "LIVE"), ("아무 날", "LIVE")]}).T
    table("1. 전체", main_t)

    sub = r[r.model == "LIVE"]
    pt = {}
    for tn in ("v3", "아무 날"):
        x = sub[sub.test == tn]
        pt[f"{tn} 설계(2011~2021)"] = stats(x[x.date < SPLIT])
        pt[f"{tn} 검증(2022~)"] = stats(x[x.date >= SPLIT])
    table("2. 설계 구간 vs 검증 구간 (LIVE 청산)", pd.DataFrame(pt).T)

    et = {}
    for name, a, b in ERAS:
        for tn in ("v3", "아무 날"):
            x = sub[(sub.test == tn) & (sub.date >= a) & (sub.date <= f"{b}-12-31")]
            et[f"{name} {tn}"] = stats(x)
    table("3. 연도 구간별 (LIVE 청산)", pd.DataFrame(et).T)

    rt = {}
    for reg in ["약세(폭≤40)", "중립", "강세(폭≥60)"]:
        for tn in ("v3", "아무 날"):
            rt[f"{reg} {tn}"] = stats(sub[(sub.test == tn) & (sub.regime == reg)])
    if "spy200" in sub:
        for reg in ["SPY 200일선 위", "SPY 200일선 아래"]:
            for tn in ("v3", "아무 날"):
                rt[f"{reg} {tn}"] = stats(sub[(sub.test == tn) & (sub.spy200 == reg)])
    table("4. 시장 국면별 (LIVE 청산)", pd.DataFrame(rt).T)

    v = sub[sub.test == "v3"]
    why = v.why.value_counts(normalize=True).mul(100).round(1).rename("비율%").to_frame()
    table("5. v3 청산 사유 (LIVE)", why)
    yr = pd.DataFrame({y: stats(x) for y, x in v.groupby(v.date.dt.year)}).T if len(v) else pd.DataFrame()
    if len(yr):
        table("6. v3 연도별 (LIVE)", yr[["매매", "승률", "평균R", "평균수익률"]])

    lines += ["## 한계", "",
              "- 생존 편향: 2026-11 기준 시총 10억 달러·거래대금 4,000만 달러 이상으로 살아 있는 종목만 넣었다. 상장 폐지·몰락한 종목이 빠져 결과가 실제보다 좋게 나올 수 있다.",
              "- 시총·흑자·업종 관문은 과거 값이 없어 그날 기준 주가·ATR%·거래대금 관문만 적용했다.",
              "- 같은 날 손절과 목표가 모두 닿으면 손절로 처리했다(보수적).",
              "- 계좌 단위 결과(동시에 여러 종목 보유, 현금 한도)는 아니다. 신호 하나하나의 질을 본 것이다.", ""]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    r.to_csv(os.path.join(out, "trades.csv.gz"), index=False, compression="gzip")
    print("\n".join(lines))
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
