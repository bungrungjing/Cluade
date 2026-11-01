"""그림자 장부 — 실제로 사지 않은 것까지 신호를 전부 기록하고 자동 채점 (전향 검증)

규칙(rule 칸):
 - v3         : v3 매수 신호
 - 섹터RSI≤40  : 1단계 관문 통과 + 그 종목 섹터의 평균 RSI(14) ≤ 40 (2026-11-01 추가, 사용자 가설 —
                 백테스트에서 매매당 우위는 있었지만 계좌 단위로는 우위 없음 → 전향 검증용)

사용법: python shadow.py <일봉 폴더(*.csv.gz)> <장부 폴더>

매일(미국 장 마감 후) 실행:
 1) 최근 5거래일 안의 v3 신호 중 장부에 없는 것을 추가 (작업이 하루 빠져도 놓치지 않게)
 2) 아직 안 끝난 매매를 실전 규칙(LIVE)으로 다시 채점: 손절 max(7%, 2ATR) / +1R에서 손절을 매수가+0.3R로 / 3R 목표 / 최대 60거래일
 3) <장부 폴더>/ledger.csv, summary.md 갱신

진입가는 신호일 종가(백테스트와 같은 기준). 실제로는 다음 날 지정가로 사므로 next_open도 함께 기록한다.
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import indicators, sim, load_csv, LIVE  # noqa: E402
from full_bt import stats  # noqa: E402
from sector_rsi_bt import rsi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = 420  # 지표 계산에 쓰는 최근 거래일 수
START = pd.Timestamp(os.environ.get("SHADOW_START", "2026-11-02"))  # 장부 시작일(이전 신호는 기록하지 않음)


def main():
    folder, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ledger.csv")
    led = pd.read_csv(path, parse_dates=["date", "exit_date"]) if os.path.exists(path) else pd.DataFrame()
    uni = pd.read_csv(os.path.join(HERE, "universe.csv")).set_index("ticker")

    data = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv.gz"))):
        t = os.path.basename(f)[:-7]
        df = load_csv(f).iloc[-KEEP:]
        if len(df) >= 260:
            data[t] = df
    spy = data.pop("SPY", None); data.pop("QQQ", None); vix = data.pop("_VIX", None)
    ind = {t: indicators(df) for t, df in data.items()}
    breadth = pd.DataFrame({t: d.above50 for t, d in ind.items()}).mean(axis=1) * 100
    secname = uni.sector.fillna("Unknown")
    for t, d in ind.items():
        d["rsi"] = rsi(d.close)
    R = pd.DataFrame({t: d.rsi for t, d in ind.items() if t in secname.index})
    secrsi = {sname: R[[t for t in R.columns if secname[t] == sname]].mean(axis=1) for sname in set(secname[t] for t in R.columns)}
    RULES = {"v3": lambda t, d: d.v3,
             "섹터RSI≤40": lambda t, d: d.gate & (secrsi[secname[t]].reindex(d.index) <= 40) if t in secname.index else d.gate & False}
    last = max(d.index[-1] for d in ind.values())

    def mkt(day):
        b = breadth.get(day, np.nan)
        s = spy.close.loc[:day] if spy is not None else None
        v = vix.close.loc[:day].iloc[-1] if vix is not None and len(vix.close.loc[:day]) else np.nan
        s200 = bool(s.iloc[-1] > s.rolling(200).mean().iloc[-1]) if s is not None and len(s) > 200 else None
        return dict(breadth=round(b, 1), vix=round(v, 2), spy_above200=s200)

    # 1) 새 신호 추가
    # 종목당 한 번에 한 매매(백테스트와 같은 기준): 앞 매매가 끝나기 전의 신호는 '겹침'으로 기록하지 않는다
    if len(led) and "rule" not in led:
        led["rule"] = "v3"
    have = set(zip(led.date, led.ticker, led.rule)) if len(led) else set()
    busy = {}
    if len(led):
        for _, x in led.iterrows():
            end = pd.Timestamp.max if x.get("status") != "closed" else pd.Timestamp(x.exit_date)
            key = (x.rule, x.ticker)
            busy[key] = max(busy.get(key, pd.Timestamp.min), end)
    new = []
    lb = int(os.environ.get("SHADOW_LOOKBACK", "5"))
    for t, d in ind.items():
      for rule, fn in RULES.items():
        sig = fn(t, d).fillna(False).astype(bool)
        recent = sig[d.index >= max(START, d.index[-lb] if len(d) >= lb else d.index[0])]
        for day in recent.index[recent]:
            key = (rule, t)
            if (day, t, rule) in have or day <= busy.get(key, pd.Timestamp.min):
                continue
            tr = sim(d, pd.Series(d.index == day, index=d.index), **LIVE)
            busy[key] = pd.Timestamp.max if (not tr or tr[0]["status"] == "open") else tr[0]["exit_date"]
            row = d.loc[day]
            e, risk = row.close, max(0.07 * row.close, 2 * row.atr)
            new.append(dict(date=day, rule=rule, ticker=t, name=uni.name.get(t, ""), sector=uni.sector.get(t, ""),
                            entry=round(e, 2), stop=round(e - risk, 2), plus1R=round(e + risk, 2),
                            target=round(e + 3 * risk, 2), relvol=round(row.relvol, 2), k=round(row.k, 1),
                            vwap_state=row.vwap_state, atr_pct=round(row.atr_pct, 2), rsi=round(row.rsi, 1),
                            sector_rsi=round(secrsi[secname[t]].get(day, np.nan), 1) if t in secname.index else np.nan, **mkt(day)))
    if new:
        led = pd.concat([led, pd.DataFrame(new)], ignore_index=True)

    # 2) 채점(아직 안 끝난 것과 새로 들어온 것)
    if len(led):
        for c in ("status", "why", "R", "ret", "hold", "exit_date", "next_open"):
            if c not in led:
                led[c] = np.nan
        for c in ("status", "why", "exit_date"):
            led[c] = led[c].astype(object)
        todo = led.status.ne("closed")
        for i in led.index[todo]:
            t, day = led.at[i, "ticker"], led.at[i, "date"]
            d = ind.get(t)
            if d is None or day not in d.index:
                continue
            pos = d.index.get_loc(day)
            if pos + 1 < len(d):
                led.at[i, "next_open"] = round(d.open.iloc[pos + 1], 2)
            mask = pd.Series(d.index == day, index=d.index)
            tr = sim(d, mask, **LIVE)
            if tr:
                tr = tr[0]
                led.loc[i, ["status", "why", "R", "ret", "hold", "exit_date"]] = [
                    tr["status"], tr["why"], round(tr["R"], 3), round(tr["ret"], 4), tr["hold"], tr["exit_date"]]
            else:
                led.loc[i, ["status", "why"]] = ["open", "진입 대기"]
        led = led.sort_values(["date", "rule", "ticker"]).reset_index(drop=True)
        led["date"] = pd.to_datetime(led.date).dt.strftime("%Y-%m-%d")
        led["exit_date"] = pd.to_datetime(led.exit_date).dt.strftime("%Y-%m-%d")
        led.to_csv(path, index=False)
        led["date"] = pd.to_datetime(led.date)

    # 3) 요약
    m = mkt(last)
    L = [f"# 그림자 장부 요약 (기준일 {last:%Y-%m-%d})", "",
         f"- 시장: 50일선 위 비율 {m['breadth']}% / VIX {m['vix']} / SPY 200일선 {'위' if m['spy_above200'] else '아래'}",
         f"- 장부 시작 {START:%Y-%m-%d}, 누적 신호 {len(led)}건", ""]
    if len(led):
        done = led[led.status == "closed"]
        L.append("## 끝난 매매 성적 (실전 규칙 LIVE, 규칙별)")
        L.append("")
        L.append(pd.DataFrame({k: stats(g) for k, g in done.groupby("rule")}).T.to_markdown() if len(done) else "아직 끝난 매매 없음")
        L.append("")
        if len(done):
            done = done.assign(reg=pd.cut(done.breadth, [-1, 40, 60, 101], labels=["약세(폭≤40)", "중립", "강세(폭≥60)"]))
            L.append(pd.DataFrame({f"{a} / {b}": stats(g) for (a, b), g in done.groupby(["rule", "reg"], observed=True)}).T.to_markdown())
            L.append("")
        L.append("## 현재 섹터 평균 RSI")
        L.append("")
        L.append(", ".join(f"{k} {v.dropna().iloc[-1]:.1f}" for k, v in sorted(secrsi.items(), key=lambda kv: kv[1].dropna().iloc[-1])))
        L.append("")
        today = led[led.date == last]
        L.append(f"## 오늘 새 신호 ({len(today)}건)")
        L.append("")
        if len(today):
            for rule, g in today.groupby("rule"):
                L.append(f"### {rule} ({len(g)}건)")
                L.append(g[["ticker", "name", "sector", "entry", "stop", "plus1R", "target", "rsi", "sector_rsi", "atr_pct"]].head(40).to_markdown(index=False))
                L.append("")
        L.append("")
        op = led[led.status == "open"]
        L.append(f"## 진행 중 {len(op)}건")
        L.append("")
        if len(op):
            L.append(op.assign(date=op.date.dt.strftime("%Y-%m-%d"))[["date", "rule", "ticker", "name", "entry", "stop", "plus1R", "target", "R", "hold"]].tail(60).to_markdown(index=False))
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
