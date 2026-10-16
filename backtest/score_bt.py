"""종합점수(score.py) 기술적 부분 백테스트 — 등급별 승률·평균 R

사용법: python score_bt.py <일봉 CSV 폴더> <섹터맵 CSV(ticker,sector)> [--warmup 200]

과거 날짜마다 score.py와 같은 방식으로 점수를 다시 계산한다.
 - 계산 가능한 항목만 사용: 추세 25, 타이밍 20(스크리너 신호 5점 제외), 시장 15, 감점
 - 펀더 20·수급 15는 과거 값이 없어 제외 → score.py처럼 가용 만점 65로 환산
   (스크리너 신호 5점은 재현 불가라 실제 최대 60 → A등급이 실제보다 덜 나오는 보수적 설정)
 - 통과 조건: 주가 5$ 이상, ATR% 1~4, 20일 평균 거래대금 4,000만$ 이상 (그날 기준)
   시총·업종 제외·베타·흑자·손익비는 과거 값이 없어 미적용
 - 시장(섹터 50일선 위 비율)은 표본 안 같은 섹터 종목들로 계산 (표본이 작아 거침)
매매: 그날 종가 진입, 손절 max(7%, 2ATR), 목표 3R, +1R 도달 후 본전 손절, 최대 60거래일 후 종가 청산,
      종목당 한 번에 한 포지션, 왕복 비용 0.2%
"""
import sys, os, glob, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "signals"))
from signals import build  # noqa: E402

FEE, MAXHOLD = 0.002, 60


def indicators(df):
    d = build(df.copy(), "v2")
    d["sma200"] = d.close.rolling(200).mean()
    hi, lo = d.high, d.low
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    d["cloud_up"] = d.close > np.maximum(span_a, span_b)
    tr = pd.concat([hi - lo, (hi - d.close.shift()).abs(), (lo - d.close.shift()).abs()], axis=1).max(axis=1)
    d["chop"] = 100 * np.log10(tr.rolling(14).sum() / (hi.rolling(14).max() - lo.rolling(14).min())) / np.log10(14)
    cross = (d.k > d.d) & (d.k.shift() <= d.d.shift())
    d["cross2"] = cross | cross.shift(fill_value=False)
    d["vwap_dist"] = (d.close / d.vwap20 - 1) * 100
    d["atr_pct"] = d.atr / d.close * 100
    d["dvol20"] = (d.close * d.volume).rolling(20).mean() / 1e6
    d["above50"] = d.close > d.sma50
    return d


def trades_for(d, mask):
    out, i, n = [], 0, len(d)
    idx = np.where(mask.values)[0]
    nxt = 0
    for i in idx:
        if i < nxt or i >= n - 1:
            continue
        e = d.close.iloc[i]
        risk = max(0.07 * e, 2 * d.atr.iloc[i])
        stop, tgt, be = e - risk, e + 3 * risk, False
        px, why, j = None, None, i
        for j in range(i + 1, min(n, i + 1 + MAXHOLD)):
            if d.low.iloc[j] <= stop:
                px, why = stop, ("본전" if be else "손절"); break
            if d.high.iloc[j] >= tgt:
                px, why = tgt, "익절"; break
            if not be and d.high.iloc[j] >= e + risk:
                stop, be = e, True
        if px is None:
            px, why = d.close.iloc[j], ("기간" if j == i + MAXHOLD else "미결")
        out.append(dict(date=d.index[i], R=(px - e) / risk - FEE * e / risk, ret=px / e - 1 - FEE, why=why, hold=j - i))
        nxt = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("sectors"); ap.add_argument("--warmup", type=int, default=200)
    a = ap.parse_args()
    sec = pd.read_csv(a.sectors).set_index("ticker").sector.to_dict()
    data = {}
    for f in sorted(glob.glob(os.path.join(a.folder, "*.csv"))):
        t = os.path.basename(f)[:-4]
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < a.warmup + 60:
            continue
        data[t] = indicators(df).iloc[a.warmup:]
    # 섹터 breadth (표본 내)
    a50 = pd.DataFrame({t: d.above50 for t, d in data.items()})
    breadth = {}
    for s in set(sec.get(t, "?") for t in data):
        cols = [t for t in data if sec.get(t, "?") == s]
        breadth[s] = a50[cols].mean(axis=1)
    rows = []
    for t, d in data.items():
        b = breadth[sec.get(t, "?")].reindex(d.index)
        trend = d.above50 * 5 + (d.close > d.sma200) * 5 + ((d.adx > 25) & (d.pdi > d.mdi)) * 10 + d.cloud_up * 5
        timing = (((d.k <= 40) & d.cross2) | (d.vwap_dist.abs() <= 3)) * 20
        market = (b >= 0.5) * 10 + d.vwap_state.isin(["강세", "중립"]) * 5
        pen = -((d.chop >= 61.8) * 5 + (d.k >= 90) * 5)
        tot = (trend + timing + market + pen).clip(lower=0)
        conv = tot / 65 * 100
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        grade = np.select([conv >= 80, conv >= 65], ["A", "B"], "C")
        g = pd.Series(grade, index=d.index).where(gate, "탈락")
        for name, m in (("A", g == "A"), ("B", g == "B"), ("C", g == "C"), ("v2 BUY", gate & (d.signal == "BUY")),
                        ("A+v2 BUY", (g == "A") & (d.signal == "BUY")), ("통과 전체(기준선)", gate)):
            for tr in trades_for(d, m):
                tr.update(ticker=t, group=name); rows.append(tr)
    r = pd.DataFrame(rows)
    span = (min(d.index[0] for d in data.values()).date(), max(d.index[-1] for d in data.values()).date())
    print(f"종목 {len(data)}개, 기간 {span[0]} ~ {span[1]}")
    summ = r.groupby("group").agg(매매=("R", "size"), 승률=("ret", lambda x: (x > 0).mean() * 100),
                                  익절률=("why", lambda x: (x == "익절").mean() * 100),
                                  손절률=("why", lambda x: (x == "손절").mean() * 100),
                                  평균R=("R", "mean"), 중앙R=("R", "median"), 평균수익률=("ret", lambda x: x.mean() * 100),
                                  평균보유일=("hold", "mean"))
    order = ["A", "A+v2 BUY", "B", "C", "v2 BUY", "통과 전체(기준선)"]
    print(summ.reindex([o for o in order if o in summ.index]).round(2).to_string())
    r["year"] = pd.to_datetime(r.date).dt.year
    print("\n연도별 A·B 평균R / 매매수")
    print(r[r.group.isin(["A", "B"])].pivot_table(index="year", columns="group", values="R", aggfunc=["mean", "size"]).round(2).to_string())
    r.to_csv(os.path.join(a.folder, "..", "score_bt_trades.csv"), index=False)


if __name__ == "__main__":
    main()
