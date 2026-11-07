"""차트 점수(100점) + 등급 S/A/B/C/D/F 정의와 백테스트 (2026-11-08 사용자 요청)

사용법:
  백테스트: python chart_grade_bt.py bt <일봉 폴더> <QQQ csv(.gz 가능)>
  한 종목 채점: python chart_grade_bt.py score <종목 csv> <QQQ csv>

차트 점수 (100점) — 차트·거래량만 사용 (펀더·뉴스 제외)
 추세 30 : 50일선 위 6 / 200일선 위 6 / 50일선>200일선 3 / 일목 구름 위 5 /
           ADX>25 & +DI>−DI 10 (ADX 20~25 & +DI>−DI 5)
 위치 20 : 20일선 또는 20일 VWAP ±3% 이내 10 / 52주 고점 대비 −25% 이내 5 / 20일 저점 대비 +5% 이내 5
 타이밍 25: 스토캐스틱 K≤40 상향 교차(오늘·어제) 10 / 반전 캔들(망치형·상승장악형) 5 /
           MACD 히스토그램 2일 연속 개선 5 / 일봉 VWAP 강세·중립 5
 거래량 15: 최근 5일 하락일 평균 거래량 < 20일 평균 5 / 양봉 + 거래량 1.2배↑ 5 / OBV > OBV 20일 평균 5
 상대강도 10: 63일 수익률 > QQQ 5 / 20일 수익률 > QQQ 5
 감점    : CHOP ≥61.8 −5 / K ≥90 −5 / 강한 하락추세(50일선 아래 & ADX>25 & −DI>+DI) −10
 등급    : S ≥85 / A 75~84 / B 65~74 / C 50~64 / D 35~49 / F <35
매매(검증용, score_bt.py와 같음): 종가 진입, 손절 max(7%,2ATR), 목표 3R, +1R 후 본전, 최대 60일, 왕복 0.2%
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators, trades_for  # noqa: E402

GRADES = [(85, "S"), (75, "A"), (65, "B"), (50, "C"), (35, "D")]


def load(p):
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def chart_score(df, qqq):
    d = indicators(df)
    c, o, h, l, v = d.close, d.open, d.high, d.low, d.volume
    up = (d.pdi > d.mdi)
    trend = ((c > d.sma50) * 6 + (c > d.sma200) * 6 + (d.sma50 > d.sma200) * 3 + d.cloud_up * 5
             + np.where((d.adx > 25) & up, 10, np.where((d.adx >= 20) & up, 5, 0)))
    near = ((c / d.sma20 - 1).abs() <= 0.03) | (d.vwap_dist.abs() <= 3)
    pos = (near * 10 + (c >= h.rolling(252, min_periods=120).max() * 0.75) * 5
           + (c <= l.rolling(20).min() * 1.05) * 5)
    body = (c - o).abs(); rng = (h - l).replace(0, np.nan)
    hammer = ((np.minimum(o, c) - l) >= 2 * body) & ((h - np.maximum(o, c)) <= body) & (body / rng < 0.4)
    engulf = (c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    hist = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    timing = (((d.k <= 40) & d.cross2) * 10 + (hammer | engulf) * 5
              + ((hist > hist.shift()) & (hist.shift() > hist.shift(2))) * 5
              + d.vwap_state.isin(["강세", "중립"]) * 5)
    v20 = v.rolling(20).mean()
    downv = v.where(c < c.shift()).rolling(5, min_periods=1).mean()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    vol = ((downv < v20) * 5 + ((c > o) & (v >= 1.2 * v20)) * 5 + (obv > obv.rolling(20).mean()) * 5)
    q = qqq.close.reindex(d.index).ffill()
    rs = ((c.pct_change(63) > q.pct_change(63)) * 5 + (c.pct_change(20) > q.pct_change(20)) * 5)
    pen = ((d.chop >= 61.8) * 5 + (d.k >= 90) * 5 + ((c < d.sma50) & (d.adx > 25) & (d.mdi > d.pdi)) * 10)
    d["s_trend"], d["s_pos"], d["s_timing"], d["s_vol"], d["s_rs"], d["s_pen"] = trend, pos, timing, vol, rs, -pen
    d["chart_score"] = (trend + pos + timing + vol + rs - pen).clip(lower=0)
    d["grade"] = np.select([d.chart_score >= t for t, _ in GRADES], [g for _, g in GRADES], "F")
    return d


def backtest(folder, qpath):
    qqq = load(qpath)
    rows, n = [], 0
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = load(f)
        if len(df) < 320:
            continue
        d = chart_score(df, qqq).iloc[252:]
        n += 1
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        groups = {g: gate & (d.grade == g) for g in "SABCDF"}
        groups["S+A"] = gate & d.grade.isin(["S", "A"])
        groups["아무 날(기준선)"] = gate
        for name, m in groups.items():
            for tr in trades_for(d, m):
                tr.update(ticker=os.path.basename(f)[:-4], group=name); rows.append(tr)
    r = pd.DataFrame(rows)
    r["year"] = pd.to_datetime(r.date).dt.year
    r["half"] = np.where(r.year <= 2024, "학습 2023~24", "검증 2025~26")
    print(f"종목 {n}개")
    def summ(x):
        return pd.Series({"매매": len(x), "승률%": (x.ret > 0).mean() * 100, "익절%": (x.why == "익절").mean() * 100,
                          "평균R": x.R.mean(), "중앙R": x.R.median(), "평균수익%": x.ret.mean() * 100,
                          "R 95%하한": x.R.mean() - 1.96 * x.R.std() / np.sqrt(max(len(x), 1)),
                          "R 95%상한": x.R.mean() + 1.96 * x.R.std() / np.sqrt(max(len(x), 1))})
    order = ["S", "A", "S+A", "B", "C", "D", "F", "아무 날(기준선)"]
    out = r.groupby("group").apply(summ).reindex(order).round(2)
    print(out.to_string())
    print("\n기간별 평균R (매매 수)")
    p = r.groupby(["group", "half"]).R.agg(["mean", "size"]).round(2).unstack()
    print(p.reindex(order).to_string())
    r.to_csv(os.path.join(folder, "..", "chart_grade_bt_trades.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "bt":
        backtest(sys.argv[2], sys.argv[3])
    else:
        d = chart_score(load(sys.argv[2]), load(sys.argv[3]))
        cols = ["close", "chart_score", "grade", "s_trend", "s_pos", "s_timing", "s_vol", "s_rs", "s_pen"]
        print(d[cols].tail(5).round(2).to_string())
