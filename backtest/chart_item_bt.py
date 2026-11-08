"""차트 점수 항목별 예측력 시험 (2026-11-08) — 학습 2023~24에서 고르고 검증 2025~26에서 한 번만 확인

사용법: python chart_item_bt.py <일봉 폴더> <QQQ csv>
결과: 항목별 '참일 때 − 거짓일 때' 평균 R(그날 종가 진입, score_bt.py와 같은 청산) 차이, 학습/검증 각각
      → 학습에서 +0.05R 이상이고 검증에서도 같은 방향인 항목만 새 점수에 남긴다(가중치 = 학습 차이)
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_grade_bt import load, chart_score  # noqa: E402

FEE, MAXHOLD = 0.002, 60


def day_R(d):
    c, h, l, atr = d.close.values, d.high.values, d.low.values, d.atr.values
    n, out = len(c), np.full(len(c), np.nan)
    for i in range(n - 1):
        e = c[i]; risk = max(0.07 * e, 2 * atr[i])
        if not np.isfinite(risk):
            continue
        stop, tgt, be, px = e - risk, e + 3 * risk, False, None
        j = i
        for j in range(i + 1, min(n, i + 1 + MAXHOLD)):
            if l[j] <= stop: px = stop; break
            if h[j] >= tgt: px = tgt; break
            if not be and h[j] >= e + risk: stop, be = e, True
        if px is None:
            if j < i + MAXHOLD: continue  # 미결 제외
            px = c[j]
        out[i] = (px - e) / risk - FEE * e / risk
    return out


def items(d, q):
    c, o, h, l, v = d.close, d.open, d.high, d.low, d.volume
    up = d.pdi > d.mdi
    body = (c - o).abs(); rng = (h - l).replace(0, np.nan)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    hist = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    v20 = v.rolling(20).mean(); downv = v.where(c < c.shift()).rolling(5, min_periods=1).mean()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    qc = q.close.reindex(d.index).ffill()
    return pd.DataFrame({
        "50일선 위": c > d.sma50, "200일선 위": c > d.sma200, "정배열 50>200": d.sma50 > d.sma200,
        "구름 위": d.cloud_up, "ADX>25 상승": (d.adx > 25) & up, "ADX20~25 상승": d.adx.between(20, 25) & up,
        "20일선/VWAP ±3%": ((c / d.sma20 - 1).abs() <= 0.03) | (d.vwap_dist.abs() <= 3),
        "52주고점 −25% 이내": c >= h.rolling(252, min_periods=120).max() * 0.75,
        "20일저점 +5% 이내": c <= l.rolling(20).min() * 1.05,
        "K≤40 상향교차": (d.k <= 40) & d.cross2,
        "반전캔들": ((np.minimum(o, c) - l) >= 2 * body) & ((h - np.maximum(o, c)) <= body) & (body / rng < 0.4)
                  | ((c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())),
        "MACD 2일 개선": (hist > hist.shift()) & (hist.shift() > hist.shift(2)),
        "VWAP 강세·중립": d.vwap_state.isin(["강세", "중립"]),
        "하락일 거래량 감소": downv < v20, "양봉+거래량1.2배": (c > o) & (v >= 1.2 * v20),
        "OBV>20일평균": obv > obv.rolling(20).mean(),
        "63일 QQQ보다 강": c.pct_change(63) > qc.pct_change(63), "20일 QQQ보다 강": c.pct_change(20) > qc.pct_change(20),
        "CHOP≥61.8(감점)": d.chop >= 61.8, "K≥90(감점)": d.k >= 90,
        "강한 하락추세(감점)": (c < d.sma50) & (d.adx > 25) & (d.mdi > d.pdi),
    })


def main(folder, qpath):
    q = load(qpath); frames = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = load(f)
        if len(df) < 320: continue
        d = chart_score(df, q)
        it = items(d, q).astype(float)
        it["R"] = day_R(d); it["score"] = d.chart_score
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
        it = it[gate].iloc[252:] if len(it[gate]) > 252 else it[gate]
        it["tk"] = os.path.basename(f)[:-4]; frames.append(it)
    a = pd.concat(frames).dropna(subset=["R"])
    a["half"] = np.where(a.index.year <= 2024, "학습", "검증")
    rows = []
    for col in [c for c in a.columns if c not in ("R", "score", "tk", "half")]:
        r = {"항목": col}
        for hf in ("학습", "검증"):
            x = a[a.half == hf]
            on, off = x.R[x[col] == 1], x.R[x[col] == 0]
            r[f"{hf} 차이R"] = on.mean() - off.mean(); r[f"{hf} 참비율%"] = len(on) / len(x) * 100
        rows.append(r)
    t = pd.DataFrame(rows).round(3)
    print(f"표본 일수 학습 {sum(a.half=='학습')} / 검증 {sum(a.half=='검증')}, 종목 {a.tk.nunique()}")
    print(t.to_string(index=False))
    keep = t[(t["학습 차이R"].abs() >= 0.05)]
    w = dict(zip(keep["항목"], keep["학습 차이R"]))
    print("\n학습에서 고른 항목·가중치:", {k: round(v, 3) for k, v in w.items()})
    s = sum(a[k] * v for k, v in w.items())
    a["new"] = s
    tr = a[a.half == "학습"]; cuts = tr.new.quantile([0.8, 0.6, 0.4, 0.2]).values  # 학습 분포로 등급 컷
    a["g"] = np.select([a.new >= cuts[0], a.new >= cuts[1], a.new >= cuts[2], a.new >= cuts[3]], ["상위20%", "60~80%", "40~60%", "20~40%"], "하위20%")
    print("\n새 점수 등급별 평균R (매일 진입, 겹침 포함 — 구간은 과소추정)")
    print(a.groupby(["g", "half"]).R.agg(["mean", "size"]).round(3).unstack().reindex(["상위20%", "60~80%", "40~60%", "20~40%", "하위20%"]).to_string())
    print("\n기존 차트 점수 상위20% vs 하위20% (비교)")
    oc = tr.score.quantile([0.8, 0.2]).values
    a["og"] = np.select([a.score >= oc[0], a.score <= oc[1]], ["상위20%", "하위20%"], "중간")
    print(a.groupby(["og", "half"]).R.mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
