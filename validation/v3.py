"""v3 매수 신호(가벼운 버전)와 매매 시뮬레이터 — 검증 파이프라인 공용

signals/signals.py 의 build(df, "v3") 와 같은 BUY 규칙을 스퀴즈·바닥형 계산 없이 빠르게 다시 만든다.
(test_equivalence() 로 build() 결과와 같은지 확인)

v3 BUY = 5거래일 전 ADX>25 & +DI>-DI, 오늘 종가>50일선
         + (저가가 20일선 또는 20일 VWAP ±2%에 닿고 양봉, 또는 %K가 40 이하에서 %D 상향 교차)
         + 일봉 VWAP 상태가 '약세'가 아님 + 거래량 ≥ 20일 평균 × 1.2 + 같은 날 SELL 조건 아님
1단계 관문(그날 기준): 주가 ≥ 5달러, ATR% 1~4, 20일 평균 거래대금 ≥ 4,000만 달러
"""
import os, sys
import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, SMAIndicator, MACD
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "signals"))
from signals import daily_vwap_state, VOL_MULT  # noqa: E402

FEE = 0.002  # 왕복 수수료·체결 차이


def indicators(df):
    d = df.copy()
    adx = ADXIndicator(d.high, d.low, d.close, 14)
    d["adx"], d["pdi"], d["mdi"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    d["sma20"] = SMAIndicator(d.close, 20).sma_indicator()
    d["sma50"] = SMAIndicator(d.close, 50).sma_indicator()
    st = StochasticOscillator(d.high, d.low, d.close, 14, 3)
    d["k"], d["d"] = st.stoch(), st.stoch_signal()
    m = MACD(d.close)
    d["macd"], d["macd_sig"] = m.macd(), m.macd_signal()
    d["atr"] = AverageTrueRange(d.high, d.low, d.close, 14).average_true_range()
    d["vwap20"], d["vwap_state"] = daily_vwap_state(d)
    d["relvol"] = d.volume / d.volume.rolling(20).mean()
    d["atr_pct"] = d.atr / d.close * 100
    d["dvol20"] = (d.close * d.volume).rolling(20).mean() / 1e6
    d["above50"] = d.close > d.sma50

    trend = (d.adx.shift(5) > 25) & (d.pdi.shift(5) > d.mdi.shift(5)) & (d.close > d.sma50)
    k_up40 = (d.k > d.d) & (d.k.shift() <= d.d.shift()) & (d.k.shift() <= 40)

    def touch(level):
        return (d.low <= level * 1.02) & (d.close >= level * 0.98) & (d.close > d.open)
    pullback = touch(d.sma20) | touch(d.vwap20)
    k_dn = (d.k < d.d) & (d.k.shift() >= d.d.shift()) & (d.k.shift() >= 80)
    macd_dn = (d.macd < d.macd_sig) & (d.macd.shift() >= d.macd_sig.shift())
    sell = k_dn | (macd_dn & (d.close < d.sma20))
    buy = trend & (pullback | k_up40) & (d.vwap_state != "약세") & (d.volume >= VOL_MULT * d.volume.rolling(20).mean()) & ~sell
    d["gate"] = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)
    d["v3raw"] = buy
    d["v3"] = buy & d.gate
    return d


# 청산 모델
LIVE = dict(tgt=3.0, be_at=1.0, lock=0.3, maxhold=60)   # 현재 실전 규칙: 3R 목표, +1R에서 손절을 매수가+0.3R로
DEF = dict(tgt=3.0, be_at=1.0, lock=0.0, maxhold=60)    # 예전 방어형: +1R에서 본전 손절
TRD = dict(tgt=None, be_at=1.0, lock=0.0, trail=4.0, maxhold=250)  # 수익추종: 목표 없음, 최고가−4ATR 추적


def sim(d, mask, tgt=3.0, be_at=1.0, lock=0.0, trail=None, maxhold=60, entry="close"):
    """신호일 종가(entry='close') 또는 다음 날 시가(entry='open')에 진입.
    손절 = 진입가 − max(7%, 2ATR). 같은 날 손절·목표가 모두 닿으면 손절로 본다(보수적).
    반환: 매매 목록(dict) — 마지막까지 안 끝난 매매는 status='open'"""
    out, nxt, n = [], 0, len(d)
    o_, h_, l_, c_, a_ = (d[x].values for x in ("open", "high", "low", "close", "atr"))
    for i in np.where(mask.values)[0]:
        if i < nxt:
            continue
        s = i + 1 if entry == "open" else i
        if s >= n:
            continue
        e = o_[s] if entry == "open" else c_[i]
        risk = max(0.07 * e, 2 * a_[i])
        stop, hh, moved, res, why, j = e - risk, e, False, None, "", s
        for j in range(s + 1 if entry == "close" else s, min(n, s + 1 + maxhold)):
            if l_[j] <= stop:
                res, why = (min(stop, o_[j]) - e) / risk, ("손절" if not moved else "이익보호"); break
            if tgt is not None and h_[j] >= e + tgt * risk:
                res, why = max(tgt, (o_[j] - e) / risk), "목표"; break
            hh = max(hh, h_[j])
            if not moved and hh >= e + be_at * risk:
                stop, moved = max(stop, e + lock * risk), True
            if moved and trail is not None:
                stop = max(stop, hh - trail * a_[j])
        status = "closed"
        if res is None:
            res = (c_[j] - e) / risk
            if j >= n - 1 and j - s < maxhold:
                status, why = "open", "보유중"
            else:
                why = "기간만료"
        res -= FEE * e / risk
        out.append(dict(date=d.index[i], entry=e, risk=risk, R=res, ret=res * risk / e, why=why,
                        hold=j - s, exit_date=d.index[j], status=status))
        nxt = j + 1
    return out


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]


def test_equivalence(path):
    """signals.build(df,'v3')의 BUY 날짜와 관문 전 v3raw 날짜가 같은지 확인"""
    from signals import build
    df = load_csv(path)
    a = indicators(df)
    b = build(df.copy(), "v3")
    x, y = set(a.index[a.v3raw]), set(b.index[b.signal == "BUY"])
    return x == y, len(x), len(y)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, test_equivalence(p))
