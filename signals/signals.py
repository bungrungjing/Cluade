"""일봉 매수/매도 신호 생성기 (bukosabino/ta 라이브러리 사용, MIT)

사용법: python signals.py ../backtest/ntsk_daily.csv [--plot out.png]
CSV 컬럼: timestamp,open,high,low,close,volume

신호 규칙 (매매 원칙 문서 기준)
 매수(BUY):  ADX>25, +DI>-DI, 종가>SMA50, 스토캐스틱 %K가 30 이하에서 %D 상향 돌파,
             일봉 VWAP 상태가 '약세'가 아닐 것
 돌파(BRK):  종가가 직전 20일 고점 돌파 + ADX>30 + VWAP '강세'
 매도(SELL): 스토캐스틱 %K가 80 이상에서 %D 하향 돌파, 또는
             MACD 데드크로스 + 종가<SMA20
 손절가:     진입가 - max(7%, 2ATR)  /  목표가: 손절폭의 3배 (R:R 3:1)

변동성 스퀴즈 (멀티 스퀴즈 프로 방식, 존 카터 Squeeze Pro 3단계 + 주봉)
 압축 단계 : 볼린저밴드(20, 2σ)가 켈트너채널(20, k·ATR) 안쪽일 때
             k=2.0 → 약(1) / k=1.5 → 중(2, 기존 TTM) / k=1.0 → 강(3)  ※ 숫자가 클수록 강한 압축
 sqz_lvl   : 일봉 압축 단계 0~3,  sqz_w : 주봉 압축 단계 0~3 (직전 확정 주봉 기준)
 sqz_fire  : 전날 압축(1단계 이상) → 오늘 해제 = 발사
 sqz_mom   : 모멘텀 = 종가 − (20일 고저 중간값과 SMA20의 평균)의 20일 선형회귀값
 표시      : ON약/ON중/ON강 (+주봉ON) / 발사 시 SQZ↑ 또는 SQZ↓ (+주봉ON이면 '★' = 다중 시간대 일치)
 ※ 방향은 맞히지 못함 → 추세(ADX·DI·50일선)와 함께 판단
"""
import sys, argparse
import pandas as pd
from ta.trend import ADXIndicator, SMAIndicator, MACD
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange


def daily_vwap_state(df):
    tp = (df.high + df.low + df.close) / 3
    pv = tp * df.volume
    v5 = pv.rolling(5).sum() / df.volume.rolling(5).sum()
    v20 = pv.rolling(20).sum() / df.volume.rolling(20).sum()
    rising = v20 > v20.shift(5)
    state = pd.Series("중립", index=df.index)
    state[(df.close > v5) & (v5 > v20) & rising] = "강세"
    state[(df.close < v20) & ~rising] = "약세"
    return v20, state


def squeeze_pro(df, n=20, bb_k=2.0):
    """멀티 스퀴즈 프로: (압축 단계 0~3, 발사 여부, 모멘텀) 반환."""
    import numpy as np
    c = df.close
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std()
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    lvl = pd.Series(0, index=df.index)
    for k, v in ((2.0, 1), (1.5, 2), (1.0, 3)):          # 좁은 켈트너 안에 들어갈수록 강한 압축
        inside = (bb_k * sd) < (k * atr)
        lvl[inside] = v
    fire = (lvl.shift(1, fill_value=0) > 0) & (lvl == 0)
    base = ((df.high.rolling(n).max() + df.low.rolling(n).min()) / 2 + mid) / 2
    x = np.arange(n)

    def linreg_last(y):
        b, a = np.polyfit(x, y, 1)
        return a + b * (n - 1)
    mom = (c - base).rolling(n).apply(linreg_last, raw=True)
    return lvl, fire, mom


def weekly_squeeze(df):
    """주봉 압축 단계를 일봉 날짜에 맞춰 반환 (직전 확정 주봉 기준)."""
    w = df[["open", "high", "low", "close", "volume"]].resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if len(w) < 25:
        return pd.Series(0, index=df.index)
    lvl, _, _ = squeeze_pro(w)
    lvl.index = lvl.index + pd.Timedelta(days=1)          # 금요일 마감 후부터 적용 (미래 참조 방지)
    return lvl.reindex(df.index, method="ffill").fillna(0).astype(int)


def build(df):
    adx = ADXIndicator(df.high, df.low, df.close, 14)
    df["adx"], df["pdi"], df["mdi"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["sma20"] = SMAIndicator(df.close, 20).sma_indicator()
    df["sma50"] = SMAIndicator(df.close, 50).sma_indicator()
    st = StochasticOscillator(df.high, df.low, df.close, 14, 3)
    df["k"], df["d"] = st.stoch(), st.stoch_signal()
    m = MACD(df.close)
    df["macd"], df["macd_sig"] = m.macd(), m.macd_signal()
    df["atr"] = AverageTrueRange(df.high, df.low, df.close, 14).average_true_range()
    df["vwap20"], df["vwap_state"] = daily_vwap_state(df)

    k_up = (df.k > df.d) & (df.k.shift() <= df.d.shift()) & (df.k.shift() <= 30)
    k_dn = (df.k < df.d) & (df.k.shift() >= df.d.shift()) & (df.k.shift() >= 80)
    macd_dn = (df.macd < df.macd_sig) & (df.macd.shift() >= df.macd_sig.shift())

    buy = (df.adx > 25) & (df.pdi > df.mdi) & (df.close > df.sma50) & k_up & (df.vwap_state != "약세")
    brk = (df.close > df.high.rolling(20).max().shift()) & (df.adx > 30) & (df.vwap_state == "강세")
    sell = k_dn | (macd_dn & (df.close < df.sma20))

    df["signal"] = ""
    df.loc[sell, "signal"] = "SELL"
    df.loc[brk, "signal"] = "BRK"
    df.loc[buy, "signal"] = "BUY"
    df["sqz_lvl"], df["sqz_fire"], df["sqz_mom"] = squeeze_pro(df)
    df["sqz_w"] = weekly_squeeze(df)
    df["sqz_on"] = df.sqz_lvl > 0
    name = {1: "ON약", 2: "ON중", 3: "ON강"}
    df["sqz"] = df.sqz_lvl.map(name).fillna("")
    up = df.sqz_fire & (df.sqz_mom > 0) & (df.sqz_mom > df.sqz_mom.shift())
    dn = df.sqz_fire & (df.sqz_mom < 0) & (df.sqz_mom < df.sqz_mom.shift())
    df.loc[up, "sqz"] = "SQZ↑"
    df.loc[dn, "sqz"] = "SQZ↓"
    wk = df.sqz_w > 0
    df.loc[wk & df.sqz.str.startswith("SQZ"), "sqz"] += "★"
    df.loc[wk & df.sqz.str.startswith("ON"), "sqz"] += "+주봉ON"
    risk = (df.close * 0.07).combine(2 * df.atr, max)
    df["stop"] = (df.close - risk).round(2)
    df["target"] = (df.close + 3 * risk).round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plot")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df = build(df)
    sig = df[(df.signal != "") | df.sqz.str.startswith("SQZ")]
    cols = ["close", "signal", "sqz", "adx", "k", "vwap_state", "stop", "target"]
    print(sig[cols].round(2).tail(15).to_string())
    last = df.iloc[-1]
    on_days = int(df.sqz_on[::-1].cumprod().sum())
    print(f"\n최근 {df.index[-1].date()} 종가 {last.close:.2f} | ADX {last.adx:.1f} | %K {last.k:.1f} "
          f"| VWAP {last.vwap_state} | 신호 {last.signal or '없음'} | 스퀴즈 "
          f"{(last.sqz or 'OFF') + (f' ({on_days}일째)' if on_days else '')} | 주봉 압축 {int(last.sqz_w)}단계 "
          f"| 모멘텀 {last.sqz_mom:+.2f}")
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df.close, lw=1, label="close")
        ax.plot(df.index, df.sma50, lw=.8, label="SMA50")
        ax.plot(df.index, df.vwap20, lw=.8, ls="--", label="VWAP20")
        for s, c, mk in [("BUY", "g", "^"), ("BRK", "b", "^"), ("SELL", "r", "v")]:
            p = df[df.signal == s]
            ax.scatter(p.index, p.close, c=c, marker=mk, s=60, label=s, zorder=3)
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)


if __name__ == "__main__":
    main()
