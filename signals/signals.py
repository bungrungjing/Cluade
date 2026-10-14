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
    sig = df[df.signal != ""]
    cols = ["close", "signal", "adx", "k", "vwap_state", "stop", "target"]
    print(sig[cols].round(2).tail(15).to_string())
    last = df.iloc[-1]
    print(f"\n최근 {df.index[-1].date()} 종가 {last.close:.2f} | ADX {last.adx:.1f} | %K {last.k:.1f} "
          f"| VWAP {last.vwap_state} | 신호 {last.signal or '없음'}")
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
