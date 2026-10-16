"""트레이딩뷰 기술적 분석 요약 조회 (tradingview-ta 사용).

트레이딩뷰 종목 페이지의 '기술적 분석' 게이지(매수/중립/매도)와
RSI·MACD·이동평균 같은 지표 값을 가져옵니다. 로그인이 필요 없습니다.

사용법:
    python tradingview_rating.py QCOM               # 기본: NASDAQ, 일봉
    python tradingview_rating.py IBM NYSE
    python tradingview_rating.py 005930 KRX korea
    python tradingview_rating.py QCOM NASDAQ america 4h
"""
import sys

from tradingview_ta import Interval, TA_Handler

INTERVALS = {
    "1m": Interval.INTERVAL_1_MINUTE, "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES, "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS, "1d": Interval.INTERVAL_1_DAY,
    "1W": Interval.INTERVAL_1_WEEK, "1M": Interval.INTERVAL_1_MONTH,
}


def rating(symbol, exchange="NASDAQ", screener="america", interval="1d"):
    h = TA_Handler(symbol=symbol, exchange=exchange, screener=screener,
                   interval=INTERVALS[interval])
    return h.get_analysis()


if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 4
    sym = a[0] or "QCOM"
    ex, scr, iv = a[1] or "NASDAQ", a[2] or "america", a[3] or "1d"
    r = rating(sym, ex, scr, iv)
    print(f"{ex}:{sym} ({iv})  종합: {r.summary['RECOMMENDATION']}  "
          f"매수 {r.summary['BUY']} / 중립 {r.summary['NEUTRAL']} / 매도 {r.summary['SELL']}")
    print(f"  오실레이터: {r.oscillators['RECOMMENDATION']}   이동평균: {r.moving_averages['RECOMMENDATION']}")
    ind = r.indicators
    for k in ("close", "RSI", "MACD.macd", "MACD.signal", "SMA20", "SMA50", "SMA200", "EMA20"):
        if ind.get(k) is not None:
            print(f"  {k:12s} {ind[k]:.2f}")
