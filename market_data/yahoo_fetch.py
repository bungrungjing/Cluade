"""야후 파이낸스 일봉 다운로드 (yfinance 사용).

사용법:
    python yahoo_fetch.py QCOM              # 최근 1년 → QCOM_yahoo.csv
    python yahoo_fetch.py QCOM 5y           # 기간 지정 (1mo, 6mo, 1y, 5y, max ...)
    python yahoo_fetch.py 005930.KS 2y      # 코스피 종목은 .KS, 코스닥은 .KQ

저장 형식은 backtest/run_bt.py와 같은 timestamp,open,high,low,close,volume 입니다.
"""
import sys

import yfinance as yf


def fetch(ticker: str, period: str = "1y"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        raise SystemExit(f"{ticker}: 데이터가 없습니다. 티커를 확인하세요.")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.tz_localize(None).date
    df.index.name = "timestamp"
    df.columns = [c.lower() for c in df.columns]
    return df.round(4)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "QCOM"
    period = sys.argv[2] if len(sys.argv) > 2 else "1y"
    df = fetch(ticker, period)
    out = f"{ticker.replace('.', '_')}_yahoo.csv"
    df.to_csv(out)
    print(f"{ticker}: {len(df)}행 {df.index[0]} ~ {df.index[-1]} → {out}")
    print(df.tail())
