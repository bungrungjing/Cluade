# 시세 데이터 가져오기 (야후 파이낸스 · 트레이딩뷰)

깃허브 오픈소스 라이브러리 두 개를 쓰는 스크립트입니다.

- [yfinance](https://github.com/ranaroussi/yfinance) (Apache-2.0): 야후 파이낸스 시세·재무 데이터
- [python-tradingview-ta](https://github.com/brian-the-dev/python-tradingview-ta) (MIT): 트레이딩뷰 기술적 분석 요약과 지표 값

## 설치

```
pip install -r requirements.txt
```

## 사용법

```
# 야후: 일봉 CSV 저장 (backtest/run_bt.py에 바로 넣을 수 있는 형식)
python yahoo_fetch.py QCOM 1y
python yahoo_fetch.py 005930.KS 2y

# 트레이딩뷰: 매수/중립/매도 판정과 RSI·MACD·이평
python tradingview_rating.py QCOM
python tradingview_rating.py 005930 KRX korea
```

백테스트와 연결:

```
python yahoo_fetch.py QCOM 5y
python ../backtest/run_bt.py QCOM_yahoo.csv
```

## 주의

- 두 라이브러리 모두 공식 API가 아니라 사이트를 비공식적으로 조회합니다. 사이트 구조가 바뀌면 일시적으로 깨질 수 있고, 너무 자주 호출하면 차단될 수 있습니다. 개인 연구용으로만 쓰세요.
- tradingview-ta는 현재 시점의 지표·판정만 줍니다. 트레이딩뷰의 과거 OHLC 봉 데이터는 이 라이브러리로 받을 수 없으니, 과거 데이터는 yfinance를 쓰세요.
- 이 저장소를 만든 클라우드 환경은 외부 사이트 접속이 막혀 있어 실제 호출은 시험하지 못했습니다. 개인 PC에서 실행하세요.
