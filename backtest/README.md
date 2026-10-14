# 백테스트 실험

[backtesting.py](https://github.com/kernc/backtesting.py)(AGPL-3.0)로 만든 간단한 전략 검증 스크립트입니다.

## 실행 방법

```
pip install backtesting
python run_bt.py qcom_daily.csv
```

- `run_bt.py`: 이평 교차(SmaCross)와 RSI 스윙(RsiSwing) 전략 백테스트
- `qcom_daily.csv`: 퀄컴(QCOM) 일봉 100거래일(2026-05-21 ~ 2026-10-13), Alpha Vantage 데이터

## 실행 결과 (2026-10-14)

| 구분 | 거래 수 | 수익률 | 단순 보유 | 최대낙폭 |
|---|---|---|---|---|
| GOOG 샘플(2004~2013) / SmaCross 5,20 | 56 | 388.02% | 607.37% | -29.66% |
| QCOM 100일 / SmaCross 5,20 | 1 | 11.25% | -21.78% | -9.81% |
| QCOM 100일 / RsiSwing (RSI 35 미만 매수, 60 초과 매도, 손절 5%) | 0 | 0.0% | -25.74% | 0.0% |

## 주의

- QCOM 구간은 데이터가 100일뿐이고 거래가 1회라 통계적 의미가 없습니다. 설치와 실행 확인용입니다.
- RsiSwing 거래가 0회인 이유: 이 기간 RSI 최저가 36.3이라 35 아래로 내려간 날이 없었습니다.
- 전략 파라미터는 예시 값입니다.
