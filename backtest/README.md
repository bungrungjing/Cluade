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

## rule_compare.py — 매수 규칙 v1 vs v2 비교 (2026-10-16)
```
python rule_compare.py ntsk_daily.csv qcom_daily.csv spcx_daily.csv ftnt_daily.csv ffiv_daily.csv
```
진입 = BUY 신호일 종가, 손절 max(7%, 2ATR), 목표 3R, 15거래일 보유, 왕복 0.2%.
첫 실행 결과(5종목, 약 650거래일): v1 매매 1회 / v2 매매 6회(이익 4, 손실 2, 평균 +0.88R).
표본이 매우 작고 NTSK 상승 구간 비중이 커서 확정 근거로 쓰기엔 부족 → 매매 20회 후 재평가.

### 2026-10-16 보유기간 폐기 + 본전 손절 반영
청산은 손절·목표가로만 하고, +1R 도달 시 다음 날부터 손절을 진입가로 올림. 30거래일 경과는 '점검' 표시만.
v2 결과(5종목): 매매 5회 — 손절 1(−1R), 익절 1(NTSK +49%, 3R, 35일), 본전 1(SPCX 0R), 보유중 2(NTSK +1.73R, FTNT +1.52R)
본전 손절 때문에 v1 NTSK 8/27 매매는 15일 기준 +23%에서 본전(−0.2%)으로 바뀜 → 본전 손절은 이익 반납을 막는 대신 눌림 후 재상승을 놓칠 수 있음
