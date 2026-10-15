# signals

일봉 매수(BUY)·돌파(BRK)·매도(SELL) 신호 생성기. 지표 계산은 [bukosabino/ta](https://github.com/bukosabino/ta) (MIT).

```
pip install -r requirements.txt
python signals.py ../backtest/ntsk_daily.csv --plot ntsk.png
```

규칙은 signals.py 상단 주석 참고.

## regime.py — 시장 국면 + 종목 필터
```
python regime.py --qqq qqq.csv --tnx us10y.csv --cands cands.csv
```
방어/공격 국면 판정, 방어 계획 폐기 신호(①10년물 20일평균 하향 ②QQQ 20일선 회복), ATR%·베타·흑자 필터.

## journal.py — 매매 일지 통계
```
python journal.py trades.csv
```
승률, 평균 R, 국면별·변동성별 성과. 20회 전에는 평가 보류.

## 스퀴즈
- `signals.py`에 TTM 변동성 스퀴즈 추가: `sqz` 칸에 ON / SQZ↑ / SQZ↓ 표시
- `short_squeeze.py` — 숏 스퀴즈 후보 (서버에서 야후 공매도 데이터로 실행)
```
python3 short_squeeze.py ~/autotrader/screens/full_YYYYMMDD_HHMM.csv --top 300
```
