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

## score.py — 종합 점수 (100점)
```
python3 score.py full_YYYYMMDD_HHMM.csv [--enrich enrich.csv] [--regime 방어|공격] [--top 30]
```
- enrich.csv에 일부 종목만 넣어도 됨: 보강되지 않은 종목은 펀더(20)·수급(15)을 '미확인'으로 두고
  종목별 `가용만점`에서 빼서 환산한다 (이전에는 파일에 칸만 있으면 전 종목을 100점 만점으로 환산하던 문제 수정).

## 바닥 전환형(트랙 B) — `bottom` 칸 (2026-10-16)
저평가_종목_판단_기준 7번 트랙 B. 200일선 아래 종목에서 ① 스토캐스틱 상승 다이버전스 ② %K 20 이하 상향 교차
③ 거래량 1.5배 양봉이 모두(②③은 최근 3일 안) 나오면 `BOTTOM`, 종가가 120일선 아래면 `BOTTOM(120↓)`(관찰 단계),
2개만 맞으면 `관찰(2/3)`. 데이터가 200·120일보다 짧으면 CSV의 `above200`·`sma120` 칸을 사용한다.
추세 트랙(BUY)과 별개이며, 실패 확률이 높아 비중은 트랙 A의 절반으로 한다.
