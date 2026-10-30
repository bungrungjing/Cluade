# day-trader-skill (한국어 안내)

원본: https://github.com/Jakeschincariol/day-trader-skill (MIT 라이선스, 커밋 `3d0f4ef`, 2026-10-20)
Jake Schincariol이 만든 Claude 스킬 11개 묶음입니다. 원본 파일은 수정하지 않고 그대로 넣었습니다.

## 스킬 구성

| 스킬 | 하는 일 |
|---|---|
| trader-scanner | 미국 주식 약 5,900개에서 시가 범위 돌파(ORB) 같은 셋업 찾기 |
| trader-news | 관심 종목 속보·SEC 8-K 공시 추적 |
| trader-analyst | SEC 재무 데이터로 실적 요약 |
| trader-pelosi | 미 하원의원 주식 거래 공시 읽기 |
| trader-entry | 봉 단위로 셋업을 지켜보다 진입 신호 |
| trader-exit | 진입 전에 손절가·목표가 정하기 |
| trader-timemachine | 남은 장을 10,000번 재현해 목표가/손절가 중 먼저 닿을 확률 계산 |
| trader-budget | 예산과 1회 최대 손실로 수량 계산 |
| trader-risk | 주문마다 한도 점검 (PASS / BLOCK) |
| trader-broker | 알파카(Alpaca) 주문 전송 - 기본은 모의투자, 사용자가 승인해야만 전송 |
| trader-alerts | 가격·손절·목표·장 시작 알림 |

## 설치

Claude Code에서 플러그인으로:
```
/plugin marketplace add Jakeschincariol/day-trader-skill
/plugin install day-trader-skill@day-trader-skill
```
또는 `day-trader-skill/` 안의 `trader-*` 폴더를 `~/.claude/skills/`에 복사하세요.

파이썬 3.8 이상만 있으면 되고 pip 설치는 필요 없습니다. 데이터는 야후 파이낸스, SEC EDGAR, 미 하원 공시 등 무료 공개 자료라 API 키가 없어도 됩니다. SEC를 쓰는 스킬(news, analyst)은 `SEC_USER_AGENT` 환경변수에 "이름 이메일"을 넣어야 합니다.

## 바로 해볼 수 있는 예시 (인터넷 없이)

```
python3 day-trader-skill/trader-timemachine/timemachine.py run --symbol TSLA --side buy --date 2026-10-19 \
  --entry 364.18 --stop 358.32 --target 370.04 --entry-time "10:15 AM" --setup orb \
  --offline day-trader-skill/tests/fixtures/trader-timemachine/TSLA-60d5m.json.gz --now "2026-10-19 10:50" --seed 1
```
결과: 목표가 먼저 29.6%, 손절가 먼저 25.2%, 장 마감까지 둘 다 안 닿음 45.1% → "거의 동전 던지기"

## 점검 결과 (2026-10-30)

- 동봉 테스트 315개 전부 통과 (`cd day-trader-skill && python3 -m pytest tests`)
- 코드 검토: 표준 라이브러리만 사용, 접속하는 곳은 야후·SEC·미 하원/상원 공시·나스닥 종목 목록 등 공개 데이터뿐. 외부 프로그램 호출은 PDF 변환용 `pdftotext` 하나. 자동 주문 기능은 없고 주문은 알파카 MCP를 통해 사용자 승인 후에만 나갑니다.

## 주의

- 투자 조언이 아닙니다. 원작자도 데이트레이딩은 대부분 손실을 본다고 적어 두었고, 공개된 실적 기록도 없습니다.
- 미국 장 시간(한국 기준 밤 10:30~새벽 5시, 서머타임 해제 후 밤 11:30~새벽 6시)에 맞춰 만든 도구입니다.
- 야후 차트 주소는 비공식이라 지연되거나 막힐 수 있습니다.
