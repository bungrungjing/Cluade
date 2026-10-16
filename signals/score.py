"""종합 점수 (100점) — 서버 전체 파일(full_*.csv)에 통과 조건 + 점수를 매김

사용법:
  python3 score.py full_YYYYMMDD_HHMM.csv [--enrich enrich.csv] [--regime 방어|공격] [--top 30] [--out scored.csv]

enrich.csv (선택, Claude가 트레이딩뷰·스톡나우로 채우는 보강 데이터)
  ticker,beta,pe,earn_days,fcf_pos,discount,catalyst,news_up,sqz,short_score
  - beta, pe        : 방어 국면 통과 조건 (pe 비어 있으면 적자)
  - earn_days       : 실적 발표까지 남은 일수 (보유 기간 15일 안이면 탈락)
  - fcf_pos/discount: FCF 흑자 / PER 업종 대비 할인 (1/0)
  - catalyst        : 성장 촉매 존재 (1/0)
  - news_up         : 회사 뉴스 ▲ (1/0)
  - sqz             : signals.py 스퀴즈 표시 (ON중, ON강, SQZ↑★ 등)
  - short_score     : short_squeeze.py 점수
  보강 데이터가 없는 항목은 0점 + '미확인'으로 표시 (추정으로 채우지 않음)

1단계 통과 조건 (하나라도 걸리면 탈락)
  제외 업종·OTC(excluded), 주가 5달러 이상, 시총 10억 달러 이상, ATR% 1.0~4.0 (1% 미만은 인수합병 의심), stale 데이터 아님,
  손익비 3:1 이상(산정된 경우), 방어 국면이면 베타 1.5 이하 + 흑자, 실적 15일 이내 아님
2단계 점수
  추세 25      : 50일선 위 5, 200일선 위 5, ADX>25 & +DI>−DI 10, 일목 구름 위 5
  진입 타이밍 25 : 눌림 트리거 20 — 스토캐스틱 K≤40 상향 교차(오늘·어제) 또는 20일 VWAP ±3% 이내 중 하나 (v2, 2026-10-16)
                 스크리너 매수 신호(A·B·C·추세시작) 5
  펀더멘털 20   : FCF 흑자 또는 PER 할인 10, 성장 촉매 10   (보강 데이터 필요)
  시장·섹터 15  : 같은 업종의 50일선 위 비율 50% 이상 10, 일봉 VWAP 강세·중립 5
  수급·촉매 15  : 스퀴즈 ON중·ON강·★ 5, 숏 스퀴즈 40점 이상 5, 회사 뉴스 ▲ 5   (보강 데이터 필요)
  감점          : CHOP≥61.8 −5, 스토캐스틱 K≥90 −5
등급: 80↑ A(매수 후보, 손익비 산정 필수) / 65~79 B(관찰) / ~64 C
  보강 데이터가 없으면 '환산' = 총점 ÷ 가용 만점 × 100 으로 등급 (예: 펀더·수급 미확인 → 65점 만점)
"""
import argparse
import numpy as np
import pandas as pd

ATR_MAX, BETA_MAX, MCAP_MIN, PRICE_MIN, EARN_DAYS = 4.0, 1.5, 1.0, 5.0, 15
BUY_SIGNALS = {"A", "B", "C", "추세시작"}


def _b(s):
    return s.fillna(False).astype(bool) if s.dtype == object else s.fillna(0).astype(bool)


def gates(d, defensive):
    why = pd.Series("", index=d.index)

    def flag(mask, text):
        why[mask & (why == "")] = text
    if "excluded" in d:
        flag(_b(d.excluded), "제외 업종")
    flag(d.close < PRICE_MIN, "주가 5$ 미만")
    flag(d.mcap_b < MCAP_MIN, "시총 10억$ 미만")
    flag(d.atr_pct > ATR_MAX, f"ATR {ATR_MAX}% 초과")
    flag(d.atr_pct < 1.0, "ATR 1% 미만(인수합병 진행 의심)")
    if "stale" in d:
        flag(_b(d.stale), "오래된 데이터")
    flag(d.rr.notna() & (d.rr < 3), "손익비 3 미만")
    if "earn_days" in d:
        flag(d.earn_days.notna() & (d.earn_days >= 0) & (d.earn_days <= EARN_DAYS), "실적 임박")
    if defensive and "beta" in d:
        flag(d.beta.notna() & (d.beta > BETA_MAX), "베타 1.5 초과")
        flag(d.beta.notna() & d.pe.isna(), "적자")
    return why


def score(d, defensive):
    s = pd.DataFrame(index=d.index)
    s["추세"] = (_b(d.above50) * 5 + _b(d.above200) * 5
               + ((d.adx > 25) & (d.pdi > d.mdi)) * 10 + (d.cloud == "위") * 5)
    cross = _b(d.stoch_cross_today) | _b(d.stoch_cross_yday)
    # v2 (2026-10-16): 'K≤30 교차'와 'VWAP ±3%'는 동시에 성립하기 어려워 합산 방식이면 20점이 거의 안 나옴
    #  → 눌림 트리거(K≤40 교차 또는 20일 VWAP ±3% 이내) 중 하나만 충족해도 20점
    pullback = ((d.k <= 40) & cross) | (d.vwap20_dist.abs() <= 3)
    s["타이밍"] = pullback * 20 + d.signal.isin(BUY_SIGNALS) * 5
    s["펀더"] = 0
    if "fcf_pos" in d:
        s["펀더"] += ((d.fcf_pos.fillna(0) > 0) | (d.discount.fillna(0) > 0)) * 10
    if "catalyst" in d:
        s["펀더"] += (d.catalyst.fillna(0) > 0) * 10
    breadth = d.groupby("sector").above50.transform(lambda x: _b(x).mean())
    s["시장"] = (breadth >= 0.5) * 10 + d.vwap_state.isin(["강세", "중립"]) * 5
    s["수급"] = 0
    if "sqz" in d:
        s["수급"] += d.sqz.fillna("").str.contains("ON중|ON강|★").astype(int) * 5
    if "short_score" in d:
        s["수급"] += (d.short_score.fillna(0) >= 40) * 5
    if "news_up" in d:
        s["수급"] += (d.news_up.fillna(0) > 0) * 5
    s["감점"] = -((d.chop >= 61.8) * 5 + (d.k >= 90) * 5)
    s = s.astype(int)
    s["총점"] = s.sum(axis=1).clip(lower=0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("--enrich")
    ap.add_argument("--regime", default="방어", choices=["방어", "공격"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="scored.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.full, encoding="utf-8-sig")
    if a.enrich:
        d = d.merge(pd.read_csv(a.enrich), on="ticker", how="left")
    defensive = a.regime == "방어"
    d["탈락"] = gates(d, defensive)
    s = score(d, defensive)
    d = pd.concat([d, s], axis=1)
    # 미확인 여부는 종목(행)별로 판정한다: enrich.csv에 일부 종목만 있어도
    # 보강되지 않은 종목은 해당 항목을 '미확인'으로 두고 가용 만점에서 뺀다.
    none = pd.Series(False, index=d.index)

    def known(*cols):
        out = none.copy()
        for c in cols:
            if c in d:
                out |= d[c].notna()
        return out
    fund_ok = known("catalyst", "fcf_pos", "discount")
    flow_ok = known("sqz", "short_score", "news_up")
    beta_ok = known("beta") if defensive else ~none
    miss = pd.DataFrame({"펀더": ~fund_ok, "수급": ~flow_ok, "베타·흑자": ~beta_ok})
    d["미확인"] = miss.apply(lambda r: ",".join(k for k, v in r.items() if v), axis=1)
    # 가용 만점(100 − 미확인 배점) 기준으로 환산해 등급을 매김
    d["가용만점"] = 100 - (~fund_ok) * 20 - (~flow_ok) * 15
    d["환산"] = (d.총점 / d.가용만점 * 100).round().astype(int)
    grade = np.select([d.환산 >= 80, d.환산 >= 65], ["A", "B"], "C")
    d["등급"] = np.where((grade == "A") & d.rr.isna(), "B(손익비 미산정)", grade)
    ok = d[d.탈락 == ""].sort_values(["환산", "rr"], ascending=False)
    cols = [c for c in ["ticker", "name", "sector", "close", "atr_pct", "rr", "signal", "추세", "타이밍", "펀더",
                        "시장", "수급", "감점", "총점", "가용만점", "환산", "등급"] if c in ok]
    ok[cols + ["미확인"]].to_csv(a.out, index=False, encoding="utf-8-sig")
    enriched = int((ok.미확인 != "펀더,수급,베타·흑자" if defensive else ok.미확인 != "펀더,수급").sum())
    print(f"국면 {a.regime} | 전체 {len(d)} → 통과 {len(ok)} | 보강된 종목 {enriched}개 "
          f"(미확인 항목은 종목별 가용 만점에서 제외 후 환산)")
    print(ok[cols].head(a.top).to_string(index=False))
    print("\n탈락 사유:", d[d.탈락 != ""].탈락.value_counts().to_dict())


if __name__ == "__main__":
    main()
