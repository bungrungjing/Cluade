"""숏 스퀴즈 후보 스캐너 (서버에서 실행 — 야후 파이낸스 공매도 데이터 사용)

사용법 (서버):
  python3 short_squeeze.py ~/autotrader/screens/full_YYYYMMDD_HHMM.csv [--top 300] [--out squeeze.csv]
  python3 short_squeeze.py --tickers GME,AMC,BYND          # 특정 종목만

1단계 (전체 파일로 1차 압축, 공매도 조회 횟수 줄이기)
  주가 5달러 이상, 시총 3억 달러 이상, OTC·제외 업종 제외,
  그리고 [거래량 1.5배 이상] 또는 [20일 고점 5% 이내] 또는 [5일 수익률 +5% 이상] 중 하나
2단계 (야후 공매도 데이터)
  short_pct_float : 유통주식 대비 공매도 비율 (%)
  days_to_cover   : 공매도 청산 소요일 (공매도 수량 ÷ 일평균 거래량)
  short_chg       : 전월 대비 공매도 수량 변화 (%)
점수 (최대 100)
  공매도 비율    20%↑ 30점 / 15%↑ 20점 / 10%↑ 10점
  청산 소요일    5일↑ 20점 / 3일↑ 10점
  공매도 증가    전월 대비 +10%↑ 10점 (쌓이는 중)
  촉발 신호      거래량 2배↑ 15점, 20일 고점 2% 이내(돌파 시도) 15점, 5일 +5%↑ 10점
  판정: 60점↑ "스퀴즈 임박", 40점↑ "관찰", 그 외 제외
※ 야후 공매도 데이터는 FINRA 2주 발표 기준이라 최대 2주 늦음
※ 숏 스퀴즈 종목은 대부분 고베타·고변동 → 방어 국면에서는 관찰만, 매수는 공격 국면 + ATR% 필터 통과 시
"""
import argparse
import time
import pandas as pd


def prefilter(full, top):
    d = full.copy()
    if "excluded" in d:
        d = d[~d.excluded.astype(bool)]
    d = d[(d.close >= 5) & (d.mcap_b >= 0.3)]
    near_hi = d.close >= d.hi20 * 0.95
    trig = (d.relvol >= 1.5) | near_hi | (d.ret5d >= 5)
    d = d[trig].copy()
    d["pre"] = d.relvol.fillna(0) + near_hi.astype(int) + (d.ret5d.fillna(0) / 10)
    return d.sort_values("pre", ascending=False).head(top)


def short_data(tickers, pause=0.4):
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            sp = info.get("shortPercentOfFloat")
            sh, prev = info.get("sharesShort"), info.get("sharesShortPriorMonth")
            rows.append(dict(ticker=t,
                             short_pct_float=round(sp * 100, 1) if sp else None,
                             days_to_cover=info.get("shortRatio"),
                             short_chg=round((sh / prev - 1) * 100, 1) if sh and prev else None,
                             float_m=round(info["floatShares"] / 1e6, 1) if info.get("floatShares") else None))
        except Exception as e:
            rows.append(dict(ticker=t, error=str(e)[:60]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(tickers)}")
        time.sleep(pause)
    return pd.DataFrame(rows)


def score(df):
    s = pd.Series(0, index=df.index, dtype=float)
    sp = df.short_pct_float.fillna(0)
    s += (sp >= 20) * 30 + ((sp >= 15) & (sp < 20)) * 20 + ((sp >= 10) & (sp < 15)) * 10
    dtc = df.days_to_cover.fillna(0)
    s += (dtc >= 5) * 20 + ((dtc >= 3) & (dtc < 5)) * 10
    s += (df.short_chg.fillna(0) >= 10) * 10
    if "relvol" in df:
        s += (df.relvol.fillna(0) >= 2) * 15
        if "hi20" in df:   # hi20은 오늘 고가 포함 → 2% 이내면 돌파 시도로 간주
            s += (df.close >= df.hi20 * 0.98).fillna(False) * 15
        s += (df.ret5d.fillna(0) >= 5) * 10
    df["score"] = s.astype(int)
    df["verdict"] = pd.cut(df.score, [-1, 39, 59, 100], labels=["제외", "관찰", "스퀴즈 임박"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full", nargs="?")
    ap.add_argument("--tickers")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--out", default="squeeze.csv")
    a = ap.parse_args()
    if a.tickers:
        base = pd.DataFrame({"ticker": [t.strip().upper() for t in a.tickers.split(",")]})
    else:
        base = prefilter(pd.read_csv(a.full), a.top)
        print(f"1차 압축: {len(base)}종목 → 야후 공매도 조회")
    sd = short_data(base.ticker.tolist())
    df = score(base.merge(sd, on="ticker", how="left"))
    cols = [c for c in ["ticker", "name", "sector", "close", "atr_pct", "relvol", "ret5d", "short_pct_float",
                        "days_to_cover", "short_chg", "float_m", "score", "verdict"] if c in df]
    df = df.sort_values("score", ascending=False)
    df[cols].to_csv(a.out, index=False, encoding="utf-8-sig")
    print(df[df.score >= 40][cols].head(30).to_string(index=False))
    print(f"\n저장: {a.out}  (스퀴즈 임박 {int((df.score >= 60).sum())} / 관찰 {int(((df.score >= 40) & (df.score < 60)).sum())})")


if __name__ == "__main__":
    main()
