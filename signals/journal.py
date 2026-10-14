"""매매 일지 통계 (매매_원칙_업데이트 5번 (5)항)

사용법:
  python journal.py trades.csv

trades.csv 컬럼:
  ticker,entry_date,entry,exit_date,exit,shares,stop,target,regime,beta,atr_pct,reason
  - exit 비어 있으면 보유 중으로 처리
  - stop: 진입 시 손절가 (R 계산용)

출력: 거래별 손익%, R배수 / 전체 승률, 평균 R, 기대값, 국면별·변동성별 성과
판정 기준: 20회 이상 쌓인 뒤 기대값(평균 R) > 0 이면 전략 유지
"""
import sys
import pandas as pd


def main(path):
    t = pd.read_csv(path)
    closed = t[t.exit.notna()].copy()
    opn = t[t.exit.isna()]
    closed["ret"] = closed.exit / closed.entry - 1
    closed["R"] = (closed.exit - closed.entry) / (closed.entry - closed.stop)
    closed["pnl"] = (closed.exit - closed.entry) * closed.shares
    print(closed[["ticker", "entry_date", "entry", "exit", "ret", "R", "pnl", "regime"]]
          .round({"ret": 3, "R": 2, "pnl": 2}).to_string(index=False))
    n = len(closed)
    if n:
        win = (closed.R > 0).mean()
        print(f"\n종료 거래 {n}회 | 승률 {win:.0%} | 평균 R {closed.R.mean():.2f} | "
              f"평균 이익 R {closed.R[closed.R > 0].mean():.2f} | 평균 손실 R {closed.R[closed.R <= 0].mean():.2f} | "
              f"누적 손익 ${closed.pnl.sum():.2f}")
        if "regime" in closed:
            print("\n국면별:")
            print(closed.groupby("regime").R.agg(["count", "mean"]).round(2).to_string())
        if "atr_pct" in closed:
            closed["vol_grp"] = pd.cut(closed.atr_pct, [0, 3.5, 100], labels=["ATR≤3.5%", "ATR>3.5%"])
            print("\n변동성별:")
            print(closed.groupby("vol_grp", observed=True).R.agg(["count", "mean"]).round(2).to_string())
        if n < 20:
            print(f"\n※ 표본 {n}회 — 20회 전에는 전략 평가 보류")
    if len(opn):
        print(f"\n보유 중: {', '.join(opn.ticker)}")


if __name__ == "__main__":
    main(sys.argv[1])
