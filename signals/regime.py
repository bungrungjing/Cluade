"""시장 국면 판정 + 종목 필터 (매매_원칙_업데이트 5번 조항 자동화)

사용법:
  python regime.py --qqq qqq.csv --tnx us10y.csv [--cands cands.csv]

입력 CSV
  qqq.csv / us10y.csv : timestamp,close[,volume]   (일봉, 과거→최신 순서 상관없음)
  cands.csv           : ticker,close,atr,beta,pe   (pe 비어 있으면 적자로 간주)

판정 규칙
  방어 국면 : 10년물이 20일 평균 위에서 상승 중(5일 전보다 높음)  또는  QQQ 종가 < 20일선
  공격 국면 : 그 외 (10년물 20일 평균 아래 하락 + QQQ 20일선 위)
  방어 계획 즉시 폐기 신호 (하나라도 해당 시 출력)
    ① 10년물 종가가 20일 평균 아래로 내려가고 5일 전보다 낮음
    ② QQQ 종가가 20일선 위로 회복 + 거래량 20일 평균 이상
  (뉴스 신호 ③~⑤는 Claude가 뉴스로 별도 확인)

종목 필터
  공통   : ATR% = ATR/종가 <= 4.0% (2ATR = 8%, 손절 7~8% 원칙과 일치)
  방어   : + 베타 <= 1.5 + 흑자(PER 존재)
"""
import argparse
import pandas as pd

ATR_MAX = 0.040
BETA_MAX = 1.5


def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    return df


def regime(qqq, tnx):
    q, t = qqq.copy(), tnx.copy()
    q["ma20"] = q.close.rolling(20).mean()
    t["ma20"] = t.close.rolling(20).mean()
    ql, tl = q.iloc[-1], t.iloc[-1]
    t_rising = tl.close > t.close.iloc[-6]
    defensive = (tl.close > tl.ma20 and t_rising) or (ql.close < ql.ma20)

    signals = []
    if tl.close < tl.ma20 and not t_rising:
        signals.append("① 10년물 20일 평균 아래 + 하락")
    if "volume" in q and ql.close > ql.ma20 and q.close.iloc[-2] <= q.ma20.iloc[-2] \
            and ql.volume >= q.volume.rolling(20).mean().iloc[-1]:
        signals.append("② QQQ 20일선 회복 + 거래량 동반")

    info = {
        "date": str(q.index[-1].date()),
        "qqq": round(ql.close, 2), "qqq_ma20": round(ql.ma20, 2),
        "tnx": round(tl.close, 3), "tnx_ma20": round(tl.ma20, 3),
        "tnx_5d_ago": round(t.close.iloc[-6], 3),
        "regime": "방어" if defensive else "공격",
        "exit_signals": signals,
    }
    return info


def screen(cands, defensive):
    c = cands.copy()
    c["atr_pct"] = c.atr / c.close
    c["profitable"] = c.pe.notna() & (c.pe > 0)
    c["pass"] = c.atr_pct <= ATR_MAX
    if defensive:
        c["pass"] &= (c.beta <= BETA_MAX) & c.profitable
    why = []
    for _, r in c.iterrows():
        w = []
        if r.atr_pct > ATR_MAX:
            w.append(f"ATR {r.atr_pct:.1%}")
        if defensive and r.beta > BETA_MAX:
            w.append(f"β{r.beta:.2f}")
        if defensive and not r.profitable:
            w.append("적자")
        why.append(", ".join(w) or "-")
    c["reason"] = why
    return c.sort_values(["pass", "atr_pct"], ascending=[False, True])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qqq", required=True)
    ap.add_argument("--tnx", required=True)
    ap.add_argument("--cands")
    a = ap.parse_args()
    info = regime(load(a.qqq), load(a.tnx))
    print(f"[{info['date']}] 국면: {info['regime']}")
    print(f"  QQQ {info['qqq']} / 20일선 {info['qqq_ma20']}")
    print(f"  10년물 {info['tnx']} / 20일평균 {info['tnx_ma20']} / 5일전 {info['tnx_5d_ago']}")
    if info["exit_signals"]:
        print("  ★ 방어 계획 폐기 신호: " + " / ".join(info["exit_signals"]))
    else:
        print("  폐기 신호 없음 (뉴스 ③~⑤는 별도 확인)")
    if a.cands:
        c = screen(pd.read_csv(a.cands), info["regime"] == "방어")
        c["atr_pct"] = (c.atr_pct * 100).round(1)
        print(c[["ticker", "close", "atr_pct", "beta", "pe", "pass", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
