"""매수 규칙 v1 vs v2 비교 백테스트 (signals.py의 BUY 신호 사용)

사용법: python rule_compare.py ntsk_daily.csv qcom_daily.csv ...
매매 방식 (매매 원칙 기준)
 진입: BUY 신호가 난 날 종가
 손절: 진입가 − max(7%, 2ATR)   목표: 진입가 + 손절폭 × 3 (R:R 3:1)
 청산: 저가가 손절가 이하면 손절(같은 날 목표도 닿으면 손절로 처리 = 보수적),
       고가가 목표가 이상이면 익절, 15거래일이 지나면 그날 종가로 정리
 수수료·슬리피지: 왕복 0.2%
 한 번에 한 포지션만 보유 (보유 중 신호는 무시)
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "signals"))
from signals import build  # noqa: E402

HOLD, FEE = 15, 0.002


def run(df, rule):
    d = build(df.copy(), rule)
    trades, i, n = [], 0, len(d)
    while i < n:
        if d.signal.iloc[i] != "BUY":
            i += 1
            continue
        entry = d.close.iloc[i]
        risk = max(0.07 * entry, 2 * d.atr.iloc[i])
        stop, target = entry - risk, entry + 3 * risk
        exit_px, why, j = None, None, i
        for j in range(i + 1, min(i + 1 + HOLD, n)):
            if d.low.iloc[j] <= stop:
                exit_px, why = stop, "손절"
                break
            if d.high.iloc[j] >= target:
                exit_px, why = target, "익절"
                break
        if exit_px is None:
            if j == i:          # 마지막 날 신호 → 결과 없음
                break
            exit_px, why = d.close.iloc[j], "기간"
        ret = exit_px / entry - 1 - FEE
        trades.append(dict(진입일=d.index[i].date(), 진입가=round(entry, 2), 청산일=d.index[j].date(),
                           청산가=round(exit_px, 2), 사유=why, 수익률=round(ret * 100, 2),
                           R=round((exit_px - entry) / risk, 2)))
        i = j + 1
    return pd.DataFrame(trades)


def main():
    rows = []
    for path in sys.argv[1:]:
        name = os.path.basename(path).split("_")[0].split(".")[0].upper()
        df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        for rule in ("v1", "v2"):
            t = run(df, rule)
            if len(t):
                print(f"\n=== {name} {rule}\n{t.to_string(index=False)}")
            rows.append(dict(종목=name, 규칙=rule, 일수=len(df), 매매=len(t),
                             승률=round((t.수익률 > 0).mean() * 100, 1) if len(t) else None,
                             평균R=round(t.R.mean(), 2) if len(t) else None,
                             누적수익률=round(((1 + t.수익률 / 100).prod() - 1) * 100, 2) if len(t) else 0.0))
    s = pd.DataFrame(rows)
    print("\n=== 요약\n" + s.to_string(index=False))
    tot = s.groupby("규칙")[["매매"]].sum()
    print("\n=== 규칙별 합계\n" + tot.to_string())


if __name__ == "__main__":
    main()
