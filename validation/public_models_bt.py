"""공개 문헌의 레버리지·추세 모델 4개를 같은 데이터로 한 번씩 시험 (2026-11-01, 사전 등록)

사용법: python public_models_bt.py <일봉 폴더>
대상: 나스닥100(QQQ) 일간 수익률로 만든 2배·3배 근사 (실제 ETF 보수 0.95% + 차입 비용 (배수−1)×연 2% 반영)
 M1 Gayed & Bilello(2016) 'Leverage for the Long Run': 종가 > 200일선이면 다음 날 3배, 아니면 현금
 M1b 같은 규칙, 2배
 M2 변동성 목표(Moreira & Muir 2017 방식): 배수 = min(3, 연 25% ÷ 최근 20일 변동성), 200일선 아래면 현금
 M3 절대 모멘텀(Antonacci 듀얼 모멘텀의 위험 회피 부분): 월말 12개월 수익률 > 0이면 다음 달 3배, 아니면 현금
 기준: QQQ 보유, 3배 보유
 매매 비용: 배수가 바뀔 때마다 바뀐 비중 × 0.1%. 현금 이자는 0으로 둠(보수적).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3 import load_csv  # noqa: E402

RF, FEE_ETF, TC = 0.02, 0.0095, 0.001


def run(rq, lev):
    lev = lev.fillna(0)
    daily = lev * rq - (lev > 0) * FEE_ETF / 252 - np.maximum(lev - 1, 0) * RF / 252
    daily -= lev.diff().abs().fillna(0) * TC
    return (1 + daily).cumprod()


def perf(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    m = eq.resample("ME").last().pct_change().dropna()
    r12 = eq.resample("ME").last().pct_change(12).dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {"연복리": f"{100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1):.1f}%",
            "월복리 환산": f"{100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / (yrs * 12)) - 1):.2f}%",
            "최대낙폭": f"{100 * dd:.1f}%", "최악 12개월": f"{100 * r12.min():.1f}%",
            "월 +5% 이상 달": f"{100 * (m >= 0.05).mean():.0f}%",
            "12개월 +80% 이상 비율": f"{100 * (r12 >= 0.796).mean():.0f}%"}


def main():
    q = load_csv(os.path.join(sys.argv[1], "QQQ.csv.gz")).close["2009-01-01":]
    rq = q.pct_change().fillna(0)
    above = (q > q.rolling(200).mean()).shift(1, fill_value=False)
    vol = rq.rolling(20).std() * np.sqrt(252)
    vt = np.minimum(3, 0.25 / vol).shift(1)
    mom = (q.resample("ME").last().pct_change(12) > 0).shift(1)
    mom = mom.reindex(q.index, method="ffill").fillna(False).astype(bool)
    one = pd.Series(1.0, index=q.index)
    M = {"QQQ 보유(기준)": one, "3배 보유(기준)": one * 3,
         "M1 200일선 + 3배": (one * 3).where(above, 0), "M1b 200일선 + 2배": (one * 2).where(above, 0),
         "M2 변동성 목표 25% (최대 3배) + 200일선": vt.where(above, 0), "M3 12개월 모멘텀 + 3배": (one * 3).where(mom, 0)}
    start = "2011-01-01"
    t = pd.DataFrame({k: perf(run(rq[start:], v[start:])) for k, v in M.items()}).T
    print(t.to_markdown())
    for k in ("M1 200일선 + 3배", "M2 변동성 목표 25% (최대 3배) + 200일선"):
        eq = run(rq[start:], M[k][start:])
        y = eq.resample("YE").last().pct_change()
        y.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
        print(f"\n{k} 연도별:", ", ".join(f"{d.year} {100 * v:+.0f}%" for d, v in y.items()))


if __name__ == "__main__":
    main()
