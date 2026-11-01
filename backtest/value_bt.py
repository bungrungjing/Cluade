"""저평가(PER·FCF) + v2 BUY 조합 백테스트 — score_bt.py의 지표·매매 규칙 재사용

사용법: python value_bt.py <일봉 폴더> <섹터맵 CSV> <재무 JSON> <시총 JSON>
 재무 JSON: {"EXCH:TICKER": [분기 현금흐름표 행...]} (스톡나우 get_stock_financials cash_flow)
 시총 JSON: {"EXCH:TICKER": {"close": 현재가, "market_cap_basic": 현재 시총}}

시점 처리(미래 정보 누설 방지):
 - 분기 실적은 period_end + 50일(4분기는 +80일) 뒤부터 사용 가능하다고 가정
 - TTM 순이익 = 직전 4분기 합, TTM FCF = 영업현금흐름 + 설비투자(음수) 4분기 합
 - 과거 시총 = 현재 시총 × (그날 종가 / 현재 종가) → 주식 수 변화(자사주·증자)는 반영 못 함(근사)
저평가 정의 (저평가_종목_판단_기준 1·4번을 데이터로 가능한 범위에서):
 - V_FCF: TTM FCF > 0 이고 FCF 수익률 ≥ 5%
 - V_PER: TTM 순이익 > 0 이고 PER ≤ 15
 - 저평가 = V_FCF 또는 V_PER
"""
import sys, os, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators, trades_for  # noqa: E402


def ttm_series(rows):
    q = pd.DataFrame(rows)
    q["pe"] = pd.to_datetime(q.period_end)
    q = q.sort_values("pe").drop_duplicates("pe")
    q["fcf"] = q.net_cash_from_operating_activities.fillna(0) + q.purchase_of_property_plant_and_equipment.fillna(0)
    q["ni"] = q.net_income
    q["ttm_ni"] = q.ni.rolling(4).sum()
    q["ttm_fcf"] = q.fcf.rolling(4).sum()
    lag = np.where(q.fiscal_quarter == 4, 80, 50)
    q["avail"] = q.pe + pd.to_timedelta(lag, unit="D")
    return q.dropna(subset=["ttm_ni"]).set_index("avail")[["ttm_ni", "ttm_fcf"]].sort_index()


def main():
    folder, secf, fundf, capf = sys.argv[1:5]
    fund, caps = json.load(open(fundf)), json.load(open(capf))
    rows = []
    for sym, qrows in fund.items():
        t = sym.split(":")[1]
        f = os.path.join(folder, t + ".csv")
        if not os.path.exists(f) or sym not in caps:
            continue
        df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        if len(df) < 260:
            continue
        d = indicators(df).iloc[200:]
        tt = ttm_series(qrows)
        tt = tt.reindex(d.index.union(tt.index)).ffill().reindex(d.index)
        mcap = caps[sym]["market_cap_basic"] * d.close / caps[sym]["close"]
        fy = tt.ttm_fcf / mcap * 100
        per = mcap / tt.ttm_ni
        has = tt.ttm_ni.notna()
        v_fcf = (tt.ttm_fcf > 0) & (fy >= 5)
        v_per = (tt.ttm_ni > 0) & (per <= 15)
        value = v_fcf | v_per
        gate = (d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40) & has
        buy = gate & (d.signal == "BUY")
        groups = {
            "통과 전체(기준선)": gate,
            "저평가만": gate & value,
            "고평가(저평가 아님)": gate & ~value,
            "v2 BUY 전체": buy,
            "v2 BUY + 저평가": buy & value,
            "v2 BUY + FCF 5%↑": buy & v_fcf,
            "v2 BUY + PER 15↓": buy & v_per,
            "v2 BUY + 고평가": buy & ~value,
        }
        for name, m in groups.items():
            for tr in trades_for(d, m):
                tr.update(ticker=t, group=name); rows.append(tr)
    r = pd.DataFrame(rows)
    print(f"종목 {r.ticker.nunique()}개")
    s = r.groupby("group").agg(매매=("R", "size"), 종목수=("ticker", "nunique"),
                               승률=("ret", lambda x: (x > 0).mean() * 100),
                               익절률=("why", lambda x: (x == "익절").mean() * 100),
                               손절률=("why", lambda x: (x == "손절").mean() * 100),
                               평균R=("R", "mean"), 중앙R=("R", "median"), 평균수익률=("ret", lambda x: x.mean() * 100))
    order = ["통과 전체(기준선)", "저평가만", "고평가(저평가 아님)", "v2 BUY 전체", "v2 BUY + 저평가",
             "v2 BUY + FCF 5%↑", "v2 BUY + PER 15↓", "v2 BUY + 고평가"]
    print(s.reindex(order).round(2).to_string())
    r["year"] = pd.to_datetime(r.date).dt.year
    sub = r[r.group.isin(["v2 BUY + 저평가", "v2 BUY + 고평가", "통과 전체(기준선)"])]
    print("\n연도별 평균 R (매매 수)")
    p = sub.pivot_table(index="year", columns="group", values="R", aggfunc=["mean", "size"]).round(2)
    print(p.to_string())
    # 부트스트랩: v2 BUY + 저평가 평균R의 95% 구간
    x = r[r.group == "v2 BUY + 저평가"].R.values
    if len(x) > 5:
        bs = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(2000)]
        print(f"\nv2 BUY + 저평가 평균R 95% 구간: {np.percentile(bs, 2.5):.2f} ~ {np.percentile(bs, 97.5):.2f}")
    y = r[r.group == "v2 BUY + 고평가"].R.values
    if len(y) > 5:
        bs = [np.random.default_rng(i).choice(y, len(y)).mean() for i in range(2000)]
        print(f"v2 BUY + 고평가 평균R 95% 구간: {np.percentile(bs, 2.5):.2f} ~ {np.percentile(bs, 97.5):.2f}")
    r.to_csv(os.path.join(folder, "..", "value_bt_trades.csv"), index=False)


if __name__ == "__main__":
    main()
