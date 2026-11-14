"""시가 갭(장전·선물 정보가 담긴 값)이 하루 방향을 얼마나 알려주는지 사전 등록 시험.

규칙은 고정(과최적화 방지): 갭 = 시가/전일종가-1, 장중 = 종가/시가-1, 종가수익 = 갭 + 장중.
질문: 갭을 본 뒤 남은 장중 움직임까지 갭이 맞히는가?
  1) 갭과 장중의 상관(0이면 갭은 이후를 못 맞힘)
  2) 갭 반대로 장중이 되돌린 비율(50%면 동전 던지기)
  3) 큰 갭(|갭|>=0.5%)에서 같은 방향으로 장중이 이어진 비율
주의: 종가수익에는 갭이 포함되어 있어 '갭 부호=종가 방향 일치'는 기계적으로 높게 나온다. 판단은 장중 기준으로 한다.
python3 gap_signal_test.py qqq_daily_20260625_1113.csv
"""
import csv, sys, statistics as st


def load(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: r["date"])
    for r in rows:
        for k in ("open", "high", "low", "close"):
            r[k] = float(r[k])
    return rows


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den


def main(path):
    rows = load(path)
    d = []
    for i in range(1, len(rows)):
        pc, o, c = rows[i - 1]["close"], rows[i]["open"], rows[i]["close"]
        d.append((rows[i]["date"], (o / pc - 1) * 100, (c / o - 1) * 100, (c / pc - 1) * 100))
    g = [x[1] for x in d]; oc = [x[2] for x in d]; cc = [x[3] for x in d]
    print(f"표본 {len(d)}일 ({d[0][0]} ~ {d[-1][0]})")
    print(f"갭 표준편차 {st.pstdev(g):.2f}% / 장중 표준편차 {st.pstdev(oc):.2f}% / 종가수익 표준편차 {st.pstdev(cc):.2f}%")
    print(f"갭↔장중 상관 {corr(g, oc):.2f}  (갭↔종가수익 {corr(g, cc):.2f}는 갭이 종가수익에 포함돼 기계적으로 큼)")
    use = [x for x in d if abs(x[1]) > 0.05]
    fade = sum(1 for x in use if (x[1] > 0) != (x[2] > 0))
    print(f"갭 반대로 장중 되돌림 {fade}/{len(use)} = {fade / len(use) * 100:.0f}%")
    big = [x for x in d if abs(x[1]) >= 0.5]
    cont = sum(1 for x in big if (x[1] > 0) == (x[2] > 0))
    print(f"큰 갭 {len(big)}건 중 장중이 같은 방향 {cont}건 = {cont / len(big) * 100:.0f}%")
    for sign, name in ((-1, "큰 갭다운"), (1, "큰 갭업")):
        s = [x for x in big if (x[1] > 0) == (sign > 0)]
        if s:
            up = sum(1 for x in s if x[2] > 0)
            print(f"{name} {len(s)}건: 장중 평균 {st.mean([x[2] for x in s]):+.2f}%, 장중 상승 마감 {up}건")
    mg = st.mean(g)
    beta = sum((x - mg) * (y - st.mean(oc)) for x, y in zip(g, oc)) / sum((x - mg) ** 2 for x in g)
    print(f"회귀: 장중 ≈ {beta:.2f} × 갭 (0이면 갭이 장중을 전혀 못 맞힘)")
    print("최근 5일:")
    for x in d[-5:]:
        print(f"  {x[0]} 갭 {x[1]:+.2f}% 장중 {x[2]:+.2f}% 종가 {x[3]:+.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qqq_daily_20260625_1113.csv")
