"""갭 + 조건이 장중(시가→종가)을 맞히는 경우가 있는지 — 사전 등록 시험 (QQQ 2010~2026).

설계 구간 2010~2021, 검증 구간 2022~ (저장소 원칙). 가설과 문턱값은 결과를 보기 전에 고정했다.
  H1 큰 갭 같은 방향 이어짐: |갭|/ATR20% >= 1.0 → 장중이 갭 방향으로 이어지는가
  H2 상승추세 중 갭다운 반등: 갭<0, |갭|/ATR20% >= 0.5, 전일 종가 > 50일선 → 장중 롱
  H3 공포 구간 갭다운 반등: 갭<0, 전일 VIX >= 25 → 장중 롱
  H4 과열 후 갭업 되돌림: 갭>0, 직전 3일 수익 >= +3% → 장중 숏(= 장중 수익이 음수인가)
  H5 연속 하락 후 갭다운 반등: 갭<0, 전일 종가 하락, 전전일 종가 하락 → 장중 롱
통과 기준(사전 등록): 같은 부호로 설계·검증 둘 다 '전체 평균 대비 초과 장중 수익'의 t>=2.6(5개 시험 보정)이고 검증 표본 >=60.
비용 0.1% 왕복(QQQ 시가 매수·종가 매도 가정)을 평균에서 뺀 값도 함께 본다.
python3 gap_conditions_test.py QQQ.csv _VIX.csv
"""
import csv, sys, math, statistics as st

COST = 0.10  # % 왕복


def load(p):
    rows = list(csv.DictReader(open(p)))
    rows.sort(key=lambda r: r["timestamp"])
    for r in rows:
        for k in ("open", "high", "low", "close"):
            r[k] = float(r[k])
    return rows


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)


def build(q, vix):
    vmap = {r["timestamp"]: r["close"] for r in vix}
    out = []
    for i in range(55, len(q)):
        pc, p2, p3, p4 = q[i - 1]["close"], q[i - 2]["close"], q[i - 3]["close"], q[i - 4]["close"]
        atr = st.mean([(q[j]["high"] - q[j]["low"]) / q[j - 1]["close"] for j in range(i - 20, i)]) * 100
        sma50 = st.mean([q[j]["close"] for j in range(i - 50, i)])
        gap = (q[i]["open"] / pc - 1) * 100
        intra = (q[i]["close"] / q[i]["open"] - 1) * 100
        out.append(dict(
            date=q[i]["timestamp"], gap=gap, intra=intra, z=gap / atr if atr else 0,
            up=pc > sma50, vix=vmap.get(q[i - 1]["timestamp"], float("nan")),
            r3=(pc / p4 - 1) * 100, d1=pc < p2, d2=p2 < p3))
    return out


HYP = {
    "H1 큰 갭 같은 방향": (lambda r: abs(r["z"]) >= 1.0, lambda r: r["intra"] * (1 if r["gap"] > 0 else -1)),
    "H2 상승추세 갭다운 롱": (lambda r: r["gap"] < 0 and abs(r["z"]) >= 0.5 and r["up"], lambda r: r["intra"]),
    "H3 VIX>=25 갭다운 롱": (lambda r: r["gap"] < 0 and r["vix"] >= 25, lambda r: r["intra"]),
    "H4 과열 갭업 숏": (lambda r: r["gap"] > 0 and r["r3"] >= 3, lambda r: -r["intra"]),
    "H5 연속하락 갭다운 롱": (lambda r: r["gap"] < 0 and r["d1"] and r["d2"], lambda r: r["intra"]),
}


def report(name, rows, label):
    cond, pay = HYP[name]
    sel = [pay(r) for r in rows if cond(r)]
    base = [r["intra"] for r in rows]  # 전체 평균(방향 무관 롱) 대비
    base_pay = st.mean(base) if name.split()[0] in ("H2", "H3", "H5") else (0.0 if name.startswith("H1") else -st.mean(base))
    if len(sel) < 2:
        return label, len(sel), float("nan"), float("nan"), float("nan"), float("nan")
    m = st.mean(sel)
    t = (m - base_pay) / (st.stdev(sel) / math.sqrt(len(sel)))
    hit = sum(1 for x in sel if x > 0) / len(sel) * 100
    return label, len(sel), m, m - base_pay, t, hit


def main(qp, vp):
    rows = build(load(qp), load(vp))
    design = [r for r in rows if r["date"] < "2022-01-01"]
    valid = [r for r in rows if r["date"] >= "2022-01-01"]
    print(f"설계 {len(design)}일({design[0]['date']}~{design[-1]['date']}) / 검증 {len(valid)}일({valid[0]['date']}~{valid[-1]['date']})")
    print(f"전체 평균 장중: 설계 {st.mean([r['intra'] for r in design]):+.3f}% / 검증 {st.mean([r['intra'] for r in valid]):+.3f}%")
    print(f"갭↔장중 상관: 설계 {corr(design):.3f} / 검증 {corr(valid):.3f}")
    print(f"{'가설':<22}{'구간':<6}{'건수':>6}{'평균%':>9}{'초과%':>9}{'t':>7}{'적중%':>7}")
    passed = []
    for name in HYP:
        res = []
        for label, rr in (("설계", design), ("검증", valid)):
            res.append(report(name, rr, label))
            l, n, m, ex, t, hit = res[-1]
            print(f"{name:<22}{l:<6}{n:>6}{m:>9.3f}{ex:>9.3f}{t:>7.2f}{hit:>7.1f}")
        ok = all(r[1] >= (60 if r[0] == "검증" else 30) and r[4] >= 2.6 for r in res) and res[1][3] > COST
        passed.append((name, ok))
    print("\n사전 등록 통과 기준(설계·검증 모두 t>=2.6, 검증 건수>=60, 비용 차감 후에도 +):")
    for name, ok in passed:
        print(f"  {name}: {'통과' if ok else '불합격'}")


def corr(rows):
    a = [r["gap"] for r in rows]; b = [r["intra"] for r in rows]
    ma, mb = st.mean(a), st.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
