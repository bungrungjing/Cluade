"""진입 선별 탐색 — 청산은 고정(방어형 3R / 수익추종 추적), 진입일 특징을 나눠 승률·수익률이 같이 오르는지 찾는다

사용법: python entry_bt.py <일봉 폴더> <섹터맵 CSV>
 표본: 통과 종목의 모든 거래일 중 10거래일 간격으로 뽑은 진입일 (겹침 상관 완화), 두 청산 방식으로 각각 시뮬레이션
 탐색: 특징 N개를 학습 구간(2022-10~2024-12)의 3분위로 나눠 '더 좋은 쪽'을 고르고, 검증 구간(2025~)에서 같은 쪽이 기준선을 이기는지 확인
 채택 기준: 두 청산 방식 모두에서 승률·평균수익률 모두 학습·검증 둘 다 기준선보다 높을 것 (+ 3개 시기 구간 중 2개 이상)
 → 20개 안팎을 뒤지므로 우연히 통과하는 것이 섞일 수 있음. 마지막에 v3 BUY 매매(실제 규칙)로 한 번 더 확인
"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bt import indicators  # noqa: E402
from combo_bt import sim, DEF, TRD  # noqa: E402

SPLIT = pd.Timestamp("2025-01-01")
folder, secf = sys.argv[1:3]
sec = pd.read_csv(secf).set_index("ticker").sector.to_dict()
data = {}
for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    df = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if len(df) < 260: continue
    data[os.path.basename(f)[:-4]] = indicators(df)
tick = list(data)
a50 = pd.DataFrame({t: data[t].above50 for t in tick})
ret63 = pd.DataFrame({t: data[t].close.pct_change(63) for t in tick})
rs_rank = ret63.rank(axis=1, pct=True)
sec_s = pd.Series({t: sec.get(t) for t in tick})
sec_breadth = {s: a50[[t for t in tick if sec_s[t] == s]].mean(axis=1)
               for s in sec_s.dropna().unique() if (sec_s == s).sum() >= 3}

rows = []
for t in tick:
    d = data[t].copy()
    d["relvol"] = d.volume / d.volume.rolling(20).mean()
    F = pd.DataFrame(index=d.index)
    F["거래량배수"] = d.relvol
    F["ATR수축(20/60)"] = d.atr.rolling(20).mean() / d.atr.rolling(60).mean()
    F["거래량건조(5/50)"] = d.volume.rolling(5).mean() / d.volume.rolling(50).mean()
    F["갭(시가/전일종가)"] = (d.open / d.close.shift() - 1) * 100
    F["종가위치(고저내)"] = (d.close - d.low) / (d.high - d.low).replace(0, np.nan)
    F["52주고점대비"] = (d.close / d.close.rolling(252, min_periods=120).max() - 1) * 100
    F["20일선거리"] = (d.close / d.sma20 - 1) * 100
    F["50일선거리"] = (d.close / d.sma50 - 1) * 100
    F["200일선거리"] = (d.close / d.sma200 - 1) * 100
    F["5일수익률"] = d.close.pct_change(5) * 100
    F["20일수익률"] = d.close.pct_change(20) * 100
    F["63일수익률"] = d.close.pct_change(63) * 100
    F["상대강도순위"] = rs_rank[t]
    F["섹터폭"] = sec_breadth.get(sec.get(t), pd.Series(np.nan, index=d.index)).reindex(d.index)
    F["시장폭"] = a50.mean(axis=1).reindex(d.index) * 100
    F["ADX"] = d.adx
    F["스토캐스틱K"] = d.k
    F["CHOP"] = d.chop
    F["ATR%"] = d.atr_pct
    F["연속상승일"] = (d.close.diff() > 0).astype(int).groupby((d.close.diff() <= 0).cumsum()).cumsum()
    gate = ((d.close >= 5) & d.atr_pct.between(1, 4) & (d.dvol20 >= 40)).iloc[200:]
    idx = np.where(gate.values)[0][::10] + 200
    for i in idx:
        if i >= len(d) - 2: continue
        m = pd.Series(False, index=d.index); m.iloc[i] = True
        for nm, kw in (("방어", DEF), ("추종", TRD)):
            tr = sim(d, m, **kw)
            if tr:
                rows.append({**F.iloc[i].to_dict(), "date": d.index[i], "ticker": t, "model": nm,
                             "ret": tr[0]["ret"], "v3": bool(d.signal.iloc[i] == "BUY" and d.relvol.iloc[i] >= 1.2)})
R = pd.DataFrame(rows); R["win"] = R.ret > 0
tr_, te_ = R[R.date < SPLIT], R[R.date >= SPLIT]
print(f"표본 {R.groupby('model').size().to_dict()}  (학습 {len(tr_)//2}, 검증 {len(te_)//2} 진입일)")
for nm in ("방어", "추종"):
    for lab, s in (("학습", tr_), ("검증", te_)):
        x = s[s.model == nm]; print(f"  기준선 {nm} {lab}: 승률 {x.win.mean()*100:.1f}%  평균수익률 {x.ret.mean()*100:+.2f}%")
feats = [c for c in R.columns if c not in ("date", "ticker", "model", "ret", "win", "v3")]
base = {(nm, lab): (s[s.model == nm].win.mean(), s[s.model == nm].ret.mean()) for nm in ("방어", "추종") for lab, s in (("학습", tr_), ("검증", te_))}
out = []
folds = [(pd.Timestamp("2022-09-01"), pd.Timestamp("2024-01-01")), (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")), (pd.Timestamp("2025-01-01"), pd.Timestamp("2027-01-01"))]
for f in feats:
    ent = tr_[tr_.model == "추종"][f].dropna()
    q1, q2 = ent.quantile([1 / 3, 2 / 3])
    best = None
    for side, fn in (("낮은 쪽", lambda s: s[f] <= q1), ("높은 쪽", lambda s: s[f] >= q2)):
        x = tr_[fn(tr_)]
        sc = x[x.model == "추종"].ret.mean() + x[x.model == "방어"].ret.mean()
        if best is None or sc > best[0]: best = (sc, side, fn)
    _, side, fn = best
    rec = dict(특징=f, 쪽=side, 기준=f"{q1:.2f}" if side == "낮은 쪽" else f"{q2:.2f}")
    ok = True; nfold = 0
    for nm in ("방어", "추종"):
        for lab, s in (("학습", tr_), ("검증", te_)):
            x = s[(s.model == nm)]; x = x[fn(x)]
            dw, dr = x.win.mean() - base[(nm, lab)][0], x.ret.mean() - base[(nm, lab)][1]
            rec[f"{nm}{lab}Δ승률"] = dw * 100; rec[f"{nm}{lab}Δ수익"] = dr * 100; rec["n" + lab] = len(x)
            if not (dw > 0 and dr > 0): ok = False
    for a, b in folds:
        s = R[(R.date >= a) & (R.date < b)]
        gain = []
        for nm in ("방어", "추종"):
            x = s[s.model == nm]; xs = x[fn(x)]
            gain.append(len(xs) >= 15 and xs.ret.mean() > x.ret.mean() and xs.win.mean() > x.win.mean())
        nfold += all(gain)
    rec["통과"] = ok and nfold >= 2; rec["시기통과"] = nfold
    out.append(rec)
O = pd.DataFrame(out)
cols = ["특징", "쪽", "기준", "n학습", "n검증", "방어학습Δ승률", "방어학습Δ수익", "방어검증Δ승률", "방어검증Δ수익", "추종학습Δ승률", "추종학습Δ수익", "추종검증Δ승률", "추종검증Δ수익", "시기통과", "통과"]
print("\n특징별 (Δ = 기준선 대비 %p, 학습에서 고른 쪽을 검증에 그대로 적용)")
print(O[cols].sort_values("통과", ascending=False).round(1).to_string(index=False))
P = O[O.통과]
print(f"\n통과 {len(P)}개 / {len(O)}개 (우연 기대치: 20개 중 기준이 엄격해도 0~1개)")
R.to_csv(os.path.join(folder, "..", "entry_bt_samples.csv"), index=False)
# v3 BUY 실제 규칙 확인
for _, p in P.iterrows():
    f = p.특징
    ent = tr_[tr_.model == "추종"][f].dropna(); q1, q2 = ent.quantile([1 / 3, 2 / 3])
    fn = (lambda s: s[f] <= q1) if p.쪽 == "낮은 쪽" else (lambda s: s[f] >= q2)
    v = R[R.v3]
    for nm in ("방어", "추종"):
        x = v[v.model == nm]; xs = x[fn(x)]
        print(f"[v3 BUY 확인] {f} {p.쪽}: {nm} 전체 {len(x)}회 승률 {x.win.mean()*100:.0f}%/{x.ret.mean()*100:+.2f}% → 조건 {len(xs)}회 {xs.win.mean()*100:.0f}%/{xs.ret.mean()*100:+.2f}%")
