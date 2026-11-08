"""우위 찾기 2차 후속 — 사전 등록 (2026-11-08, edge_hunt.py 결과를 본 뒤 만든 후속이라 다중비교 주의)

질문: 200일선 필터(아래면 3배 정리)와 공포 매수(시장 폭 ≤30%면 산다)가 충돌한다. 둘을 합치면?
기준 C = 3배 + QQQ 200일선(아래면 현금)
F  = C + 시장 폭 ≤30% 사건 다음 날부터 60거래일은 200일선과 상관없이 3배 보유
G  = QQQ 보유 + 시장 폭 ≤30% 사건 때 60거래일 2배 (3배 대신)
H  = 2배 + 200일선 + 공포 60일 예외(F의 2배판)
합격: F는 C보다 연복리가 두 구간 모두 높고 최대 낙폭이 C보다 5%p 넘게 나쁘지 않음
      G는 연복리가 두 구간 모두 QQQ보다 높고 최대 낙폭이 QQQ의 1.5배(−52.7%) 이내
      H는 연복리가 두 구간 모두 QQQ보다 높고 최대 낙폭이 QQQ의 1.5배 이내
비용: 3배 연 3%, 2배 연 2% (근사)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from edge_hunt import rd, half, stats, REPO
import subprocess
names=[n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
qqq=rd("daily/QQQ.csv.gz")
ab={}
for p in names:
    t=os.path.basename(p).split(".")[0]
    if t in("SPY","QQQ","_VIX"): continue
    try: d=rd(p)
    except Exception: continue
    if len(d)<300: continue
    s=d.close.rolling(50).mean(); ab[t]=(d.close>s).where(s.notna())
a=pd.DataFrame(ab); br=a.mean(axis=1)[a.count(axis=1)>=200].reindex(qqq.index).ffill()
qr=qqq.close.pct_change().fillna(0); l3=3*qr-0.03/252; l2=2*qr-0.02/252
up=(qqq.close>qqq.close.rolling(200).mean()).shift(1,fill_value=False)
f30=br<=0.30; ev=f30&~f30.shift(1).rolling(20,min_periods=1).max().fillna(0).astype(bool)
on=pd.Series(False,index=qqq.index)
for i in np.where(ev.shift(1,fill_value=False).values)[0]: on.iloc[i:i+60]=True
S={"A QQQ 보유":qr,"C 3배+200일선":l3.where(up,0.0),"F C + 공포 60일 예외":l3.where(up|on,0.0),
   "G QQQ + 공포 60일 2배":qr.where(~on,l2),"H 2배+200일선+공포 예외":l2.where(up|on,0.0),"(참고) 2배+200일선":l2.where(up,0.0)}
st=qqq.index>="2010-04-01"; out=["\n## 후속(사전 등록, 다중비교 주의) — 200일선 필터 × 공포 매수","| 전략 | 연복리 | 최대낙폭 | 최악 연도 | 플러스 달 | 2010~17 | 2018~26 |","|---|---|---|---|---|---|---|"]
for k,r in S.items():
    r=r[st]; c,m,_=stats(r,252); yr=(1+r).groupby(r.index.year).prod()-1; mo=(1+r).groupby(r.index.to_period("M")).prod()-1
    hv=[stats(r[half(r.index)==h],252)[0] for h in("2010~17","2018~26")]
    out.append(f"| {k} | {c*100:.1f}% | {m*100:.1f}% | {yr.min()*100:.1f}% | {(mo>0).mean()*100:.0f}% | {hv[0]*100:.1f}% | {hv[1]*100:.1f}% |")
# 공포 사건 시점에 QQQ가 200일선 아래였던 비율
out.append(f"\n공포 사건 {int(ev.sum())}회 중 그날 QQQ가 200일선 아래였던 사건: {int((ev & ~(qqq.close>qqq.close.rolling(200).mean())).sum())}회")
print("\n".join(out))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","edge_hunt_2026-11-08.md"),"a").write("\n".join(out)+"\n")
