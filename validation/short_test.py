"""수익용 숏(인버스) 규칙 사전 등록 시험 (2026-11-08, 사용자: "숏은 ATR 신경 쓰며 할 게 아니다, 수익용이어야")

ATR 필터 없음. 수량은 ATR로 조절(손절 시 계좌 손실 일정) → 결과는 R(손절폭 대비 배수)로 비교.
대상(기초 지수에 대해 숏 = 인버스 ETF 근사): QQQ, SPY, 그리고 data 갈래 종목으로 만든 업종 균등지수 11개(sector 칸)
규칙 S1(기존 문서 규칙): 기초가 50일선 아래 & 20일선 < 50일선 & 종가가 20일선 ±1% 안(반등 시점) → 다음 날부터 숏
      손절: 진입가 + 2ATR(기초 기준) / 목표: 진입가 − 3×(2ATR) / 최대 10거래일 / 왕복 0.2%
규칙 S2(추세 붕괴 추격): 기초 종가가 20일 신저가 & 50일선 아래 → 숏, 손절·목표·기간 S1과 같음
규칙 S3(롱 대비 기준선): 아무 날 숏(같은 청산) — 시장 상승 편향 때문에 음수일 것
합격: 평균 R > 0, 95% 구간 하한 > 0, 2010~17·2018~26 두 구간 모두 양수 (지수 2개 합산 / 업종 11개 합산 각각)
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from edge_hunt import rd, REPO
FEE=0.002
def ind(c,h,l):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return c.rolling(20).mean(), c.rolling(50).mean(), tr.ewm(alpha=1/14,adjust=False).mean()
def sim(c,h,l,atr,sig):
    out=[];i=0;n=len(c);C,H,L,A,S=c.values,h.values,l.values,atr.values,sig.values
    while i<n-1:
        if not S[i] or not np.isfinite(A[i]): i+=1; continue
        e=C[i]; risk=2*A[i]; stop=e+risk; tgt=e-3*risk; px=None
        for j in range(i+1,min(n,i+11)):
            if H[j]>=stop: px=stop;break
            if L[j]<=tgt: px=tgt;break
        if px is None: px=C[j]
        out.append((c.index[i],(e-px)/risk-FEE*e/risk)); i=j+1
    return out
def rules(c,h,l):
    s20,s50,atr=ind(c,h,l)
    down=(c<s50)&(s20<s50)
    S1=down&((c/s20-1).abs()<=0.01)
    S2=(c<=c.rolling(20).min())&(c<s50)
    S3=s50.notna()
    return atr,{"S1 반등 숏":S1,"S2 신저가 추격 숏":S2,"S3 아무 날 숏":S3}
uni=pd.read_csv(os.path.join(REPO,"validation","universe.csv")).set_index("ticker")
names=[n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
C={}
for p in names:
    t=os.path.basename(p).split(".")[0]
    if t in("SPY","QQQ","_VIX"): continue
    try: d=rd(p)
    except Exception: continue
    if len(d)>=300: C[t]=d.close
close=pd.DataFrame(C); ret=close.pct_change(fill_method=None).clip(-0.5,1)
rows=[]
for nm in("QQQ","SPY"):
    d=rd(f"daily/{nm}.csv.gz"); atr,R=rules(d.close,d.high,d.low)
    for k,s in R.items():
        for dt,r in sim(d.close,d.high,d.low,atr,s): rows.append(("지수",nm,k,dt,r))
for sec,g in uni.groupby("sector"):
    cols=[t for t in g.index if t in close]
    if len(cols)<15: continue
    idx=(1+ret[cols].mean(axis=1).fillna(0)).cumprod()
    idx=idx[ret[cols].count(axis=1)>=10]
    atr,R=rules(idx,idx,idx)  # 종가만 → ATR은 종가 변화로 근사
    for k,s in R.items():
        for dt,r in sim(idx,idx,idx,atr,s): rows.append(("업종",sec,k,dt,r))
r=pd.DataFrame(rows,columns=["grp","u","rule","dt","R"]); r["half"]=np.where(r.dt.dt.year<=2017,"2010~17","2018~26")
out=["# 수익용 숏 규칙 시험 (2026-11-08)\n","| 대상 | 규칙 | 매매 | 승률 | 평균R | 95% 구간 | 2010~17 | 2018~26 | 판정 |","|---|---|---|---|---|---|---|---|---|"]
for (g,k),x in r.groupby(["grp","rule"]):
    m=x.R.mean(); se=x.R.std()/np.sqrt(len(x)); hv=x.groupby("half").R.mean()
    ok=m>0 and m-1.96*se>0 and all(hv.get(h,-1)>0 for h in("2010~17","2018~26"))
    out.append(f"| {g} | {k} | {len(x)} | {(x.R>0).mean()*100:.0f}% | {m:+.3f} | {m-1.96*se:+.3f}~{m+1.96*se:+.3f} | {hv.get('2010~17',np.nan):+.3f} | {hv.get('2018~26',np.nan):+.3f} | {'합격' if ok else '불합격'} |")
print("\n".join(out)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","short_test_2026-11-08.md"),"w").write("\n".join(out)+"\n")
