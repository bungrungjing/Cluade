"""v3 SELL 신호 때만 숏 — 사전 등록 (2026-11-08, 사용자 요청, 숏 시험 2차·다중비교 주의)

v3 SELL(signals.py): 스토캐스틱 K가 80 이상에서 하향 교차, 또는 MACD 데드크로스 & 종가 < 20일선
대상: QQQ, SPY(지수), 관문 통과 개별 종목 전체(참고: 국내 계좌로는 개별주 숏 불가 → 규칙 자체의 힘 확인용)
규칙 V1: v3 SELL 다음 날 숏 / V2: v3 SELL & 종가 < 50일선(하락 추세) / V3: v3 SELL & 종가 > 50일선(상승 중 고점 신호)
청산 A: 손절 +2ATR, 목표 −3×2ATR, 최대 10거래일 / 청산 B: 손절 +2ATR, 최대 20거래일, 목표 없음
비교: 같은 대상 '아무 날 숏'(앞 시험 S3: 지수 −0.27R)
합격: 평균R > 0, 95% 하한 > 0, 두 구간 모두 양수
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "signals"))
import numpy as np, pandas as pd
from edge_hunt import rd, REPO
from signals import build
FEE=0.002
def sim(d,sig,tgtR,maxh):
    C,H,L,A,S=d.close.values,d.high.values,d.low.values,d.atr.values,sig.values
    out=[];i=0;n=len(C)
    while i<n-1:
        if not S[i] or not np.isfinite(A[i]) or A[i]<=0: i+=1;continue
        e=C[i];risk=2*A[i];stop=e+risk;tgt=e-tgtR*risk if tgtR else -1e18;px=None
        for j in range(i+1,min(n,i+1+maxh)):
            if H[j]>=stop: px=stop;break
            if L[j]<=tgt: px=tgt;break
        if px is None: px=C[j]
        out.append((d.index[i],(e-px)/risk-FEE*e/risk)); i=j+1
    return out
def run(df,grp,u,rows):
    try: d=build(df.copy(),"v3")
    except Exception: return
    d=d.iloc[60:]
    s=d.signal=="SELL"; above=d.close>d.sma50.where(d.sma50.notna())
    sma50=d.close.rolling(50).mean()
    R={"V1 SELL":s,"V2 SELL+50일선 아래":s&(d.close<sma50),"V3 SELL+50일선 위":s&(d.close>sma50),"기준 아무 날":sma50.notna()}
    if grp=="종목":
        gate=(d.close>=5)&((d.close*d.volume).rolling(20).mean()>=4e7)
        R={k:v&gate for k,v in R.items()}
    for k,sig in R.items():
        for ex,(t,m) in {"A 3R·10일":(3,10),"B 20일·목표없음":(0,20)}.items():
            for dt,r in sim(d,sig,t,m): rows.append((grp,u,k,ex,dt,r))
rows=[]
for nm in ("QQQ","SPY"): run(rd(f"daily/{nm}.csv.gz"),"지수",nm,rows)
names=[n for n in subprocess.run(["git","-C",REPO,"ls-tree","-r","--name-only","origin/data"],capture_output=True,text=True).stdout.split() if n.startswith("daily/")]
for p in names:
    t=os.path.basename(p).split(".")[0]
    if t in ("SPY","QQQ","_VIX"): continue
    try: df=rd(p)
    except Exception: continue
    if len(df)>=300: run(df,"종목",t,rows)
r=pd.DataFrame(rows,columns=["grp","u","rule","exit","dt","R"]); r["half"]=np.where(pd.to_datetime(r.dt).dt.year<=2017,"2010~17","2018~26")
out=["# v3 SELL 숏 시험 (2026-11-08)\n","| 대상 | 규칙 | 청산 | 매매 | 승률 | 평균R | 95% 구간 | 2010~17 | 2018~26 | 판정 |","|---|---|---|---|---|---|---|---|---|---|"]
for (g,k,e),x in r.groupby(["grp","rule","exit"]):
    m=x.R.mean(); se=x.R.std()/np.sqrt(len(x)); hv=x.groupby("half").R.mean()
    ok=(not k.startswith("기준")) and m>0 and m-1.96*se>0 and all(hv.get(h,-1)>0 for h in("2010~17","2018~26"))
    out.append(f"| {g} | {k} | {e} | {len(x)} | {(x.R>0).mean()*100:.0f}% | {m:+.3f} | {m-1.96*se:+.3f}~{m+1.96*se:+.3f} | {hv.get('2010~17',np.nan):+.3f} | {hv.get('2018~26',np.nan):+.3f} | {'기준' if k.startswith('기준') else ('합격' if ok else '불합격')} |")
print("\n".join(out)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","short_v3_2026-11-08.md"),"w").write("\n".join(out)+"\n")
