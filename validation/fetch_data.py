"""일봉 데이터 내려받기 (야후 파이낸스, yfinance) — 깃허브 Actions에서 실행

사용법:
  python fetch_data.py <저장 폴더> --mode full    # 2010-01-01부터 전체 (주 1회 권장)
  python fetch_data.py <저장 폴더> --mode update  # 최근 15거래일만 받아 이어 붙임 (평일)

- 대상: validation/universe.csv 의 종목 + 시장 지표(SPY, QQQ, ^VIX)
- 저장: <폴더>/<티커>.csv.gz  (timestamp,open,high,low,close,volume — 배당·분할 수정주가)
- update 모드에서 겹치는 날 종가가 0.5% 넘게 다르면(분할·배당 재조정) 그 종목만 전체를 다시 받는다.
- 받지 못한 종목은 <폴더>/_failed.txt 에 남긴다.
"""
import argparse, os, sys, time
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
MARKET = ["SPY", "QQQ", "^VIX"]
COLS = ["open", "high", "low", "close", "volume"]


def fname(folder, t):
    return os.path.join(folder, t.replace("^", "_") + ".csv.gz")


def yf_symbol(t):
    return t.replace(".", "-")


def download(tickers, **kw):
    """여러 종목을 한 번에 받는다. 반환: {티커: DataFrame}"""
    out = {}
    for attempt in range(3):
        try:
            raw = yf.download([yf_symbol(t) for t in tickers], auto_adjust=True, progress=False,
                              group_by="ticker", threads=True, **kw)
            break
        except Exception as e:  # 일시 차단 등
            print("download error", e, file=sys.stderr)
            time.sleep(20 * (attempt + 1))
    else:
        return out
    for t in tickers:
        s = yf_symbol(t)
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.rename(columns=str.lower)[COLS].dropna(subset=["close"])
        if len(df):
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "timestamp"
            out[t] = df
    return out


def save(folder, t, df):
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.round(4).to_csv(fname(folder, t), compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--mode", choices=["full", "update"], default="update")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞의 N종목만")
    a = ap.parse_args()
    os.makedirs(a.folder, exist_ok=True)
    tickers = pd.read_csv(os.path.join(HERE, "universe.csv")).ticker.tolist()
    if a.limit:
        tickers = tickers[:a.limit]
    tickers = MARKET + tickers

    need_full = [t for t in tickers if a.mode == "full" or not os.path.exists(fname(a.folder, t))]
    upd = [t for t in tickers if t not in need_full]
    failed = []

    for i in range(0, len(upd), a.chunk):
        part = upd[i:i + a.chunk]
        got = download(part, period="1mo")
        for t in part:
            if t not in got:
                failed.append(t); continue
            old = pd.read_csv(fname(a.folder, t), parse_dates=["timestamp"], index_col="timestamp")
            new = got[t]
            common = old.index.intersection(new.index)
            if len(common) and ((old.loc[common, "close"] / new.loc[common, "close"] - 1).abs() > 0.005).any():
                need_full.append(t)  # 분할·배당 재조정 → 전체 다시
                continue
            save(a.folder, t, pd.concat([old[COLS], new]))
        time.sleep(2)

    for i in range(0, len(need_full), a.chunk):
        part = need_full[i:i + a.chunk]
        got = download(part, start="2010-01-01")
        for t in part:
            if t in got:
                save(a.folder, t, got[t])
            elif t not in failed:
                failed.append(t)
        print(f"full {i + len(part)}/{len(need_full)}", flush=True)
        time.sleep(3)

    with open(os.path.join(a.folder, "_failed.txt"), "w") as f:
        f.write("\n".join(failed))
    ok = len(tickers) - len(failed)
    print(f"완료: {ok}/{len(tickers)}종목, 실패 {len(failed)}")
    if ok < 0.8 * len(tickers):
        sys.exit("80% 미만 수신 — 야후 차단 가능성")


if __name__ == "__main__":
    main()
