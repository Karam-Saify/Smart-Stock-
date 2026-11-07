"""Defensive Yahoo/FRED ingestion. Raw vendor responses are retained for auditability."""
from __future__ import annotations
import json, logging, os, random, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from . import config, universe
log=logging.getLogger("research.data")
REQUIRED=("Open","High","Low","Close","Adj Close","Volume")
def safe_name(t): return t.replace("^","_").replace("=","_").replace("/","_")
def _normalise_index(df):
    out=df.copy(); idx=pd.to_datetime(out.index,errors="coerce")
    if getattr(idx,"tz",None) is not None: idx=idx.tz_convert("UTC").tz_localize(None)
    out.index=idx.normalize(); out=out[~out.index.isna()].sort_index(); return out[~out.index.duplicated(keep="last")]
def fetch_yahoo(ticker,start=None,end=None,retries=5,session=None):
    import yfinance as yf
    last=None
    for attempt in range(retries):
        try:
            kwargs=dict(start=start or config.START,end=end or config.END,auto_adjust=False,actions=True,raise_errors=True)
            if session is not None: kwargs["session"]=session
            frame=yf.Ticker(ticker).history(**kwargs)
            if frame is None or frame.empty: raise ValueError("Yahoo returned an empty history")
            frame=_normalise_index(frame)
            missing=[c for c in REQUIRED if c not in frame.columns]
            if missing: raise ValueError(f"Missing required Yahoo columns: {missing}")
            for col in REQUIRED: frame[col]=pd.to_numeric(frame[col],errors="coerce")
            frame=frame.replace([np.inf,-np.inf],np.nan)
            if (frame["Adj Close"].dropna()<=0).any(): raise ValueError("Non-positive adjusted prices")
            if frame["Adj Close"].notna().sum()<2: raise ValueError("Insufficient valid adjusted prices")
            return frame
        except Exception as exc:
            last=exc
            if attempt+1<retries: time.sleep(min(30,1.5*(2**attempt))+random.random())
    raise RuntimeError(f"{ticker}: failed after {retries} attempts: {last}") from last
def _fred_cpi():
    import urllib.request
    with urllib.request.urlopen(config.FRED_CPI_URL,timeout=30) as response:
        from io import BytesIO
        d=pd.read_csv(BytesIO(response.read()),index_col=0,parse_dates=True)
    if d.empty: raise ValueError("FRED CPI response was empty")
    d.columns=["CPI"]; d["CPI"]=pd.to_numeric(d["CPI"],errors="coerce"); return d.dropna()
def build_panels(data_dir):
    root=Path(data_dir); raw=root/"raw"; root.mkdir(parents=True,exist_ok=True)
    adj={}; close={}; volume={}; metadata={}
    for ticker in universe.all_tickers():
        path=raw/f"{safe_name(ticker)}.csv"
        if not path.exists(): metadata[ticker]={"status":"missing","reason":"raw file absent"}; continue
        try:
            df=_normalise_index(pd.read_csv(path,index_col=0,parse_dates=True))
            missing=[c for c in REQUIRED if c not in df.columns]
            if missing: raise ValueError(f"missing columns: {missing}")
            for c in REQUIRED: df[c]=pd.to_numeric(df[c],errors="coerce")
            valid=df["Adj Close"].where(df["Adj Close"]>0)
            adj[ticker]=valid; close[ticker]=df["Close"]; volume[ticker]=df["Volume"]
            ret=np.log(valid).diff(); gaps=df.index.to_series().diff().dt.days
            metadata[ticker]={"status":"ok","rows":int(len(df)),"first":str(df.index.min().date()),"last":str(df.index.max().date()),"missing_adjusted_prices":int(valid.isna().sum()),"missing_volume":int(df.Volume.isna().sum()),"zero_volume_days":int((df.Volume==0).sum()),"gaps_over_5_days":int((gaps>5).sum()),"absolute_log_returns_over_40pct":int((ret.abs()>.4).sum()),"split_events":int((pd.to_numeric(df.get("Stock Splits",0),errors="coerce")>0).sum()),"dividend_events":int((pd.to_numeric(df.get("Dividends",0),errors="coerce")>0).sum())}
        except Exception as exc: metadata[ticker]={"status":"invalid","reason":str(exc)}
    for name,frame in [("panel_adjclose",adj),("panel_close",close),("panel_volume",volume)]:
        if frame: pd.concat(frame,axis=1).sort_index().to_csv(root/f"{name}.csv")
    macro={}
    for ticker,name in config.MACRO.items():
        p=raw/f"{safe_name(ticker)}.csv"
        if p.exists():
            d=pd.read_csv(p,index_col=0,parse_dates=True); col="Close"
            if col in d: macro[name]=pd.to_numeric(d[col],errors="coerce")
    cp=raw/"_CPI.csv"
    if cp.exists(): macro["CPI"]=pd.read_csv(cp,index_col=0,parse_dates=True).iloc[:,0]
    if macro: pd.concat(macro,axis=1).sort_index().to_csv(root/"macro.csv")
    payload={"built_at":datetime.now(timezone.utc).isoformat(),"requested_start":config.START,"instruments":metadata,"source_semantics":{"Close":"split-adjusted close","Adj Close":"split- and dividend-adjusted close","Volume":"vendor-reported volume"},"survivorship_bias":"present: current selected universe only"}
    (root/"metadata.json").write_text(json.dumps(payload,indent=2))
    return payload
def download_all(data_dir,start=None,end=None,tickers=None,include_macro=True):
    root=Path(data_dir); raw=root/"raw"; raw.mkdir(parents=True,exist_ok=True)
    requested=list(tickers or universe.all_tickers())+(list(config.MACRO) if include_macro else [])
    failures=[]; manifest={"started_at":datetime.now(timezone.utc).isoformat(),"start":start or config.START,"end":end or config.END,"provider":"Yahoo Finance via yfinance","retrieval_policy":{"auto_adjust":False,"actions":True,"price_panel":"Adj Close","point_in_time_validity":"not established; vendor history may be revised"},"downloaded":{},"failures":{}}
    for ticker in requested:
        path=raw/f"{safe_name(ticker)}.csv"
        try:
            df=fetch_yahoo(ticker,start,end)
            temporary=path.with_suffix(path.suffix+".tmp")
            df.to_csv(temporary)
            os.replace(temporary,path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["downloaded"][ticker]={"rows":len(df),"first":str(df.index.min().date()),"last":str(df.index.max().date()),"file":str(path.relative_to(root)),"sha256":digest,"retrieved_at":datetime.now(timezone.utc).isoformat(),"columns":list(df.columns),"timezone":"normalized to UTC-naive dates"}
            log.info("%s: %s rows",ticker,len(df))
        except Exception as exc:
            failures.append(ticker); manifest["failures"][ticker]=repr(exc); log.error("%s: %s",ticker,exc)
    if include_macro:
        try: _fred_cpi().to_csv(raw/"_CPI.csv")
        except Exception as exc: manifest["failures"]["FRED_CPI"]=repr(exc); log.warning("FRED CPI unavailable: %s",exc)
    manifest["finished_at"]=datetime.now(timezone.utc).isoformat(); manifest["failed_tickers"]=failures
    (root/"download_manifest.json").write_text(json.dumps(manifest,indent=2))
    build_panels(root)
    return failures
