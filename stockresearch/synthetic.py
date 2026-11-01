"""Deterministic synthetic observations for offline pipeline tests; never real prices."""
from pathlib import Path
import numpy as np,pandas as pd
from . import config,universe
from .data import build_panels,safe_name
def make(data_dir,end="2025-12-31",seed=config.SEED):
    rng=np.random.default_rng(seed); root=Path(data_dir); raw=root/"raw"; raw.mkdir(parents=True,exist_ok=True)
    dates=pd.bdate_range(config.START,end); market=rng.normal(.0003,.012,len(dates))
    for ticker in universe.all_tickers():
        ret=market*rng.uniform(.6,1.6)+rng.normal(0,.009,len(dates)); price=50*np.exp(np.cumsum(ret))
        vol=rng.lognormal(15,.35,len(dates)); df=pd.DataFrame({"Open":price,"High":price*1.01,"Low":price*.99,"Close":price,"Adj Close":price,"Volume":vol,"Dividends":0.,"Stock Splits":0.},index=dates)
        df.to_csv(raw/f"{safe_name(ticker)}.csv")
    for ticker,name in config.MACRO.items():
        pd.DataFrame({"Close":np.cumsum(rng.normal(0,.01,len(dates)))+3},index=dates).to_csv(raw/f"{safe_name(ticker)}.csv")
    build_panels(root)
