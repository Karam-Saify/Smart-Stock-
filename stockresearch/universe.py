"""Explicit research universe; tiers are sampling strata, not quality rankings."""
STOCKS = [
("AAPL",1,"Technology"),("MSFT",1,"Technology"),("NVDA",1,"Semiconductors"),("AMZN",1,"Consumer/Technology"),("GOOGL",1,"Technology"),("META",1,"Technology"),("TSLA",1,"Automotive"),("JPM",1,"Financials"),("BRK-B",1,"Financials"),("WMT",1,"Staples"),("XOM",1,"Energy"),("LLY",1,"Healthcare"),
("KO",2,"Staples"),("JNJ",2,"Healthcare"),("PFE",2,"Healthcare"),("CVX",2,"Energy"),("GS",2,"Financials"),("BAC",2,"Financials"),("MCD",2,"Consumer"),("PG",2,"Staples"),("AMD",2,"Semiconductors"),("CRM",2,"Software"),("NFLX",2,"Media"),("ADBE",2,"Software"),
("PLTR",3,"Software/AI"),("COIN",3,"FinTech"),("HOOD",3,"FinTech"),("SMCI",3,"Hardware"),("MSTR",3,"Bitcoin treasury"),("RIVN",3,"Automotive"),("SOFI",3,"FinTech"),("SNOW",3,"Cloud"),("ENPH",3,"Renewables"),("GME",3,"Retail"),("LCID",3,"Automotive"),("MRNA",3,"Biotechnology")]
BROAD = ["SPY","QQQ","IWM","DIA"]
SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU"]
def all_tickers(): return [x[0] for x in STOCKS] + BROAD + SECTOR_ETFS
def stage_stocks(stage):
    tiers={"initial":{1},"comparative":{1,2},"exploratory":{3},"final":{1,2,3}}
    if stage not in tiers: raise ValueError(f"Unknown stage: {stage}")
    return [t for t, tier, _ in STOCKS if tier in tiers[stage]]
