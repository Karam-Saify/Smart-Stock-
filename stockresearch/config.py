"""Central, versionable research configuration."""
START = "2005-01-01"
END = None
SEED = 42
INITIAL_SIX = ["AAPL", "NVDA", "JPM", "XOM", "WMT", "TSLA"]
MACRO = {"^TNX": "TNX", "CL=F": "OIL", "DX-Y.NYB": "USD", "^VIX": "VIX", "^IRX": "IRX"}
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
