"""Data quality diagnostics that run before statistical experiments."""
from pathlib import Path
import numpy as np,pandas as pd
def audit_panels(data_dir):
    root=Path(data_dir); p=root/"panel_adjclose.csv"
    if not p.exists(): raise FileNotFoundError(f"{p} missing; run download first")
    px=pd.read_csv(p,index_col=0,parse_dates=True); out=[]
    for ticker in px:
        s=pd.to_numeric(px[ticker],errors="coerce"); r=np.log(s.where(s>0)).diff(); valid=s.notna()
        out.append({"ticker":ticker,"observations":int(valid.sum()),"missing":int(s.isna().sum()),"duplicate_dates":int(px.index.duplicated().sum()),"nonpositive":int((s<=0).sum()),"return_outliers_abs_gt_40pct":int((r.abs()>.4).sum()),"max_abs_daily_return":float(r.abs().max()) if r.notna().any() else np.nan,"first_date":str(s.first_valid_index().date()) if valid.any() else None,"last_date":str(s.last_valid_index().date()) if valid.any() else None})
    report=pd.DataFrame(out); report.to_csv(root/"quality_audit.csv",index=False); return report
