"""Persistent idempotent virtual broker. No real account or broker connection exists."""
import json, os, tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Order:
    symbol: str
    shares: float
    ref_price: float

@dataclass
class Fill:
    symbol: str
    shares: float
    price: float
    cost: float
    run_id: str

class PaperBroker:
    def __init__(self, path, costs, state):
        self.path, self.costs, self.state = Path(path), costs, state

    @classmethod
    def init(cls, path, cash, costs, overwrite=False):
        p=Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists; refusing to overwrite virtual account")
        b=cls(p,costs,dict(cash=float(cash),initial_cash=float(cash),positions={},executed_runs=[],pending=None,peak_equity=float(cash),halt_runs_left=0))
        b.save(); return b

    @classmethod
    def load(cls,path,costs):
        return cls(path,costs,json.loads(Path(path).read_text()))

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=self.path.parent,suffix=".tmp")
        with os.fdopen(fd,"w") as f: json.dump(self.state,f,indent=2)
        os.replace(tmp,self.path)

    def positions(self):
        return {k:v for k,v in self.state["positions"].items() if abs(v)>1e-12}
    def cash(self): return float(self.state["cash"])
    def already_ran(self,run_id): return run_id in self.state["executed_runs"]
    def equity(self,prices):
        return self.cash()+sum(q*prices[s] for s,q in self.positions().items() if q)

    def execute(self,orders,run_id):
        if self.already_ran(run_id): return []
        fills=[]; pos=self.state["positions"]
        for o in sorted(orders,key=lambda x:x.shares):
            if not o.shares or o.ref_price<=0: continue
            fee_rate=self.costs.fee_rate(o.symbol); slip=self.costs.slippage_rate()
            price=o.ref_price*(1-slip if o.shares<0 else 1+slip)
            if o.shares<0:
                qty=-min(-o.shares,pos.get(o.symbol,0.0))
                if qty==0: continue
            else:
                qty=min(o.shares,self.cash()/(price+o.ref_price*fee_rate))
                if qty<=1e-9: continue
            reference_notional=abs(qty)*o.ref_price
            fee=reference_notional*fee_rate
            total_cost=fee+reference_notional*slip
            self.state["cash"] += -qty*price-fee
            pos[o.symbol]=pos.get(o.symbol,0.0)+qty
            fills.append(Fill(o.symbol,qty,price,total_cost,run_id))
        self.state["executed_runs"].append(run_id); self.save()
        return fills
