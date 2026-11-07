"""Streamlit dashboard for Smart Stock Research Lab. Run: streamlit run app.py."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from stockresearch.research_registry import METHODS, list_methods

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment_results.csv"
PRICE_PANEL = ROOT / "data" / "panel_adjclose.csv"
TESTS = [method.test_id for method in list_methods()]

st.set_page_config(page_title="Smart Stock | Research Lab", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 15% 0%,#172b46 0%,#0b1220 44%,#080d18 100%)}
[data-testid="stSidebar"]{background:#0b1424;border-right:1px solid #26364c}
[data-testid="stMetric"]{background:linear-gradient(145deg,#14243a,#101b2d);padding:16px 18px;border:1px solid #263b56;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9db1cb}[data-testid="stMetricValue"]{color:#f3f7ff}
.hero{padding:24px 28px;border:1px solid #29415f;border-radius:18px;background:linear-gradient(110deg,rgba(24,54,83,.92),rgba(15,26,44,.82));margin-bottom:18px}
.eyebrow{color:#77c7d4;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.hero h1{color:#f4f7fb;margin:6px 0;font-size:32px}.hero p{color:#aebed2;margin:0}
.notice{border-left:3px solid #e5b45c;background:#2a2419;padding:12px 15px;border-radius:7px;color:#e9d8b7}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">SMART STOCK · RESEARCH TERMINAL</div>
<h1>Research Intelligence Dashboard</h1><p>Explore statistical evidence, uncertainty and data quality — not trading signals.</p></div>""", unsafe_allow_html=True)

def load_results(path):
    if not path.exists(): return pd.DataFrame()
    f = pd.read_csv(path)
    if "metrics_json" not in f: f["metrics_json"] = "{}"
    def parse(v):
        try:
            x = json.loads(v) if isinstance(v, str) else {}
            return x if isinstance(x, dict) else {}
        except (ValueError, TypeError): return {}
    f["metrics"] = f["metrics_json"].map(parse)
    return f

def numeric_metrics(f):
    rows=[]
    for _,r in f.iterrows():
        base={k:r.get(k) for k in ["test_id","ticker","method","status","observations"]}
        for key,val in r["metrics"].items():
            if isinstance(val,(int,float)) and not isinstance(val,bool):
                rows.append({**base,"metric":key,"value":float(val)})
    return pd.DataFrame(rows)

with st.sidebar:
    st.markdown("## Control room")
    st.caption("Local files · no data leaves your machine")
    result_path=st.text_input("Results CSV",value=str(RESULTS))
    st.divider()
    st.markdown("### Research mode")
    research_mode = st.radio("Experience", ["Guided", "Advanced"], horizontal=True)
    st.caption("Both modes use the same research methods and validation standards.")
    st.markdown("### Run research")
    stage=st.selectbox("Universe stage",["initial","comparative","exploratory","final"])
    research_question = st.text_input("Research question", placeholder="What market behaviour are you investigating?")
    hypothesis = st.text_input("Predeclared hypothesis", placeholder="What result would support or contradict it?")
    chosen=st.multiselect("Research questions / methods",TESTS,default=TESTS,format_func=lambda test_id: f"{test_id} · {METHODS[test_id].question}")
    if research_mode == "Guided":
        st.caption("Choose the question closest to what you want to investigate. Review method assumptions and limitations before interpreting results.")
    else:
        st.caption("Advanced mode exposes the same registered methods and raw experiment outputs; parameter-level controls remain method-specific.")
    run=st.button("▶ Run selected research",use_container_width=True,type="primary")
    if chosen:
        with st.expander("Selected method guidance", expanded=(research_mode == "Guided")):
            for test_id in chosen:
                method = METHODS[test_id]
                st.markdown(f"**{test_id} — {method.domain}**")
                st.write(method.audience_note)
                st.caption("Limitations: " + " ".join(method.limitations))
    st.caption("Runs the registered methods through the existing CLI against data/panel_adjclose.csv.")

if run:
    if not chosen: st.error("Select at least one test.")
    elif not PRICE_PANEL.exists(): st.error("No adjusted-price panel. Run python run.py download first.")
    else:
        cmd=[sys.executable,str(ROOT/"run.py"),"test","--stage",stage,"--tests",*chosen]
        if research_question.strip(): cmd.extend(["--question", research_question.strip()])
        if hypothesis.strip(): cmd.extend(["--hypothesis", hypothesis.strip()])
        with st.spinner("Running statistical experiments…"):
            proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=1800)
        if proc.returncode==0:
            st.success("Research run completed. Ledger refreshed.")
            if proc.stdout: st.code(proc.stdout[-5000:])
            st.rerun()
        else:
            st.error("Research run failed.")
            st.code((proc.stderr or proc.stdout)[-8000:])

frame=load_results(Path(result_path).expanduser())
if frame.empty:
    st.info("No experiment ledger found. Download data and run research, or start a run from the sidebar.")
    st.code("python run.py download\npython run.py test --stage initial\nstreamlit run app.py",language="bash")
    st.stop()
required={"test_id","ticker","status","method","observations"}
if required-set(frame.columns):
    st.error(f"Ledger schema missing: {sorted(required-set(frame.columns))}")
    st.stop()
frame["observations"]=pd.to_numeric(frame["observations"],errors="coerce")
with st.sidebar:
    st.divider()
    st.markdown("### Explore results")
    test_opts=sorted(frame.test_id.dropna().unique())
    ticker_opts=sorted(frame.ticker.dropna().unique())
    status_opts=sorted(frame.status.dropna().unique())
    tf=st.multiselect("Test IDs",test_opts,default=test_opts)
    sf=st.multiselect("Instruments",ticker_opts,default=ticker_opts)
    stf=st.multiselect("Statuses",status_opts,default=status_opts)
view=frame[frame.test_id.isin(tf)&frame.ticker.isin(sf)&frame.status.isin(stf)].copy()
ok=int((view.status=="OK").sum()); err=int((view.status=="ERROR").sum()); skip=int((view.status=="SKIPPED").sum())
a,b,c,d=st.columns(4)
a.metric("Result rows",f"{len(view):,}"); b.metric("Completed diagnostics",f"{ok:,}")
c.metric("Errors / skipped",f"{err:,} / {skip:,}"); d.metric("OK share",f"{ok/len(view)*100:.1f}%" if len(view) else "—")
overview,explorer,uncertainty,ledger=st.tabs(["◈ Overview","⌁ Metric explorer","◎ Confidence & uncertainty","▤ Result ledger"])
with overview:
    l,r=st.columns([1.1,1])
    counts=view.groupby(["test_id","status"]).size().reset_index(name="rows")
    with l:
        st.subheader("Execution status")
        if not counts.empty:
            fig=px.bar(counts,x="test_id",y="rows",color="status",barmode="stack",
                color_discrete_map={"OK":"#42c6a5","ERROR":"#ed6a78","SKIPPED":"#e6b65c"})
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=350)
            st.plotly_chart(fig,use_container_width=True)
    with r:
        st.subheader("Coverage by instrument")
        pc=view.groupby(["ticker","status"]).size().reset_index(name="rows")
        if not pc.empty:
            fig=px.bar(pc,x="ticker",y="rows",color="status",barmode="stack",
                color_discrete_map={"OK":"#42c6a5","ERROR":"#ed6a78","SKIPPED":"#e6b65c"})
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=350)
            st.plotly_chart(fig,use_container_width=True)
    st.subheader("Research notes and exceptions")
    notes=view[view.note.fillna("").astype(str).str.len()>0][["test_id","ticker","status","method","note"]] if "note" in view else pd.DataFrame()
    st.dataframe(notes,use_container_width=True,hide_index=True)
with explorer:
    st.subheader("Inspect calculated values")
    nums=numeric_metrics(view)
    if nums.empty: st.warning("No scalar numeric metrics found.")
    else:
        metric=st.selectbox("Metric",sorted(nums.metric.unique()))
        mv=nums[(nums.metric==metric)&(nums.status=="OK")]
        if not mv.empty:
            fig=px.box(mv,x="test_id",y="value",color="test_id",points="all",hover_data=["ticker","method","observations"])
            fig.update_layout(template="plotly_dark",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=390)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(mv.sort_values("value"),use_container_width=True,hide_index=True)
        else: st.info("No successful numeric observations for this metric.")
with uncertainty:
    st.subheader("How much confidence should I place in this result?")
    st.markdown('<div class="notice">A metric is not a confidence interval. This dashboard only displays uncertainty the experiment actually calculates; it does not invent confidence bounds from point estimates.</div>',unsafe_allow_html=True)
    nums=numeric_metrics(view)
    pvals=nums[nums.metric.str.contains("pvalue",case=False,na=False)] if not nums.empty else pd.DataFrame()
    if pvals.empty:
        st.info("No scalar p-values in this selection. Confidence intervals are not systematically recorded in the current ledger.")
    else:
        st.caption("Unadjusted p-values are exploratory. A 0.05 reference is not a universal decision boundary; account for multiple testing.")
        pv=pvals[["test_id","ticker","method","metric","value","observations"]].copy()
        pv["below_0_05_unadjusted"]=pv.value<0.05
        fig=px.strip(pv,x="test_id",y="value",color="below_0_05_unadjusted",hover_data=["ticker","method","metric","observations"],color_discrete_map={True:"#ed6a78",False:"#66b8d4"})
        fig.add_hline(y=.05,line_dash="dash",line_color="#e6b65c",annotation_text="0.05 reference only")
        fig.update_layout(template="plotly_dark",yaxis_type="log",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=390)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pv,use_container_width=True,hide_index=True)
    st.markdown("**Evidence checklist**")
    st.write("- Sample sizes and skipped/error rows remain visible.")
    st.write("- The ledger lacks systematic confidence intervals and adjusted-p-value fields.")
    st.write("- PBO is a split-based selection-instability diagnostic, not a conventional confidence interval.")
    st.write("- Serial dependence, overlapping observations, model selection and survivorship bias can invalidate naive uncertainty estimates.")
with ledger:
    st.subheader("Full calculation ledger")
    cols=[x for x in ["test_id","ticker","status","method","observations","note","metrics_json"] if x in view]
    st.dataframe(view[cols],use_container_width=True,hide_index=True,height=520)
    st.download_button("⬇ Export filtered CSV",view[cols].to_csv(index=False),"smart_stock_filtered_results.csv","text/csv")
