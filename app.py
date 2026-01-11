
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Retail Pricing Analytics Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "pricing_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week"])
    # Derived fields
    df["avg_comp_price"] = df[["walmart_price","nofrills_price","dollarama_price"]].mean(axis=1)
    df["price_index_vs_comp_avg"] = (df["gt_price"] / df["avg_comp_price"]) * 100
    df["unit_margin"] = df["gt_price"] - df["cost"]
    df["gross_margin_pct"] = np.where(df["gt_price"]>0, (df["unit_margin"] / df["gt_price"]) * 100, np.nan)
    df["gross_profit"] = df["unit_margin"] * df["units"]
    df["sales"] = df["gt_price"] * df["units"]
    return df

def weighted_price_index(group: pd.DataFrame, competitor_col: str):
    # Weighted by GT unit sales to reflect impact (basket proxy)
    w = group["units"].clip(lower=1)
    return (np.average(group["gt_price"], weights=w) / np.average(group[competitor_col], weights=w)) * 100

def make_recommendations(df_slice: pd.DataFrame):
    """
    Heuristic recommendations:
    - Per SKU: if GT is > 103 vs competitor avg AND units are below median for that subdept, suggest consider price reduction.
    - If GT is < 97 AND margin% is high, suggest opportunity to raise price slightly.
    - If promo_flag=1 and margin% drops too low, flag promo depth risk.
    """
    recs = []
    # Latest week snapshot
    latest_week = df_slice["week"].max()
    snap = df_slice[df_slice["week"] == latest_week].copy()
    if snap.empty:
        return pd.DataFrame()

    subdept_median_units = snap.groupby("sub_department")["units"].transform("median")
    snap["units_vs_median"] = snap["units"] / subdept_median_units.replace(0, np.nan)

    for _, r in snap.iterrows():
        idx = r["price_index_vs_comp_avg"]
        gm = r["gross_margin_pct"]
        promo = r["promo_flag"]
        units_ratio = r["units_vs_median"]

        action = None
        reason = None

        if idx > 103 and units_ratio < 0.85:
            action = "Consider price decrease / sharpen"
            reason = f"Price index {idx:.1f} (expensive) + low units vs subdept median ({units_ratio:.2f}x)."
        elif idx < 97 and gm > 32:
            action = "Opportunity to raise price slightly"
            reason = f"Price index {idx:.1f} (cheap) with strong margin {gm:.1f}%."
        elif promo == 1 and gm < 12:
            action = "Promo depth risk"
            reason = f"On promo with low margin {gm:.1f}% — validate funding / depth."
        else:
            continue

        recs.append({
            "week": latest_week.date(),
            "department": r["department"],
            "sub_department": r["sub_department"],
            "sku": r["sku"],
            "product": r["product"],
            "gt_price": r["gt_price"],
            "avg_comp_price": round(r["avg_comp_price"],2),
            "price_index": round(idx,1),
            "gross_margin_pct": round(gm,1),
            "units": int(r["units"]),
            "recommendation": action,
            "why": reason
        })

    out = pd.DataFrame(recs).sort_values(["recommendation","price_index"], ascending=[True, False])
    return out

df = load_data()

st.title("Retail Pricing Analytics Dashboard (Simulated)")
st.caption("Built to demonstrate pricing index tracking, competitor benchmarking, margin impact, and promo risk — using realistic simulated retail data.")

with st.sidebar:
    st.header("Filters")
    dept = st.multiselect("Department", sorted(df["department"].unique()), default=sorted(df["department"].unique()))
    subdept = st.multiselect("Sub-department", sorted(df["sub_department"].unique()))
    promo_only = st.checkbox("Promo only", value=False)
    competitor = st.selectbox("Primary competitor (for index)", ["Walmart", "No Frills", "Dollarama", "Competitor Average"])
    date_range = st.slider(
        "Weeks range",
        min_value=df["week"].min().to_pydatetime(),
        max_value=df["week"].max().to_pydatetime(),
        value=(df["week"].min().to_pydatetime(), df["week"].max().to_pydatetime())
    )
    st.divider()
    st.subheader("Quick explainer")
    st.write("- **Price index**: (GT basket price / competitor basket price) × 100")
    st.write("- **>100** means GT is more expensive; **<100** means GT is cheaper.")
    st.write("- Weighted index uses GT units as a basket proxy.")

mask = df["department"].isin(dept)
if subdept:
    mask &= df["sub_department"].isin(subdept)
if promo_only:
    mask &= (df["promo_flag"] == 1)
mask &= (df["week"] >= pd.Timestamp(date_range[0])) & (df["week"] <= pd.Timestamp(date_range[1]))
dff = df.loc[mask].copy()

if dff.empty:
    st.warning("No data for current filters.")
    st.stop()

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
total_sales = dff["sales"].sum()
total_gp = dff["gross_profit"].sum()
gm_pct = (total_gp / total_sales) * 100 if total_sales else np.nan
promo_share = dff["promo_flag"].mean() * 100
units = dff["units"].sum()

# Price index KPI (weighted)
if competitor == "Competitor Average":
    comp_cols = ["walmart_price","nofrills_price","dollarama_price"]
    w = dff["units"].clip(lower=1)
    idx_val = (np.average(dff["gt_price"], weights=w) / np.average(dff[comp_cols].mean(axis=1), weights=w)) * 100
else:
    comp_col = {"Walmart":"walmart_price","No Frills":"nofrills_price","Dollarama":"dollarama_price"}[competitor]
    idx_val = weighted_price_index(dff, comp_col)

col1.metric("Sales", f"${total_sales:,.0f}")
col2.metric("Gross Profit", f"${total_gp:,.0f}")
col3.metric("GM%", f"{gm_pct:,.1f}%")
col4.metric("Units", f"{units:,.0f}")
col5.metric("Price Index", f"{idx_val:,.1f}")

st.divider()

# Time series: Price index + GM% trend
st.subheader("Trends")
agg = dff.groupby("week").agg(
    gt_price_w=("gt_price", lambda s: np.average(s, weights=dff.loc[s.index, "units"].clip(lower=1))),
    comp_avg_w=("avg_comp_price", lambda s: np.average(s, weights=dff.loc[s.index, "units"].clip(lower=1))),
    sales=("sales","sum"),
    gross_profit=("gross_profit","sum")
).reset_index()
agg["price_index"] = (agg["gt_price_w"]/agg["comp_avg_w"]) * 100
agg["gm_pct"] = np.where(agg["sales"]>0, (agg["gross_profit"]/agg["sales"])*100, np.nan)

c1, c2 = st.columns(2)
fig_idx = px.line(agg, x="week", y="price_index", title="Weighted Price Index vs Competitor Average")
c1.plotly_chart(fig_idx, use_container_width=True)
fig_gm = px.line(agg, x="week", y="gm_pct", title="Gross Margin % Trend")
c2.plotly_chart(fig_gm, use_container_width=True)

# Category / subdept view
st.subheader("Price Index by Sub-department (Latest Week)")
latest = dff[dff["week"] == dff["week"].max()].copy()
sub = latest.groupby(["department","sub_department"]).apply(
    lambda g: pd.Series({
        "price_index": (np.average(g["gt_price"], weights=g["units"].clip(lower=1)) / np.average(g["avg_comp_price"], weights=g["units"].clip(lower=1))) * 100,
        "gm_pct": (g["gross_profit"].sum()/g["sales"].sum())*100 if g["sales"].sum()>0 else np.nan,
        "sales": g["sales"].sum()
    })
).reset_index()
sub = sub.sort_values("price_index", ascending=False)

c3, c4 = st.columns(2)
fig_bar = px.bar(sub, x="sub_department", y="price_index", color="department", title="Price Index (Latest Week)")
c3.plotly_chart(fig_bar, use_container_width=True)
fig_sc = px.scatter(sub, x="price_index", y="gm_pct", size="sales", color="department",
                    title="Index vs GM% (bubble size = sales)")
c4.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# Recommendations
st.subheader("Actionable Recommendations (Latest Week)")
recs = make_recommendations(dff)
if recs.empty:
    st.info("No recommendations triggered under current filters (heuristics are conservative).")
else:
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.caption("Rules are heuristic (demo). In production, use elasticity models + test/learn & funding constraints.")

st.divider()

# SKU drilldown
st.subheader("SKU Drilldown")
sku = st.selectbox("Choose a SKU", sorted(dff["sku"].unique()))
sku_df = dff[dff["sku"] == sku].sort_values("week")

k1,k2,k3,k4 = st.columns(4)
k1.metric("Product", sku_df["product"].iloc[0])
k2.metric("Sub-dept", sku_df["sub_department"].iloc[0])
k3.metric("Avg Price Index", f"{sku_df['price_index_vs_comp_avg'].mean():.1f}")
k4.metric("Avg GM%", f"{sku_df['gross_margin_pct'].mean():.1f}%")

sku_melt = sku_df.melt(
    id_vars=["week","promo_flag"],
    value_vars=["gt_price","gt_regular_price","walmart_price","nofrills_price","dollarama_price"],
    var_name="series", value_name="price"
)
fig_prices = px.line(sku_melt, x="week", y="price", color="series", title="Price vs Competitors")
st.plotly_chart(fig_prices, use_container_width=True)

fig_units = px.bar(sku_df, x="week", y="units", title="Units (proxy demand)")
st.plotly_chart(fig_units, use_container_width=True)

st.caption("Tip for interviews: show 1 SKU where GT is >103 index and discuss a sharpen recommendation + margin impact.")
