# Retail Pricing Analytics Dashboard (Web)

A Streamlit web dashboard that demonstrates pricing analytics for a retailer:
- Weighted **price index** tracking vs competitors
- Margin + gross profit impact
- Promo risk flags
- Category/sub-department rollups
- SKU drilldowns (GT vs Walmart / No Frills / Dollarama)

This project uses **realistic simulated data** (not Giant Tiger confidential data).

## Quick start (local)

```bash
cd gt_pricing_dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## What to demo in interviews (2–4 minutes)
1. Filters (Department/Sub-dept, Promo only)
2. KPI row (Sales, GP, GM%, Units, Price Index)
3. Trend charts (Index + GM%)
4. Sub-dept index bar + Index vs GM% scatter
5. Recommendations table
6. SKU drilldown (price lines + units)

## Data model
Dataset columns include:
- week, department, sub_department, sku, product
- gt_price, gt_regular_price, cost, promo_flag, units
- walmart_price, nofrills_price, dollarama_price

## Notes
- Price index = (GT basket / competitor basket) × 100
- Weighted basket uses GT units as a proxy
# Retail-Analytics
