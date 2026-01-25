import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Page Config | إعدادات الصفحة ---
st.set_page_config(page_title="Local Content Dashboard", layout="wide")
st.title("📊 Local Content Forecasting Dashboard | لوحة توقعات المحتوى المحلي")

# --- Load Historical Data | تحميل البيانات التاريخية ---
@st.cache_data
def load_historical():
    data_path = Path(__file__).parent.parent.parent / 'Data'
    excel_file = data_path / 'Local content historical records Over All.xlsx'
    df_excel = pd.read_excel(excel_file, sheet_name='Summary', header=None)
    
    hist_data = []
    for i, year in enumerate([2024, 2023, 2022, 2021]):
        idx = i + 2
        hist_data.append({
            'Year': year,
            'Type': 'Actual',
            'Compensation': float(df_excel.iloc[idx, 1]),
            'Goods_Services': float(df_excel.iloc[idx, 2]),
            'CapEx': float(df_excel.iloc[idx, 3]),
            'Training': float(df_excel.iloc[idx, 4]),
            'Depreciation': float(df_excel.iloc[idx, 5]),
            'Total_LC': float(df_excel.iloc[idx, 6]),
            'Total_Cost': float(df_excel.iloc[idx, 7])
        })
    return pd.DataFrame(hist_data).sort_values('Year')

df_hist = load_historical()
last_actual = df_hist.iloc[-1]

# --- Sidebar Assumptions | قائمة الافتراضات الجانبية ---
st.sidebar.header("📈 Growth Assumptions | افتراضات النمو")

g_comp = st.sidebar.slider("Compensation Growth % | نمو الرواتب", 0, 30, 8) / 100
g_goods = st.sidebar.slider("Goods & Services Growth % | نمو السلع والخدمات", 0, 30, 12) / 100
g_capex = st.sidebar.slider("CapEx Growth % | نمو الأصول", 0, 30, 10) / 100
g_train = st.sidebar.slider("Training Growth % | نمو التدريب", 0, 30, 10) / 100
g_cost = st.sidebar.slider("Total Cost Growth % | نمو إجمالي التكاليف", 0, 30, 8) / 100

# --- Forecasting Logic | منطق التنبؤ ---
forecast_data = []
for year in [2025, 2026, 2027, 2028]:
    n = year - 2024
    comp = last_actual['Compensation'] * ((1 + g_comp) ** n)
    goods = last_actual['Goods_Services'] * ((1 + g_goods) ** n)
    capex = last_actual['CapEx'] * ((1 + g_capex) ** n)
    train = last_actual['Training'] * ((1 + g_train) ** n)
    depr = last_actual['Depreciation'] # Assuming depreciation remains constant for simplicity
    
    total_lc = comp + goods + capex + train + depr
    total_cost = last_actual['Total_Cost'] * ((1 + g_cost) ** n)
    
    forecast_data.append({
        'Year': year, 'Type': 'Forecast',
        'Compensation': comp, 'Goods_Services': goods, 'CapEx': capex,
        'Training': train, 'Depreciation': depr,
        'Total_LC': total_lc, 'Total_Cost': total_cost,
        'LC_Score_%': (total_lc / total_cost * 100)
    })

df_forecast = pd.DataFrame(forecast_data)

# Combine for Visuals | دمج البيانات للرسم البياني
df_hist['LC_Score_%'] = (df_hist['Total_LC'] / df_hist['Total_Cost'] * 100)
df_full = pd.concat([df_hist, df_forecast], ignore_index=True)

# --- Top Metrics | المؤشرات العلوية ---
col1, col2, col3 = st.columns(3)
col1.metric("Current LC Score (2024)", f"{last_actual['Total_LC']/last_actual['Total_Cost']*100:.2f}%")
col2.metric("Target LC Score (2028)", f"{df_forecast.iloc[-1]['LC_Score_%']:.2f}%", 
            delta=f"{df_forecast.iloc[-1]['LC_Score_%'] - (last_actual['Total_LC']/last_actual['Total_Cost']*100):.2f}%")
col3.metric("Projected Total LC (2028)", f"{df_forecast.iloc[-1]['Total_LC']:,.0f} SAR")

# --- Charts | الرسوم البيانية ---
st.subheader("Local Content Score Trend | مسار نسبة المحتوى المحلي")
fig = px.line(df_full, x="Year", y="LC_Score_%", color="Type", markers=True, 
              title="Historical vs Forecasted LC Score")
st.plotly_chart(fig, use_container_width=True, width='stretch')

# --- Data Table | جدول البيانات ---
st.subheader("Detailed Data | البيانات التفصيلية")
st.dataframe(df_full.style.format(precision=2))