"""
LC Dashboard | لوحة تحكم المحتوى المحلي
Interactive Streamlit dashboard for Local Content Forecasting
لوحة تحكم تفاعلية لتوقعات المحتوى المحلي
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# --- Page Config | إعدادات الصفحة ---
st.set_page_config(
    page_title="Local Content Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    
    .metric-base { border-left-color: #2ca02c !important; }
    .metric-conservative { border-left-color: #ff7f0e !important; }
    .metric-optimistic { border-left-color: #d62728 !important; }
    
    h1, h2, h3 { 
        color: #1a3a3a;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Local Content Forecasting Dashboard")
st.divider()

# --- Load Data | تحميل البيانات ---
@st.cache_data
def load_forecast_data():
    """Load forecast data from CSV | تحميل بيانات التنبؤ من CSV"""
    data_path = Path(__file__).parent / 'LC_Full_Forecast_ALL_SCENARIOS.csv'
    
    if not data_path.exists():
        st.error(f"❌ Data file not found: {data_path}")
        st.info("💡 Run `lc_processor.py` first to generate forecast data.")
        st.stop()
    
    return pd.read_csv(data_path)


@st.cache_data
def load_historical_data():
    """Load historical data from Excel | تحميل البيانات التاريخية من Excel"""
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


def generate_custom_forecast(df_hist, growth_rates):
    """Generate forecast with custom growth rates | توليد التنبؤ برسوم نمو مخصصة"""
    last_actual = df_hist.iloc[-1]
    
    forecast_data = []
    for year in [2025, 2026, 2027, 2028]:
        n = year - 2024
        comp = last_actual['Compensation'] * ((1 + growth_rates['comp']) ** n)
        goods = last_actual['Goods_Services'] * ((1 + growth_rates['goods']) ** n)
        capex = last_actual['CapEx'] * ((1 + growth_rates['capex']) ** n)
        train = last_actual['Training'] * ((1 + growth_rates['train']) ** n)
        depr = last_actual['Depreciation']
        
        total_lc = comp + goods + capex + train + depr
        total_cost = last_actual['Total_Cost'] * ((1 + growth_rates['cost']) ** n)
        
        forecast_data.append({
            'Year': year,
            'Type': 'Forecast',
            'Compensation': comp,
            'Goods_Services': goods,
            'CapEx': capex,
            'Training': train,
            'Depreciation': depr,
            'Total_LC': total_lc,
            'Total_Cost': total_cost,
            'LC_Score_%': (total_lc / total_cost * 100)
        })
    
    df_hist_copy = df_hist.copy()
    df_hist_copy['LC_Score_%'] = (df_hist_copy['Total_LC'] / df_hist_copy['Total_Cost'] * 100)
    
    df_forecast = pd.DataFrame(forecast_data)
    df_combined = pd.concat([df_hist_copy, df_forecast], ignore_index=True)
    
    return df_combined


df_all = load_forecast_data()
df_hist = load_historical_data()

# Calculate default assumptions from historical data | حساب الافتراضات الافتراضية من البيانات التاريخية
def calculate_assumptions_from_history(df_hist):
    """Calculate growth rates from historical data"""
    df_sorted = df_hist.sort_values('Year')
    
    # Calculate YoY growth rates
    growth_rates = {}
    for col in ['Compensation', 'Goods_Services', 'CapEx', 'Training', 'Total_Cost']:
        values = df_sorted[col].values
        yoy_growth = []
        for i in range(1, len(values)):
            # Skip if previous value is zero or NaN
            if values[i-1] == 0 or pd.isna(values[i-1]):
                continue
            growth = ((values[i] - values[i-1]) / values[i-1]) * 100
            # Cap extreme outliers at ±100%
            growth = max(min(growth, 100), -100)
            yoy_growth.append(growth)
        
        # Calculate average, default to 8% if no valid data
        avg_growth = sum(yoy_growth) / len(yoy_growth) if yoy_growth else 8.0
        growth_rates[col] = avg_growth
    
    # Map to scenario keys
    base_rates = {
        'comp': round(growth_rates['Compensation'], 1),
        'goods': round(growth_rates['Goods_Services'], 1),
        'capex': round(growth_rates['CapEx'], 1),
        'train': round(growth_rates['Training'], 1),
        'cost': round(growth_rates['Total_Cost'], 1)
    }
    
    return base_rates

# Calculate base assumptions from history
base_assumptions = calculate_assumptions_from_history(df_hist)

# Define scenario assumptions | تعريف افتراضات السيناريوهات
SCENARIO_ASSUMPTIONS = {
    'Base': base_assumptions,
    'Conservative': {k: round(v * 0.7, 1) for k, v in base_assumptions.items()},
    'Optimistic': {k: round(v * 1.3, 1) for k, v in base_assumptions.items()}
}

# Debug: Display calculated assumptions at top
with st.expander("ℹ️ View Calculated Assumptions", expanded=False):
    st.markdown("**Base (from historical data):**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"💰 Compensation: {base_assumptions['comp']}%")
        st.write(f"📦 Goods & Services: {base_assumptions['goods']}%")
    with col2:
        st.write(f"🏗️ CapEx: {base_assumptions['capex']}%")
        st.write(f"🎓 Training: {base_assumptions['train']}%")
    with col3:
        st.write(f"📊 Total Cost: {base_assumptions['cost']}%")
    
    st.markdown("**Conservative (70% of Base):**")
    cons = SCENARIO_ASSUMPTIONS['Conservative']
    st.write(f"Comp: {cons['comp']}% | Goods: {cons['goods']}% | CapEx: {cons['capex']}% | Train: {cons['train']}%")
    
    st.markdown("**Optimistic (130% of Base):**")
    opt = SCENARIO_ASSUMPTIONS['Optimistic']
    st.write(f"Comp: {opt['comp']}% | Goods: {opt['goods']}% | CapEx: {opt['capex']}% | Train: {opt['train']}%")

# --- Sidebar Controls | عناصر التحكم الجانبية ---
st.sidebar.header("⚙️ Configuration")

# Mode selection
mode = st.sidebar.radio(
    "Select Mode",
    ["📊 View Scenarios", "🎯 Custom Scenario"],
    help="View pre-calculated scenarios or create custom forecast"
)

if mode == "📊 View Scenarios":
    st.sidebar.markdown("### 📈 Select Scenario")
    
    available_scenarios = sorted(df_all['Scenario'].unique())
    selected_scenario = st.sidebar.selectbox(
        "Scenario",
        available_scenarios,
        format_func=lambda x: f"{x.upper()}"
    )
    
    # Get assumptions for selected scenario
    scenario_key = selected_scenario.capitalize()
    assumptions = SCENARIO_ASSUMPTIONS.get(scenario_key, SCENARIO_ASSUMPTIONS['Base'])
    
    # Display scenario assumptions
    st.sidebar.markdown("### 📊 Assumptions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("💰 Compensation", f"{assumptions['comp']}%")
        st.metric("🏗️ Capital Expenses", f"{assumptions['capex']}%")
    with col2:
        st.metric("📦 Goods & Services", f"{assumptions['goods']}%")
        st.metric("🎓 Training", f"{assumptions['train']}%")
    
    # Filter data by selected scenario
    df_scenario = df_all[df_all['Scenario'] == selected_scenario].copy()
    scenario_name = scenario_key
    
else:
    st.sidebar.markdown("### 🎯 Custom Growth Rates")
    st.sidebar.markdown("Adjust the growth rates below")
    
    st.sidebar.write("**💰 Compensation Growth (%)**")
    g_comp = st.sidebar.slider("Compensation", 0, 30, 8, key="comp") / 100
    
    st.sidebar.write("**📦 Goods & Services Growth (%)**")
    g_goods = st.sidebar.slider("Goods & Services", 0, 30, 12, key="goods") / 100
    
    st.sidebar.write("**🏗️ Capital Expenses Growth (%)**")
    g_capex = st.sidebar.slider("Capital Expenses", 0, 30, 10, key="capex") / 100
    
    st.sidebar.write("**🎓 Training Growth (%)**")
    g_train = st.sidebar.slider("Training", 0, 30, 10, key="train") / 100
    
    st.sidebar.write("**📊 Total Cost Growth (%)**")
    g_cost = st.sidebar.slider("Total Cost", 0, 30, 8, key="cost") / 100
    
    # Generate custom forecast
    custom_growth = {
        'comp': g_comp,
        'goods': g_goods,
        'capex': g_capex,
        'train': g_train,
        'cost': g_cost
    }
    
    df_scenario = generate_custom_forecast(df_hist, custom_growth)
    scenario_name = "Custom"

df_scenario = df_scenario.sort_values('Year')

# Ensure LC_Score_% is calculated
if 'LC_Score_%' not in df_scenario.columns:
    df_scenario['LC_Score_%'] = (df_scenario['Total_LC'] / df_scenario['Total_Cost'] * 100)

# Get historical and forecast data
df_hist_display = df_scenario[df_scenario['Type'] == 'Actual']
df_forecast = df_scenario[df_scenario['Type'] == 'Forecast']

if len(df_hist_display) == 0 or len(df_forecast) == 0:
    st.error("❌ Invalid data structure.")
    st.stop()

# Year filter in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Year Selection")

available_years = sorted(df_scenario['Year'].unique())
selected_year = st.sidebar.select_slider(
    "Select Year",
    options=available_years,
    value=available_years[-1]
)

# Get data for selected year
df_year = df_scenario[df_scenario['Year'] == selected_year]

if len(df_year) == 0:
    st.error(f"❌ No data available for year {selected_year}")
    st.stop()

year_data = df_year.iloc[0]
last_actual = df_hist_display.iloc[-1]

# --- Top Metrics | المؤشرات العلوية ---
st.markdown(f"## 📊 {scenario_name.upper()} Scenario - Year {selected_year}")
st.markdown(f"*Data as of: {selected_year}*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_lc = year_data['LC_Score_%']
    st.metric(
        "LC Score",
        f"{current_lc:.2f}%",
        help="Local Content Percentage - The percentage of local spending relative to total project cost"
    )

with col2:
    delta = year_data['LC_Score_%'] - last_actual['LC_Score_%']
    
    # Determine color based on delta
    if delta > 0:
        color = "#00cc00"  # Green
        arrow = "↑"
    elif delta < 0:
        color = "#ff0000"  # Red
        arrow = "↓"
    else:
        color = "#808080"  # Gray
        arrow = "→"
    
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid {color};'>
        <p style='margin: 0; font-size: 14px; color: #888;'><b>Change vs 2024</b></p>
        <p style='margin: 10px 0 0 0; font-size: 28px; font-weight: bold; color: {color};'>{arrow} {delta:+.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.metric(
        "Total LC (SAR)",
        f"{year_data['Total_LC']/1e6:.2f}M",
        help="Total Local Content - Sum of Compensation, Goods/Services, Capital Expenses, Training & Depreciation"
    )

with col4:
    st.metric(
        "Total Cost (SAR)",
        f"{year_data['Total_Cost']/1e6:.2f}M",
        help="Total Project Cost - Overall project investment"
    )

st.divider()

# --- Charts | الرسوم البيانية ---
st.markdown("## 📈 Analysis & Visualization")

tabs = st.tabs(["📊 Trend", "📦 Components", "🔍 Comparison"])

with tabs[0]:
    st.markdown("### Local Content Score Trend")
    
    fig1 = px.line(
        df_scenario,
        x="Year",
        y="LC_Score_%",
        color="Type",
        markers=True,
        title=f"Historical vs Forecasted LC Score ({scenario_name})",
        labels={"LC_Score_%": "LC Score (%)", "Year": "Year"},
        color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"}
    )
    
    # Add vertical line for selected year
    fig1.add_vline(
        x=selected_year,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Year {selected_year}",
        annotation_position="top right"
    )
    
    # Add trend line for historical data
    df_hist_only = df_scenario[df_scenario['Type'] == 'Actual'].sort_values('Year')
    if len(df_hist_only) > 1:
        z = np.polyfit(df_hist_only['Year'], df_hist_only['LC_Score_%'], 1)
        p = np.poly1d(z)
        trend_years = df_hist_only['Year'].values
        trend_values = p(trend_years)
        fig1.add_scatter(x=trend_years, y=trend_values, mode='lines', name='Historical Trend',
                        line=dict(dash='dot', color='#1f77b4', width=2), showlegend=True)
    
    fig1.update_layout(
        hovermode='x unified',
        height=450,
        template="plotly_white",
        font=dict(size=12)
    )
    fig1.update_traces(marker=dict(size=8))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Show historical growth rates
    with st.expander("📊 Historical Growth Rates (2021-2024)"):
        st.markdown("**Year-over-Year Growth from Actual Data:**")
        growth_col1, growth_col2, growth_col3 = st.columns(3)
        with growth_col1:
            st.write(f"💰 Compensation: **{base_assumptions['comp']}%**")
            st.write(f"📦 Goods & Services: **{base_assumptions['goods']}%**")
        with growth_col2:
            st.write(f"🏗️ CapEx: **{base_assumptions['capex']}%**")
            st.write(f"🎓 Training: **{base_assumptions['train']}%**")
        with growth_col3:
            st.write(f"📊 Total Cost: **{base_assumptions['cost']}%**")
            st.write(f"*Used as Base scenario baseline*")
    
    # Show metrics for selected year
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("💯 LC Score (%)", f"{year_data['LC_Score_%']:.2f}", help="Local Content Percentage - The percentage of local spending relative to total project cost")
    with col2:
        st.metric("💰 Compensation (SAR)", f"{year_data['Compensation']/1e6:.2f}M", help="Employee Compensation & Benefits")
    with col3:
        st.metric("📦 Goods & Services (SAR)", f"{year_data['Goods_Services']/1e6:.2f}M", help="Procurement of Goods and Services")
    with col4:
        st.metric("🏗️ Capital Expenses (SAR)", f"{year_data['CapEx']/1e6:.2f}M", help="Capital Expenses - Investment in equipment and infrastructure")
    with col5:
        st.metric("🎓 Training (SAR)", f"{year_data['Training']/1e6:.2f}M", help="Employee Training & Development Programs")
    with col6:
        st.metric("💧 Depreciation (SAR)", f"{year_data['Depreciation']/1e6:.2f}M", help="Asset Depreciation & Amortization")

with tabs[1]:
    st.markdown("### Component Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig2 = px.bar(
            df_scenario,
            x="Year",
            y=["Compensation", "Goods_Services", "CapEx", "Training"],
            title="LC Components Growth Over Time",
            labels={"value": "Amount (SAR)", "variable": "Component"},
            barmode="group"
        )
        fig2.update_layout(
            height=450,
            template="plotly_white",
            hovermode='x unified',
            font=dict(size=12)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col1:
        # Breakdown for selected year
        selected_data = df_scenario[df_scenario['Year'] == selected_year].iloc[0]
        components = {
            'Compensation': selected_data['Compensation'],
            'Goods & Services': selected_data['Goods_Services'],
            'CapEx': selected_data['CapEx'],
            'Training': selected_data['Training'],
            'Depreciation': selected_data['Depreciation']
        }
        
        fig3 = px.pie(
            values=list(components.values()),
            names=list(components.keys()),
            title=f"{selected_year} LC Composition",
            hole=0.4
        )
        fig3.update_layout(
            height=450,
            font=dict(size=11)
        )
        st.plotly_chart(fig3, use_container_width=True)

with tabs[2]:
    st.markdown("### Scenario Comparison")
    
    if mode == "📊 View Scenarios":
        # Get values for selected year across all scenarios
        scenarios_year = df_all[df_all['Year'] == selected_year].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.bar(
                scenarios_year,
                x="Scenario",
                y="LC_Score_%",
                title=f"{selected_year} LC Score by Scenario",
                labels={"LC_Score_%": "LC Score (%)"},
                color="Scenario",
                color_discrete_map={
                    "base": "#2ca02c",
                    "conservative": "#ff7f0e",
                    "optimistic": "#d62728"
                }
            )
            fig4.update_layout(height=450, showlegend=False, template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Comparison metrics
            st.markdown(f"### Scenario Summary ({selected_year})")
            comparison_df = scenarios_year[['Scenario', 'Total_LC', 'Total_Cost', 'LC_Score_%']].copy()
            comparison_df.columns = ['Scenario', 'Total LC (SAR)', 'Total Cost (SAR)', 'LC Score (%)']
            
            st.dataframe(
                comparison_df.style.format({
                    'Total LC (SAR)': "{:,.0f}",
                    'Total Cost (SAR)': "{:,.0f}",
                    'LC Score (%)': "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("💡 Switch to 'View Scenarios' mode to compare different scenarios.")

st.divider()

# --- Detailed Data Table | جدول البيانات التفصيلية ---
st.markdown("## 📋 Detailed Data")

display_columns = ['Year', 'Type', 'Compensation', 'Goods_Services', 'CapEx', 'Training', 'Depreciation', 'Total_LC', 'Total_Cost', 'LC_Score_%']
df_display = df_scenario[display_columns].copy()

st.dataframe(
    df_display.style.format({
        'Compensation': "{:,.0f}",
        'Goods_Services': "{:,.0f}",
        'CapEx': "{:,.0f}",
        'Training': "{:,.0f}",
        'Depreciation': "{:,.0f}",
        'Total_LC': "{:,.0f}",
        'Total_Cost': "{:,.0f}",
        'LC_Score_%': "{:.2f}%"
    }),
    use_container_width=True,
    height=350
)

# --- Download Data | تحميل البيانات ---
st.markdown("## 📥 Export")

csv = df_display.to_csv(index=False)
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv,
    file_name=f"LC_Forecast_{scenario_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

if mode == "🎯 Custom Scenario":
    # Show assumptions summary
    st.markdown("## ⚙️ Custom Assumptions Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💰 Compensation (%)", f"{g_comp*100:.1f}")
    with col2:
        st.metric("📦 Goods & Services (%)", f"{g_goods*100:.1f}")
    with col3:
        st.metric("🏗️ Capital Expenses (%)", f"{g_capex*100:.1f}")
    with col4:
        st.metric("🎓 Training (%)", f"{g_train*100:.1f}")
    with col5:
        st.metric("📊 Total Cost (%)", f"{g_cost*100:.1f}")

else:
    # Display scenario comparison
    if mode == "📊 View Scenarios":
        st.markdown("## 📊 Scenario Comparison Details")
        
        scenarios_2028 = df_all[df_all['Year'] == 2028].copy()
        
        col1, col2, col3 = st.columns(3)
        
        base_data = scenarios_2028[scenarios_2028['Scenario'] == 'base'].iloc[0]
        conservative_data = scenarios_2028[scenarios_2028['Scenario'] == 'conservative'].iloc[0]
        optimistic_data = scenarios_2028[scenarios_2028['Scenario'] == 'optimistic'].iloc[0]
        
        with col1:
            st.markdown("### 🟢 Base Case")
            st.metric("LC Score (%)", f"{base_data['LC_Score_%']:.2f}", help="Local Content Percentage")
            st.metric("Total LC (SAR)", f"{base_data['Total_LC']/1e6:.2f}M", help="Total Local Content Spending")
            st.metric("Total Cost (SAR)", f"{base_data['Total_Cost']/1e6:.2f}M", help="Total Project Cost")
        
        with col2:
            st.markdown("### 🟠 Conservative")
            st.metric("LC Score (%)", f"{conservative_data['LC_Score_%']:.2f}", help="Local Content Percentage")
            st.metric("Total LC (SAR)", f"{conservative_data['Total_LC']/1e6:.2f}M", help="Total Local Content Spending")
            st.metric("Total Cost (SAR)", f"{conservative_data['Total_Cost']/1e6:.2f}M", help="Total Project Cost")
        
        with col3:
            st.markdown("### 🔴 Optimistic")
            st.metric("LC Score (%)", f"{optimistic_data['LC_Score_%']:.2f}", help="Local Content Percentage")
            st.metric("Total LC (SAR)", f"{optimistic_data['Total_LC']/1e6:.2f}M", help="Total Local Content Spending")
            st.metric("Total Cost (SAR)", f"{optimistic_data['Total_Cost']/1e6:.2f}M", help="Total Project Cost")

st.divider()
st.markdown("""
---
**📅 Local Content Forecasting Dashboard** | Updated: 2026-01-23  
**Built with:** Streamlit, Plotly, Pandas | License: Proprietary
""")


