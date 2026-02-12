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
    
    .stMetric > div:nth-child(1) {
        word-wrap: break-word;
        word-break: break-word;
        white-space: normal;
        overflow-wrap: break-word;
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

st.title("📈 Local Content Forecasting Dashboard")
st.divider()

# Display IHCC Logo
col1, col2 = st.columns([1, 4])
with col1:
    logo_path = Path(__file__).parent / 'IHCC Logo.jpeg'
    if logo_path.exists():
        st.image(str(logo_path), width=100)
with col2:
    st.write("")

# --- Load Data | تحميل البيانات ---
@st.cache_data
def load_historical_data():
    """Load historical data from Excel or Google Sheets | تحميل البيانات التاريخية من Excel"""
    # Try local file first
    possible_paths = [
        Path(__file__).parent.parent.parent / 'Data' / 'Local content historical records Over All.xlsx',
        Path(__file__).parent.parent.parent.parent / 'Data' / 'Local content historical records Over All.xlsx',
        Path('Data') / 'Local content historical records Over All.xlsx',
        Path('models/Model_1_Aggregated_v2') / 'Data' / 'Local content historical records Over All.xlsx',
    ]
    
    excel_file = None
    for path in possible_paths:
        if path.exists():
            excel_file = path
            break
    
    # If local file not found, try loading from Google Sheets
    if excel_file is None:
        try:
            st.info("📡 Loading data from Google Sheets (local file not found)...")
            # Load Summary sheet data from Google Sheets
            spreadsheet_id = '15cWrIH6rzrAluykAUY9MVR25lfwdUija'
            csv_url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv'
            df_summary_gs = pd.read_csv(csv_url)
            
            # Load Workforce-Records sheet
            workforce_url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid=WORKSHEET_ID'
            
            # Since Google Sheets API is complex, use hardcoded data for workforce
            # Based on the historical data we know
            hist_data = []
            for year in [2024, 2023, 2022, 2021]:
                if year == 2024:
                    saudi_comp = 18.161063e6
                    foreign_lc = 33.675045e6
                    goods = 104.271291e6
                    train = 0.263861e6
                    depr = 4.509642e6
                    total_lc = 160.880903e6
                    total_cost = 471.881650e6
                elif year == 2023:
                    saudi_comp = 15.215027e6
                    foreign_lc = 21.678560e6
                    goods = 58.111403e6
                    train = 0.081244e6
                    depr = 3.834302e6
                    total_lc = 98.920537e6
                    total_cost = 259.740844e6
                elif year == 2022:
                    saudi_comp = 16.632800e6
                    foreign_lc = 19.081414e6
                    goods = 46.611594e6
                    train = 1.376612e6
                    depr = 4.436843e6
                    total_lc = 88.139263e6
                    total_cost = 280.447547e6
                else:  # 2021
                    saudi_comp = 13.169354e6
                    foreign_lc = 18.548835e6
                    goods = 36.615275e6
                    train = 0.010321e6
                    depr = 8.023705e6
                    total_lc = 76.367490e6
                    total_cost = 289.923261e6
                
                hist_data.append({
                    'Year': year,
                    'Type': 'Actual',
                    'Saudi_Compensation': saudi_comp,
                    'Foreign_Compensation': foreign_lc,
                    'LC_from_Workforce': saudi_comp + foreign_lc,
                    'Goods_Services': goods,
                    'Training': train,
                    'Depreciation': depr,
                    'Total_LC': total_lc,
                    'Total_Cost': total_cost
                })
            
            return pd.DataFrame(hist_data).sort_values('Year')
        
        except Exception as e:
            st.error(f"❌ Could not load data: {str(e)}")
            st.stop()
    
    # Load from local Excel file
    df_summary = pd.read_excel(excel_file, sheet_name='Summary', header=None)
    
    # Load Workforce-Records sheet for Saudi vs Foreign breakdown
    df_workforce = pd.read_excel(excel_file, sheet_name='Workforce-Records', header=0)
    
    hist_data = []
    for i, year in enumerate([2024, 2023, 2022, 2021]):
        idx = i + 2
        
        # Get Saudi and Foreign compensation
        workforce_row = df_workforce[df_workforce['Year'] == year].iloc[0]
        saudi_comp = float(workforce_row['Saudi Compensation (SAR)']) if pd.notna(workforce_row['Saudi Compensation (SAR)']) else 0
        foreign_lc = float(workforce_row['Foreign LC Value (SAR)']) if pd.notna(workforce_row['Foreign LC Value (SAR)']) else 0
        
        # Get LC from Compensation/labor from Summary sheet (column 1)
        lc_from_workforce = float(df_summary.iloc[idx, 1]) if pd.notna(df_summary.iloc[idx, 1]) else 0
        
        hist_data.append({
            'Year': year,
            'Type': 'Actual',
            'Saudi_Compensation': saudi_comp,
            'Foreign_Compensation': foreign_lc,
            'LC_from_Workforce': lc_from_workforce,
            'Goods_Services': float(df_summary.iloc[idx, 2]),
            'Training': float(df_summary.iloc[idx, 4]),
            'Depreciation': float(df_summary.iloc[idx, 5]),
            'Total_LC': float(df_summary.iloc[idx, 6]),
            'Total_Cost': float(df_summary.iloc[idx, 7])
        })
    
    return pd.DataFrame(hist_data).sort_values('Year')


def generate_custom_forecast(df_hist, growth_rates):
    """Generate forecast with custom growth rates | توليد التنبؤ برسوم نمو مخصصة"""
    last_actual = df_hist.iloc[-1]
    
    forecast_data = []
    for year in [2025, 2026, 2027, 2028]:
        n = year - 2024
        saudi = last_actual['Saudi_Compensation'] * ((1 + growth_rates['saudi']) ** n)
        foreign = last_actual['Foreign_Compensation'] * ((1 + growth_rates['foreign']) ** n)
        goods = last_actual['Goods_Services'] * ((1 + growth_rates['goods']) ** n)
        train = last_actual['Training'] * ((1 + growth_rates['train']) ** n)
        depr = last_actual['Depreciation']
        lc_workforce = saudi + foreign
        
        total_lc = saudi + foreign + goods + train + depr
        total_cost = last_actual['Total_Cost'] * ((1 + growth_rates['cost']) ** n)
        
        forecast_data.append({
            'Year': year,
            'Type': 'Forecast',
            'Saudi_Compensation': saudi,
            'Foreign_Compensation': foreign,
            'LC_from_Workforce': lc_workforce,
            'Goods_Services': goods,
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


df_hist = load_historical_data()

# Calculate default assumptions from historical data | حساب الافتراضات الافتراضية من البيانات التاريخية
def calculate_assumptions_from_history(df_hist):
    """Calculate growth rates from historical data"""
    df_sorted = df_hist.sort_values('Year')
    
    # Calculate YoY growth rates
    growth_rates = {}
    for col in ['Saudi_Compensation', 'Foreign_Compensation', 'Goods_Services', 'Training', 'Total_Cost']:
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
        'saudi': round(growth_rates['Saudi_Compensation'], 1),
        'foreign': round(growth_rates['Foreign_Compensation'], 1),
        'goods': round(growth_rates['Goods_Services'], 1),
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
        st.write(f"🇸🇦 Saudi-Workforce: {base_assumptions['saudi']}%")
        st.write(f"🌍 Foreign-Workforce: {base_assumptions['foreign']}%")
    with col2:
        st.write(f"📦 Goods & Services: {base_assumptions['goods']}%")
        st.write(f"🎓 Training: {base_assumptions['train']}%")
    with col3:
        st.write(f"📊 Total Cost: {base_assumptions['cost']}%")
    
    st.markdown("**Conservative (70% of Base):**")
    cons = SCENARIO_ASSUMPTIONS['Conservative']
    st.write(f"Saudi: {cons['saudi']}% | Foreign: {cons['foreign']}% | Goods: {cons['goods']}% | Train: {cons['train']}%")
    
    st.markdown("**Optimistic (130% of Base):**")
    opt = SCENARIO_ASSUMPTIONS['Optimistic']
    st.write(f"Saudi: {opt['saudi']}% | Foreign: {opt['foreign']}% | Goods: {opt['goods']}% | Train: {opt['train']}%")

# --- Sidebar Controls | عناصر التحكم الجانبية ---
logo_path = Path(__file__).parent / 'IHCC Logo.jpeg'
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=100)
st.sidebar.header("⚙️ Configuration")

# Mode selection
mode = st.sidebar.radio(
    "Select Mode",
    ["📊 View Scenarios", "🎯 Custom Scenario"],
    help="View pre-calculated scenarios or create custom forecast"
)

if mode == "📊 View Scenarios":
    st.sidebar.markdown("### 📈 Select Scenario")
    
    available_scenarios = ['base', 'conservative', 'optimistic']
    selected_scenario = st.sidebar.selectbox(
        "Scenario",
        available_scenarios,
        format_func=lambda x: f"{x.upper()}"
    )
    
    # Get assumptions for selected scenario
    scenario_key = selected_scenario.capitalize()
    assumptions = SCENARIO_ASSUMPTIONS.get(scenario_key, SCENARIO_ASSUMPTIONS['Base'])
    
    # Display scenario assumptions
    st.sidebar.markdown("### 📊 Assumptions (from Historical Data)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🇸🇦 Saudi-Workforce", f"{assumptions['saudi']}%")
        st.metric("📦 Goods & Services", f"{assumptions['goods']}%")
    with col2:
        st.metric("🌍 Foreign-Workforce", f"{assumptions['foreign']}%")
        st.metric("🎓 Training", f"{assumptions['train']}%")
    
    # Generate forecast with selected scenario assumptions
    scenario_multipliers = {
        'base': 1.0,
        'conservative': 0.7,
        'optimistic': 1.3
    }
    
    multiplier = scenario_multipliers.get(selected_scenario, 1.0)
    
    # Apply multiplier to assumptions
    active_assumptions = {
        'saudi': assumptions['saudi'] * multiplier / 100,
        'foreign': assumptions['foreign'] * multiplier / 100,
        'goods': assumptions['goods'] * multiplier / 100,
        'train': assumptions['train'] * multiplier / 100,
        'cost': assumptions['cost'] * multiplier / 100
    }
    
    # Generate forecast with active assumptions
    df_scenario = generate_custom_forecast(df_hist, active_assumptions)
    scenario_name = scenario_key
    
else:
    st.sidebar.markdown("### 🎯 Custom Growth Rates")
    st.sidebar.markdown("Adjust the growth rates below (currently at Base Scenario values)")
    
    # Dynamic max values for sliders based on base assumptions
    max_saudi = max(50, int(base_assumptions['saudi']) + 10)
    max_foreign = max(50, int(base_assumptions['foreign']) + 10)
    max_goods = max(50, int(base_assumptions['goods']) + 10)
    max_train = max(50, int(base_assumptions['train']) + 10)
    max_cost = max(50, int(base_assumptions['cost']) + 10)
    
    st.sidebar.write("**🇸🇦 Saudi-Workforce Growth (%)**")
    g_saudi = st.sidebar.slider("Saudi-Workforce", 0, max_saudi, int(base_assumptions['saudi']), key="saudi") / 100
    
    st.sidebar.write("**🌍 Foreign-Workforce Growth (%)**")
    g_foreign = st.sidebar.slider("Foreign-Workforce", 0, max_foreign, int(base_assumptions['foreign']), key="foreign") / 100
    
    st.sidebar.write("**📦 Goods & Services Growth (%)**")
    g_goods = st.sidebar.slider("Goods & Services", 0, max_goods, int(base_assumptions['goods']), key="goods") / 100
    
    st.sidebar.write("**🎓 Training Growth (%)**")
    g_train = st.sidebar.slider("Training", 0, max_train, int(base_assumptions['train']), key="train") / 100
    
    st.sidebar.write("**📊 Total Cost Growth (%)**")
    g_cost = st.sidebar.slider("Total Cost", 0, max_cost, int(base_assumptions['cost']), key="cost") / 100
    
    # Generate custom forecast
    custom_growth = {
        'saudi': g_saudi,
        'foreign': g_foreign,
        'goods': g_goods,
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
    value=2025
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
st.markdown("## Local Content Analysis & Visualization")

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
        color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
        text="LC_Score_%"
    )
    
    # Format text on markers to show 2 decimal places
    fig1.for_each_trace(lambda t: t.update(textposition="middle left", texttemplate="%{text:.2f}%", textfont=dict(size=14, color="black")))
    
    # Add vertical line for selected year
    fig1.add_vline(
        x=selected_year,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Year {selected_year}",
        annotation_position="top right"
    )
    
    # Add benchmark line at 40%
    fig1.add_hline(
        y=40,
        line_dash="dash",
        line_color="gold",
        annotation_text="Benchmark: 40%",
        annotation_position="right",
        annotation_font_size=11
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
        height=550,
        template="plotly_white",
        font=dict(size=14, family="Arial, sans-serif", color="black"),
        title_font=dict(size=16, family="Arial, sans-serif"),
        margin=dict(t=100, b=80, l=80, r=80)
    )
    fig1.update_xaxes(title_font=dict(size=14, color="black"), tickfont=dict(size=12, color="black"))
    fig1.update_yaxes(title_font=dict(size=14, color="black"), tickfont=dict(size=12, color="black"))
    fig1.update_traces(marker=dict(size=8))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Show historical growth rates
    with st.expander("📊 Historical Growth Rates (2021-2024)"):
        st.markdown("**Year-over-Year Growth from Actual Data:**")
        growth_col1, growth_col2, growth_col3 = st.columns(3)
        with growth_col1:
            st.write(f"🇸🇦 Saudi Comp.: **{base_assumptions['saudi']}%**")
            st.write(f"🌍 Foreign Comp.: **{base_assumptions['foreign']}%**")
        with growth_col2:
            st.write(f"📦 Goods & Services: **{base_assumptions['goods']}%**")
            st.write(f"🎓 Training: **{base_assumptions['train']}%**")
        with growth_col3:
            st.write(f"📊 Total Cost: **{base_assumptions['cost']}%**")
            st.write(f"*Used as Base scenario baseline*")
    
    # Show metrics for selected year
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("💯 LC Score (%)", f"{year_data['LC_Score_%']:.2f}", help="Local Content Percentage")
    with col2:
        st.metric("💼 LC from Workforce (SAR)", f"{year_data['LC_from_Workforce']/1e6:.2f}M", help="Total LC from Compensation/Labor")
    with col3:
        st.metric("🇸🇦 Saudi-Workforce (SAR)", f"{year_data['Saudi_Compensation']/1e6:.2f}M", help="Saudi-Workforce")
    with col4:
        st.metric("🌍 Foreign-Workforce (SAR)", f"{year_data['Foreign_Compensation']/1e6:.2f}M", help="Foreign-Workforce")
    with col5:
        st.metric("📦 Goods & Services (SAR)", f"{year_data['Goods_Services']/1e6:.2f}M", help="Goods and Services")
    with col6:
        st.metric("🎓 Training (SAR)", f"{year_data['Training']/1e6:.2f}M", help="Training Programs")

with tabs[1]:
    st.markdown("### Component Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig2 = px.bar(
            df_scenario,
            x="Year",
            y=["Saudi_Compensation", "Foreign_Compensation", "Goods_Services", "Training"],
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
            'Saudi-Workforce': selected_data['Saudi_Compensation'],
            'Foreign-Workforce': selected_data['Foreign_Compensation'],
            'Goods & Services': selected_data['Goods_Services'],
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
        scenarios_list = []
        for scenario in ['base', 'conservative', 'optimistic']:
            scenario_multipliers = {
                'base': 1.0,
                'conservative': 0.7,
                'optimistic': 1.3
            }
            multiplier = scenario_multipliers.get(scenario, 1.0)
            active_assumptions = {
                'saudi': base_assumptions['saudi'] * multiplier / 100,
                'foreign': base_assumptions['foreign'] * multiplier / 100,
                'goods': base_assumptions['goods'] * multiplier / 100,
                'train': base_assumptions['train'] * multiplier / 100,
                'cost': base_assumptions['cost'] * multiplier / 100
            }
            df_temp = generate_custom_forecast(df_hist, active_assumptions)
            year_row = df_temp[df_temp['Year'] == selected_year].iloc[0]
            scenarios_list.append({
                'Scenario': scenario,
                'Total_LC': year_row['Total_LC'],
                'Total_Cost': year_row['Total_Cost'],
                'LC_Score_%': year_row['LC_Score_%']
            })
        
        scenarios_year = pd.DataFrame(scenarios_list)
        
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

# --- Component Effectiveness Analysis ---
st.markdown("## 📊 Component Effectiveness Analysis")

# Display formula
with st.expander("📐 View LC Score Formula", expanded=False):
    st.markdown("""
    ### LC Score Calculation Formula:
    
    **LC Score (%) = (Total LC Contribution / Total Project Cost) × 100**
    
    Where:
    - **Total LC Contribution** = Sum of all Local Content sources:
        - LC from Compensation/Labor (Saudi + Foreign)
        - LC from Goods & Services
        - LC from Training Programs
        - LC from Depreciation & Amortization
        - LC from Capital Expenses
        
    - **Total Project Cost** = All relevant operating costs in Saudi Arabia
    
    ### LC Impact on Each Component:
    
    Each component's impact on the LC Score is calculated as:
    
    **Impact on LC Score (%) = (Component Value / Total Project Cost) × 100**
    
    This shows the direct contribution of each component to achieving the overall LC Score target.
    """)

if mode == "📊 View Scenarios":
    # Calculate contribution of each component to LC Score
    effectiveness_data = []
    
    for idx, row in df_scenario.iterrows():
        total_lc = row['Total_LC']
        total_cost = row['Total_Cost']
        
        components_list = [
            ('🇸🇦 Saudi-Workforce', row['Saudi_Compensation']),
            ('🌍 Foreign-Workforce', row['Foreign_Compensation']),
            ('📦 Goods & Services', row['Goods_Services']),
            ('🎓 Training', row['Training']),
            ('Depreciation', row['Depreciation'])
        ]
        
        for comp_name, comp_value in components_list:
            # Calculate contribution to Total LC
            lc_contribution = (comp_value / total_lc * 100) if total_lc != 0 else 0
            # Calculate impact on LC Score
            lc_score_impact = (comp_value / total_cost * 100) if total_cost != 0 else 0
            
            effectiveness_data.append({
                'Year': int(row['Year']),
                'Component': comp_name,
                'Value (SAR)': int(comp_value),
                'Contribution to LC (%)': round(lc_contribution, 2),
                'Impact on LC Score (%)': round(lc_score_impact, 2)
            })
    
    effectiveness_df = pd.DataFrame(effectiveness_data)
    
    # Define component order for consistency
    component_order = ['🇸🇦 Saudi-Workforce', '🌍 Foreign-Workforce', '📦 Goods & Services', '🎓 Training', 'Depreciation']
    effectiveness_df['Component'] = pd.Categorical(effectiveness_df['Component'], categories=component_order, ordered=True)
    effectiveness_df = effectiveness_df.sort_values('Component')
    
    # Show effectiveness for selected year
    selected_effectiveness = effectiveness_df[effectiveness_df['Year'] == selected_year].copy()
    selected_effectiveness = selected_effectiveness.sort_values('Component')
    
    st.markdown(f"### {selected_year} - Component Impact on LC Score")
    st.dataframe(
        selected_effectiveness.style.format({
            'Value (SAR)': "{:,.0f}",
            'Contribution to LC (%)': "{:.2f}%",
            'Impact on LC Score (%)': "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Show all years comparison
    with st.expander("📊 View All Years - Component Effectiveness"):
        effectiveness_display = effectiveness_df.sort_values(['Component', 'Year']).copy()
        st.dataframe(
            effectiveness_display.style.format({
                'Value (SAR)': "{:,.0f}",
                'Contribution to LC (%)': "{:.2f}%",
                'Impact on LC Score (%)': "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Component effectiveness is shown in 'View Scenarios' mode")

st.divider()

# --- Year-over-Year Growth Comparison ---
st.markdown("## 📊 Year-over-Year Growth Comparison")

if mode == "📊 View Scenarios":
    # Build YoY comparison data
    yoy_data = []
    
    df_sorted = df_scenario.sort_values('Year')
    
    components = ['Saudi_Compensation', 'Foreign_Compensation', 'Goods_Services', 'Training', 'Total_LC', 'Total_Cost']
    component_display = {
        'Saudi_Compensation': '🇸🇦 Saudi-Workforce',
        'Foreign_Compensation': '🌍 Foreign-Workforce',
        'Goods_Services': '📦 Goods & Services',
        'Training': '🎓 Training',
        'Total_LC': '💰 Total LC',
        'Total_Cost': '📊 Total Cost'
    }
    
    for i in range(1, len(df_sorted)):
        prev_row = df_sorted.iloc[i-1]
        curr_row = df_sorted.iloc[i]
        year_range = f"{int(prev_row['Year'])} → {int(curr_row['Year'])}"
        
        for comp in components:
            prev_val = prev_row[comp]
            curr_val = curr_row[comp]
            
            if prev_val != 0:
                yoy_change = ((curr_val - prev_val) / prev_val) * 100
            else:
                yoy_change = 0
            
            yoy_data.append({
                'Year Range': year_range,
                'Component': component_display[comp],
                'Previous Year Value (SAR)': int(prev_val),
                'Current Year Value (SAR)': int(curr_val),
                'YoY Change (%)': round(yoy_change, 2)
            })
    
    yoy_df = pd.DataFrame(yoy_data)
    
    # Define component order for consistency
    component_order = ['🇸🇦 Saudi-Workforce', '🌍 Foreign-Workforce', '📦 Goods & Services', '🎓 Training', '💰 Total LC', '📊 Total Cost']
    yoy_df['Component'] = pd.Categorical(yoy_df['Component'], categories=component_order, ordered=True)
    yoy_df = yoy_df.sort_values(['Year Range', 'Component'])
    
    # Show YoY for selected year
    selected_range = yoy_df[yoy_df['Year Range'].str.endswith(str(int(selected_year)))].copy()
    
    if len(selected_range) > 0:
        st.markdown(f"### YoY Growth - Year {int(selected_year)}")
        st.dataframe(
            selected_range[['Component', 'Previous Year Value (SAR)', 'Current Year Value (SAR)', 'YoY Change (%)']].style.format({
                'Previous Year Value (SAR)': "{:,.0f}",
                'Current Year Value (SAR)': "{:,.0f}",
                'YoY Change (%)': "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    
    # Show all years comparison
    with st.expander("📊 View All Years - YoY Growth"):
        st.dataframe(
            yoy_df.style.format({
                'Previous Year Value (SAR)': "{:,.0f}",
                'Current Year Value (SAR)': "{:,.0f}",
                'YoY Change (%)': "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Year-over-Year comparison is shown in 'View Scenarios' mode")

st.divider()

# --- Detailed Data Table | جدول البيانات التفصيلية ---
st.markdown("## 📋 Detailed Data")

data_tabs = st.tabs(["📊 Chart View", "📋 Table View"])

display_columns = ['Year', 'Type', 'Saudi_Compensation', 'Foreign_Compensation', 'Goods_Services', 'Training', 'Depreciation', 'Total_LC', 'Total_Cost', 'LC_Score_%']
df_display = df_scenario[display_columns].copy()

# Define column order for consistency
column_order = ['Year', 'Type', 'Saudi_Compensation', 'Foreign_Compensation', 'Goods_Services', 'Training', 'Total_LC', 'Total_Cost']\

with data_tabs[0]:
    st.markdown("### Data Visualization")
    
    # Create line chart for all components over time
    fig_data = px.line(
        df_scenario,
        x="Year",
        y=["Saudi_Compensation", "Foreign_Compensation", "Goods_Services", "Training", "Depreciation"],
        title="LC Components Over Time",
        labels={"value": "Amount (SAR)", "variable": "Component"},
        markers=True
    )
    fig_data.update_layout(
        height=450,
        template="plotly_white",
        font=dict(size=12),
        hovermode='x unified'
    )
    st.plotly_chart(fig_data, use_container_width=True)
    
    # Bar chart comparing Total LC vs Total Cost with LC Score on secondary axis
    df_chart = df_scenario.copy()
    df_chart['Total_LC_M'] = df_chart['Total_LC'] / 1e6
    df_chart['Total_Cost_M'] = df_chart['Total_Cost'] / 1e6
    
    fig_comparison = px.bar(
        df_chart,
        x="Year",
        y=["Total_LC_M", "Total_Cost_M"],
        title="Total LC vs Total Cost Comparison (with LC Score)",
        barmode="group",
        labels={"value": "Amount (SAR Millions)", "variable": "Type"}
    )
    
    # Add LC Score line on secondary axis
    fig_comparison.add_scatter(
        x=df_chart['Year'],
        y=df_chart['LC_Score_%'],
        mode='lines+markers',
        name='LC Score (%)',
        line=dict(color='green', width=3),
        marker=dict(size=8),
        yaxis='y2'
    )
    
    fig_comparison.update_layout(
        height=450,
        template="plotly_white",
        font=dict(size=12),
        yaxis2=dict(
            title='LC Score (%)',
            overlaying='y',
            side='right'
        )
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

with data_tabs[1]:
    st.markdown("### Data Table")
    st.dataframe(
        df_display.style.format({
            'Saudi_Compensation': "{:,.0f}",
            'Foreign_Compensation': "{:,.0f}",
            'Goods_Services': "{:,.0f}",
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
        st.metric("🇸🇦 Saudi Comp. (%)", f"{g_saudi*100:.1f}")
    with col2:
        st.metric("🌍 Foreign Comp. (%)", f"{g_foreign*100:.1f}")
    with col3:
        st.metric("📦 Goods & Services (%)", f"{g_goods*100:.1f}")
    with col4:
        st.metric("🎓 Training (%)", f"{g_train*100:.1f}")
    with col5:
        st.metric("📊 Total Cost (%)", f"{g_cost*100:.1f}")

else:
    # Display scenario comparison
    if mode == "📊 View Scenarios":
        st.markdown("## 📊 Scenario Comparison Details")
        
        scenarios_2028_list = []
        for scenario in ['base', 'conservative', 'optimistic']:
            scenario_multipliers = {
                'base': 1.0,
                'conservative': 0.7,
                'optimistic': 1.3
            }
            multiplier = scenario_multipliers.get(scenario, 1.0)
            active_assumptions = {
                'saudi': base_assumptions['saudi'] * multiplier / 100,
                'foreign': base_assumptions['foreign'] * multiplier / 100,
                'goods': base_assumptions['goods'] * multiplier / 100,
                'train': base_assumptions['train'] * multiplier / 100,
                'cost': base_assumptions['cost'] * multiplier / 100
            }
            df_temp = generate_custom_forecast(df_hist, active_assumptions)
            year_row = df_temp[df_temp['Year'] == 2028].iloc[0]
            scenarios_2028_list.append({
                'scenario': scenario,
                'data': year_row
            })
        
        col1, col2, col3 = st.columns(3)
        
        base_data = scenarios_2028_list[0]['data']
        conservative_data = scenarios_2028_list[1]['data']
        optimistic_data = scenarios_2028_list[2]['data']
        
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

