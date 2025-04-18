import streamlit as st
import pandas as pd
import psycopg
from datetime import datetime
import plotly.graph_objects as go
import locale

# Set Indian number format
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

def format_inr(value):
    try:
        return f"₹{locale.format_string('%0.2f', value, grouping=True)}"
    except:
        return f"₹{value:,.2f}"

# DB connection
def connect_db():
    return psycopg.connect(
        dbname='postgres', user='postgres', password='admin123', host='localhost', port='5432'
    )

# Fetch goal mappings
def get_goals():
    with connect_db() as conn:
        query = """
            SELECT goal_name, scheme_name, scheme_code, current_value
            FROM goals
            ORDER BY goal_name
        """
        return pd.read_sql(query, conn)

# Fetch investment history
def get_transactions(scheme_codes):
    with connect_db() as conn:
        query = """
            SELECT * FROM portfolio_data
            WHERE code = ANY(%s)
        """
        return pd.read_sql(query, conn, params=(scheme_codes,))

# Fetch benchmark data
def get_benchmark():
    with connect_db() as conn:
        return pd.read_sql("SELECT date, price FROM benchmark ORDER BY date", conn)

# Calculate invested amount
def get_invested_amount(df):
    invest_txns = df[df['transaction_type'].isin(['invest', 'switch_in'])]
    return invest_txns['amount'].sum()

# Estimate future value
def project_value(current_value, annual_return, years):
    return current_value * ((1 + annual_return / 100) ** years)

# Estimate milestone dates
def milestone_dates(current_value, target, target_date, milestones):
    remaining_days = (target_date - datetime.today().date()).days
    growth_factor = (target / current_value) if current_value > 0 else 1
    dates = []
    for m in milestones:
        percent = m / 100
        milestone_days = int(remaining_days * percent / growth_factor)
        est_date = datetime.today() + pd.Timedelta(days=milestone_days)
        dates.append(est_date.strftime('%b %Y'))
    return dates

# Main
st.set_page_config(page_title="Am I On Track?", layout="wide")
st.title("🌟 Am I On Track Dashboard")

goals_df = get_goals()
if goals_df.empty:
    st.info("No goal mappings available. Please map investments to goals first.")
    st.stop()

# Input
selected_goal = st.selectbox("Select Goal", goals_df['goal_name'].unique())
target_amount = st.number_input("Target Amount (₹)", value=1000000, step=10000)
target_date = st.date_input("Target Date", value=datetime(2030, 3, 31))

# Projected annual investments section
with st.expander("📅 Projected Future Investments"):
    projected_annual_investment = st.number_input("Annual Investment till Target Date (₹)", value=60000, step=10000)
    investment_growth_rate = st.slider("Expected Annual Return for Your Fund (%)", 5, 20, 12)
    benchmark_growth_rate = st.slider("Expected Benchmark Return (%)", 5, 20, 10)
    analyze_investments = st.button("Analyze Projected Investments")

# What-If Analysis section
with st.expander("🧪 What-If Analysis"):
    st.markdown("""
    **How to use this:**
    - Adjust your monthly SIP amount, expected return, and time to see projected corpus.
    - Helps you simulate scenarios to decide if you need to invest more or adjust your target.
    """)
    sip = st.slider("Monthly SIP (₹)", min_value=1000, max_value=50000, value=5000, step=1000)
    expected_return = st.slider("Expected Annual Return (%)", 5, 20, 12)
    time_years = st.slider("Years Remaining", 1, 30, 10)
    analyze_what_if = st.button("Analyze What-If Scenario")

# Run Projected Investment Analysis
if analyze_investments:
    goal_data = goals_df[goals_df['goal_name'] == selected_goal]
    scheme_codes = goal_data['scheme_code'].unique().tolist()
    transactions = get_transactions(scheme_codes)

    if transactions.empty:
        st.warning("No transactions found for this goal.")
        st.stop()

    invested_amount = get_invested_amount(transactions)
    current_value = goal_data['current_value'].sum()
    progress = min(current_value / target_amount, 1.0)

    st.subheader(f"Summary for Goal: {selected_goal}")
    st.metric("Target", format_inr(target_amount))
    st.metric("Invested", format_inr(invested_amount))
    st.metric("Current Value", format_inr(current_value))
    st.progress(progress)

    st.subheader("🚩 Milestone Tracker")
    milestones = [25, 50, 75, 100]
    dates = milestone_dates(current_value, target_amount, target_date, milestones)
    for i, m in enumerate(milestones):
        st.write(f"{m}% = {format_inr(target_amount * m / 100)} by {dates[i]}")

    st.subheader("📊 Am I On Track?")
    years_left = (target_date - datetime.today().date()).days / 365
    future_fund_value = current_value * ((1 + investment_growth_rate / 100) ** years_left)
    future_contributions = projected_annual_investment * (((1 + investment_growth_rate / 100) ** years_left - 1) / (investment_growth_rate / 100))
    projected_total = future_fund_value + future_contributions

    if projected_total >= target_amount:
        st.success(f"✅ On Track! Projected Corpus: {format_inr(projected_total)}")
    else:
        st.warning(f"⚠️ Off Track! Shortfall: {format_inr(target_amount - projected_total)}")

    st.subheader("📈 Performance Comparison")
    benchmark = get_benchmark()
    transactions['date'] = pd.to_datetime(transactions['date'])
    benchmark['date'] = pd.to_datetime(benchmark['date'])

    fund_chart = transactions.groupby('date')['amount'].sum().cumsum().reset_index(name='Your Fund')
    benchmark_chart = benchmark.rename(columns={'price': 'Benchmark'})
    merged = pd.merge(fund_chart, benchmark_chart, on='date', how='inner')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['date'], y=merged['Your Fund'], mode='lines', name='Your Fund'))
    fig.add_trace(go.Scatter(x=merged['date'], y=merged['Benchmark'], mode='lines', name='Nifty50 TRI'))
    fig.update_layout(
        title='Fund vs Benchmark Over Time',
        xaxis_title='Date',
        yaxis_title='Value (₹)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Run What-If Scenario independently
if analyze_what_if:
    monthly_rate = expected_return / 100 / 12
    fv = sip * (((1 + monthly_rate) ** (12 * time_years) - 1) / monthly_rate) * (1 + monthly_rate)
    st.subheader("🔍 What-If Projection Result")
    st.write(f"📈 Projected Value with What-If: **{format_inr(fv)}**")

    # Plot What-If SIP growth
    months = list(range(1, 12 * time_years + 1))
    values = [sip * (((1 + monthly_rate) ** m - 1) / monthly_rate) * (1 + monthly_rate) for m in months]
    fig_sip = go.Figure()
    fig_sip.add_trace(go.Scatter(x=months, y=values, mode='lines+markers', name='SIP Projection'))
    fig_sip.update_layout(
        title="What-If SIP Growth Over Time",
        xaxis_title="Months",
        yaxis_title="Corpus (₹)",
        height=400
    )
    st.plotly_chart(fig_sip, use_container_width=True)
