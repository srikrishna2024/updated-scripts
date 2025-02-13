import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_asset_allocation(age, risk_appetite):
    """Dynamically adjust asset allocation based on age and risk appetite, keeping retirement age at 60."""
    if age < 40:
        allocation = {'Equity': 70, 'Debt': 20, 'Gold': 10} if risk_appetite == 'Aggressive' else \
                     {'Equity': 60, 'Debt': 30, 'Gold': 10} if risk_appetite == 'Moderate' else \
                     {'Equity': 50, 'Debt': 40, 'Gold': 10}
    elif 40 <= age < 55:
        allocation = {'Equity': 60, 'Debt': 30, 'Gold': 10} if risk_appetite == 'Aggressive' else \
                     {'Equity': 50, 'Debt': 40, 'Gold': 10} if risk_appetite == 'Moderate' else \
                     {'Equity': 40, 'Debt': 50, 'Gold': 10}
    elif 55 <= age < 60:
        allocation = {'Equity': 50, 'Debt': 40, 'Gold': 10} if risk_appetite == 'Aggressive' else \
                     {'Equity': 40, 'Debt': 50, 'Gold': 10} if risk_appetite == 'Moderate' else \
                     {'Equity': 30, 'Debt': 60, 'Gold': 10}
    else:
        allocation = {'Equity': 30, 'Debt': 60, 'Gold': 10} if risk_appetite == 'Aggressive' else \
                     {'Equity': 20, 'Debt': 70, 'Gold': 10} if risk_appetite == 'Moderate' else \
                     {'Equity': 10, 'Debt': 80, 'Gold': 10}
    return allocation

# Streamlit App UI
st.title("📊 Personalized Investment Planner")
st.write("Smart asset allocation based on your age & risk appetite, with retirement age set at 60.")

# User Inputs
age = st.slider("Select your age:", 18, 60, 30)
risk_appetite = st.selectbox("Select your risk appetite:", ["Conservative", "Moderate", "Aggressive"])

# Plot Asset Allocation Over Time
st.subheader("📈 Changing Asset Allocation Over Time (Retirement at 60)")
age_range = list(range(age, 61))
allocation_over_time = {key: [] for key in ['Equity', 'Debt', 'Gold']}
for a in age_range:
    alloc = get_asset_allocation(a, risk_appetite)
    for key in alloc:
        allocation_over_time[key].append(alloc[key])

fig = go.Figure()
for key, values in allocation_over_time.items():
    fig.add_trace(go.Scatter(x=age_range, y=values, mode='lines', name=key, hoverinfo='x+y'))
fig.add_vline(x=60, line=dict(color='red', dash='dash'), annotation_text='Retirement Age 60')
fig.update_layout(
    title="Asset Allocation Change Over Time (Retirement at 60)",
    xaxis_title="Age",
    yaxis_title="Allocation (%)",
    hovermode="x unified"
)
st.plotly_chart(fig)

# Justification for Asset Allocation Strategy
st.subheader("📢 Justification for Asset Allocation")
st.write(
    "As you age, your ability to take financial risks reduces. Younger investors can afford to allocate more to equity "
    "since they have a longer investment horizon to recover from market downturns. As retirement approaches, the focus "
    "shifts towards capital preservation, hence increasing debt allocation. Gold provides a hedge against inflation and "
    "economic uncertainty, maintaining stability in the portfolio. This strategy aims to reduce sequence of returns risk, "
    "ensuring a smoother transition into retirement with lower volatility and more predictable income sources."
)
