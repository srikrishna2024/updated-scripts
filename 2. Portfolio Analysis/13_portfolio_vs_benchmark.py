import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import psycopg

def connect_to_db():
    """Create database connection"""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_portfolio_data():
    """Retrieve portfolio data with scheme details"""
    with connect_to_db() as conn:
        query = """
            SELECT p.date, p.scheme_name, p.code, p.transaction_type, 
                   p.value, p.units, p.amount
            FROM portfolio_data p
            ORDER BY p.date
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs for all funds"""
    with connect_to_db() as conn:
        query = """
            SELECT code, value as nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav)
                FROM mutual_fund_nav
                GROUP BY code
            )
        """
        return pd.read_sql(query, conn)

def get_benchmark_data():
    """Retrieve benchmark data"""
    with connect_to_db() as conn:
        query = """
            SELECT date, price
            FROM benchmark
            ORDER BY date
        """
        return pd.read_sql(query, conn)

def calculate_units(df):
    """Calculate net units for each scheme based on transactions"""
    df['units_change'] = df.apply(lambda x: 
        x['units'] if x['transaction_type'] in ('invest', 'switch_in')
        else -x['units'] if x['transaction_type'] in ('redeem', 'switch_out')
        else 0,
        axis=1
    )
    
    # Calculate cumulative units for each scheme
    df = df.sort_values(['scheme_name', 'date'])
    df['cumulative_units'] = df.groupby(['scheme_name', 'code'])['units_change'].cumsum()
    return df

def calculate_growth(portfolio_df, benchmark_df, latest_nav, selected_fund):
    """Calculate growth for both portfolio and benchmark investments"""
    # Filter for selected fund
    fund_data = portfolio_df[portfolio_df['scheme_name'] == selected_fund].copy()
    
    if fund_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get the fund code
    fund_code = fund_data['code'].iloc[0]
    
    # Get the latest NAV for this fund
    fund_nav = latest_nav[latest_nav['code'] == fund_code]['nav_value'].iloc[0]
    
    # Calculate current value for each date
    fund_growth = []
    unique_dates = fund_data['date'].unique()
    
    for date in unique_dates:
        # Get transactions up to this date
        transactions_to_date = fund_data[fund_data['date'] <= date].copy()
        
        # Get NAV for this date
        nav_on_date = benchmark_df[benchmark_df['date'] <= date]['price'].iloc[-1]
        
        # Calculate cumulative units up to this date
        current_units = transactions_to_date['cumulative_units'].iloc[-1] if not transactions_to_date.empty else 0
        
        # Calculate current value
        current_value = current_units * fund_nav
        
        fund_growth.append({
            'date': date,
            'current_value': current_value
        })
    
    fund_growth_df = pd.DataFrame(fund_growth)
    
    # For benchmark calculation - calculate equivalent benchmark investment
    benchmark_growth = []
    initial_benchmark_value = benchmark_df.iloc[0]['price']
    
    # Get all investment dates and amounts
    investments = fund_data[fund_data['transaction_type'].isin(['invest', 'switch_in'])].copy()
    
    if not investments.empty:
        for _, row in investments.iterrows():
            # Get benchmark value on investment date
            benchmark_value_on_date = benchmark_df[benchmark_df['date'] <= row['date']]['price'].iloc[-1]
            
            # Calculate benchmark units bought
            benchmark_units = row['amount'] / benchmark_value_on_date
            
            # For each subsequent date, calculate benchmark value
            subsequent_dates = benchmark_df[benchmark_df['date'] >= row['date']]
            
            for _, b_row in subsequent_dates.iterrows():
                benchmark_growth.append({
                    'date': b_row['date'],
                    'investment_date': row['date'],
                    'benchmark_units': benchmark_units,
                    'benchmark_value': benchmark_units * b_row['price']
                })
    
    if benchmark_growth:
        benchmark_growth_df = pd.DataFrame(benchmark_growth)
        # Sum all benchmark investments for each date
        benchmark_growth_df = benchmark_growth_df.groupby('date')['benchmark_value'].sum().reset_index()
    else:
        benchmark_growth_df = pd.DataFrame(columns=['date', 'benchmark_value'])
    
    return fund_growth_df, benchmark_growth_df

def main():
    st.set_page_config(page_title="Fund vs Benchmark Comparison", layout="wide")
    st.title("Fund vs Benchmark Comparison")
    
    # Load data
    portfolio_df = get_portfolio_data()
    latest_nav = get_latest_nav()
    benchmark_df = get_benchmark_data()
    
    if portfolio_df.empty or benchmark_df.empty or latest_nav.empty:
        st.warning("No data found. Please ensure portfolio and benchmark data are available.")
        return
    
    # Convert dates
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
    
    # Calculate cumulative units
    portfolio_df = calculate_units(portfolio_df)
    
    # Create fund selector
    available_funds = portfolio_df['scheme_name'].unique()
    selected_fund = st.selectbox("Select Fund", available_funds)
    
    # Calculate growth
    fund_growth, benchmark_growth = calculate_growth(portfolio_df, benchmark_df, latest_nav, selected_fund)
    
    if fund_growth.empty or benchmark_growth.empty:
        st.warning("No growth data available for the selected fund.")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=fund_growth, x='date', y='current_value', label=selected_fund, ax=ax)
    sns.lineplot(data=benchmark_growth, x='date', y='benchmark_value', label='Nifty50 TRI', ax=ax)
    
    plt.title(f"{selected_fund} vs Nifty50 TRI")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Display plot
    st.pyplot(fig)
    
    # Calculate metrics
    # Total investment is sum of all 'invest' and 'switch_in' transactions
    total_investment = portfolio_df[
        (portfolio_df['scheme_name'] == selected_fund) & 
        (portfolio_df['transaction_type'].isin(['invest', 'switch_in']))
    ]['amount'].sum()
    
    current_fund_value = fund_growth['current_value'].iloc[-1]
    current_benchmark_value = benchmark_growth['benchmark_value'].iloc[-1]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
    
    with col2:
        st.metric("Current Fund Value", f"₹{current_fund_value:,.2f}")
    
    with col3:
        st.metric("Nifty50 TRI Value", f"₹{current_benchmark_value:,.2f}")
    
    # Calculate and display returns
    fund_returns = ((current_fund_value - total_investment) / total_investment) * 100
    benchmark_returns = ((current_benchmark_value - total_investment) / total_investment) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fund Returns", f"{fund_returns:.2f}%")
    with col2:
        st.metric("Nifty50 TRI Returns", f"{benchmark_returns:.2f}%")

if __name__ == "__main__":
    main()