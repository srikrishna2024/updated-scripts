import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

def get_fund_list():
    """Fetch list of mutual funds (scheme name and code)"""
    with connect_to_db() as conn:
        query = """
        SELECT DISTINCT code, scheme_name
        FROM mutual_fund_master_data
        ORDER BY scheme_name;
        """
        df = pd.read_sql(query, conn)
        return df

def get_fund_nav_data(code, scheme_name):
    """Fetch historical NAV data for a specific fund"""
    with connect_to_db() as conn:
        query = """
        SELECT nav, value
        FROM mutual_fund_nav
        WHERE code = %s
        ORDER BY nav;
        """
        df = pd.read_sql(query, conn, params=(code,))
        df['nav'] = pd.to_datetime(df['nav'])
        df['value'] = df['value'].astype(float)
        # Store the scheme name as a column instead of DataFrame attribute
        df['scheme_name'] = scheme_name
        return df

def get_benchmark_data():
    """Fetch historical benchmark data"""
    with connect_to_db() as conn:
        query = """
        SELECT date, price
        FROM benchmark
        ORDER BY date;
        """
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        df['price'] = df['price'].astype(float)
        return df

def calculate_rolling_returns(df, period_years):
    """Calculate rolling CAGR for a given period"""
    df = df.copy()
    df.set_index('date', inplace=True)
    
    # Drop duplicate dates if any
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['value'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def calculate_benchmark_rolling_returns(df, period_years):
    """Calculate rolling CAGR for benchmark data"""
    df = df.copy()
    df.set_index('date', inplace=True)
    
    # Drop duplicate dates if any
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['price'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def plot_rolling_returns(fund_data, benchmark_data, period_name, period_years, scheme_name):
    """Plot rolling returns for a given period"""
    # Calculate rolling returns for the fund
    fund_rolling_cagr = calculate_rolling_returns(
        fund_data.rename(columns={'nav': 'date', 'value': 'value'}),
        period_years
    )
    # Calculate rolling returns for the benchmark
    benchmark_rolling_cagr = calculate_benchmark_rolling_returns(
        benchmark_data,
        period_years
    )

    # Plot rolling returns
    plt.figure(figsize=(10, 4))
    plt.plot(fund_rolling_cagr.index, fund_rolling_cagr * 100, label=f'{scheme_name} Rolling CAGR', color='blue')
    plt.plot(benchmark_rolling_cagr.index, benchmark_rolling_cagr * 100, label='Benchmark Rolling CAGR', color='orange')
    plt.xlabel('Start Date of Rolling Period')
    plt.ylabel('CAGR (%)')
    plt.title(f'{period_name} Rolling Returns: {scheme_name} vs Benchmark')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.set_page_config(page_title='Mutual Fund Rolling Returns Analyzer', layout='wide')
    st.title('Mutual Fund Rolling Returns Analyzer')

    # Create tabs
    tab1, tab2 = st.tabs(["Single Fund vs Benchmark", "Compare 3 Funds vs Benchmark"])

    with tab1:
        st.header("Single Fund vs Benchmark")
        # Sidebar for inputs
        st.sidebar.header('Select Fund')
        fund_list = get_fund_list()
        selected_fund = st.sidebar.selectbox('Select Mutual Fund', fund_list['scheme_name'], key='single_fund')
        selected_code = fund_list[fund_list['scheme_name'] == selected_fund]['code'].values[0]

        analyze_button = st.sidebar.button('Analyze Fund', key='analyze_single')

        if analyze_button:
            with st.spinner('Analyzing fund and benchmark rolling returns...'):
                # Fetch historical NAV data for the selected fund
                fund_nav_data = get_fund_nav_data(selected_code, selected_fund)
                if fund_nav_data.empty:
                    st.error('No data found for the selected fund.')
                    return

                # Fetch historical benchmark data
                benchmark_data = get_benchmark_data()
                if benchmark_data.empty:
                    st.error('No benchmark data found.')
                    return

                # Define rolling periods (only 1, 3, 5, 7, and 10 years)
                periods = {
                    '1 Year': 1,
                    '3 Years': 3,
                    '5 Years': 5,
                    '7 Years': 7,
                    '10 Years': 10
                }

                # Calculate and plot rolling returns for each period
                for period_name, period_years in periods.items():
                    plot_rolling_returns(fund_nav_data, benchmark_data, period_name, period_years, selected_fund)

    with tab2:
        st.header("Compare 3 Funds vs Benchmark")
        # Sidebar for inputs
        st.sidebar.header('Select Funds')
        fund_list = get_fund_list()
        selected_funds = st.sidebar.multiselect(
            'Select up to 3 Mutual Funds', 
            fund_list['scheme_name'], 
            default=None, 
            key='multi_fund',
            max_selections=3
        )

        analyze_button = st.sidebar.button('Analyze Funds', key='analyze_multi')

        if analyze_button and len(selected_funds) > 0:
            with st.spinner('Analyzing funds and benchmark rolling returns...'):
                # Fetch historical benchmark data
                benchmark_data = get_benchmark_data()
                if benchmark_data.empty:
                    st.error('No benchmark data found.')
                    return

                # Define rolling periods (only 1, 3, 5, 7, and 10 years)
                periods = {
                    '1 Year': 1,
                    '3 Years': 3,
                    '5 Years': 5,
                    '7 Years': 7,
                    '10 Years': 10
                }

                # Fetch historical NAV data for the selected funds
                fund_data_list = []
                for fund_name in selected_funds:
                    fund_code = fund_list[fund_list['scheme_name'] == fund_name]['code'].values[0]
                    fund_nav_data = get_fund_nav_data(fund_code, fund_name)
                    if fund_nav_data.empty:
                        st.error(f'No data found for the selected fund: {fund_name}.')
                        return
                    fund_data_list.append((fund_nav_data, fund_name))

                # Calculate and plot rolling returns for each period
                for period_name, period_years in periods.items():
                    plt.figure(figsize=(10, 4))
                    # Plot benchmark rolling returns
                    benchmark_rolling_cagr = calculate_benchmark_rolling_returns(benchmark_data, period_years)
                    plt.plot(benchmark_rolling_cagr.index, benchmark_rolling_cagr * 100, label='Benchmark Rolling CAGR', color='orange')

                    # Plot rolling returns for each fund
                    for fund_data, fund_name in fund_data_list:
                        fund_rolling_cagr = calculate_rolling_returns(
                            fund_data.rename(columns={'nav': 'date', 'value': 'value'}),
                            period_years
                        )
                        plt.plot(fund_rolling_cagr.index, fund_rolling_cagr * 100, label=f'{fund_name} Rolling CAGR')

                    plt.xlabel('Start Date of Rolling Period')
                    plt.ylabel('CAGR (%)')
                    plt.title(f'{period_name} Rolling Returns: Funds vs Benchmark')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

if __name__ == "__main__":
    main()