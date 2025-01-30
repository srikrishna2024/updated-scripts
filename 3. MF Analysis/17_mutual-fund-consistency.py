import streamlit as st
import pandas as pd
import numpy as np
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

def get_categories():
    """Fetch unique mutual fund categories"""
    with connect_to_db() as conn:
        query = """
        SELECT DISTINCT scheme_category
        FROM mutual_fund_master_data
        ORDER BY scheme_category;
        """
        df = pd.read_sql(query, conn)
        return df['scheme_category'].tolist()

def get_funds_in_category(category):
    """Fetch funds in the selected category"""
    with connect_to_db() as conn:
        query = """
        SELECT code, scheme_name
        FROM mutual_fund_master_data
        WHERE scheme_category = %s
        ORDER BY scheme_name;
        """
        df = pd.read_sql(query, conn, params=(category,))
        return df

def get_fund_nav_data(code):
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
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['value'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def calculate_benchmark_rolling_returns(df, period_years):
    """Calculate rolling CAGR for benchmark data"""
    df = df.copy()
    df.set_index('date', inplace=True)
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['price'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def calculate_consistency_score(fund_rolling_cagr, benchmark_rolling_cagr):
    """Calculate consistency score for a fund compared to the benchmark"""
    if len(fund_rolling_cagr) == 0 or len(benchmark_rolling_cagr) == 0:
        return 0.0  # No data to compare
    # Align the dates of fund and benchmark rolling returns
    aligned_data = pd.DataFrame({
        'fund': fund_rolling_cagr,
        'benchmark': benchmark_rolling_cagr
    }).dropna()
    # Calculate outperformance
    outperformance = (aligned_data['fund'] > aligned_data['benchmark']).sum()
    total_periods = len(aligned_data)
    # Consistency score
    return round(outperformance / total_periods, 1) if total_periods > 0 else 0.0

def main():
    st.set_page_config(page_title='Mutual Fund Consistency Score Tool', layout='wide')
    st.title('Mutual Fund Consistency Score Tool')

    # Sidebar for inputs
    st.sidebar.header('Select Mutual Fund Category')
    categories = get_categories()
    selected_category = st.sidebar.selectbox('Select Category', categories)

    analyze_button = st.sidebar.button('Analyze Category')

    if analyze_button:
        with st.spinner('Analyzing funds and benchmark rolling returns...'):
            # Fetch funds in the selected category
            funds_in_category = get_funds_in_category(selected_category)
            if funds_in_category.empty:
                st.error(f'No funds found in the selected category: {selected_category}.')
                return

            # Fetch historical benchmark data
            benchmark_data = get_benchmark_data()
            if benchmark_data.empty:
                st.error('No benchmark data found.')
                return

            # Define rolling periods (1, 3, 5, 7, and 10 years)
            periods = {
                '1 Year': 1,
                '3 Years': 3,
                '5 Years': 5,
                '7 Years': 7,
                '10 Years': 10
            }

            # Initialize a DataFrame to store consistency scores
            consistency_scores = pd.DataFrame(columns=['Fund', '1 Year', '3 Years', '5 Years', '7 Years', '10 Years'])

            # Calculate rolling returns and consistency scores for each fund
            for _, fund in funds_in_category.iterrows():
                fund_code = fund['code']
                fund_name = fund['scheme_name']
                fund_nav_data = get_fund_nav_data(fund_code)
                if fund_nav_data.empty:
                    st.warning(f'No NAV data found for fund: {fund_name}. Skipping...')
                    continue

                # Calculate consistency scores for each period
                fund_scores = {'Fund': fund_name}
                for period_name, period_years in periods.items():
                    # Calculate rolling returns for the fund
                    fund_rolling_cagr = calculate_rolling_returns(
                        fund_nav_data.rename(columns={'nav': 'date', 'value': 'value'}),
                        period_years
                    )
                    # Calculate rolling returns for the benchmark
                    benchmark_rolling_cagr = calculate_benchmark_rolling_returns(benchmark_data, period_years)
                    # Calculate consistency score
                    consistency_score = calculate_consistency_score(fund_rolling_cagr, benchmark_rolling_cagr)
                    fund_scores[period_name] = consistency_score

                # Add fund's consistency scores to the DataFrame using pd.concat
                consistency_scores = pd.concat(
                    [consistency_scores, pd.DataFrame([fund_scores])],
                    ignore_index=True
                )

            # Display consistency scores
            st.subheader(f'Consistency Scores for Funds in {selected_category}')
            st.dataframe(consistency_scores)

            # Display interpretation information
            st.subheader('How to Interpret Consistency Scores')
            st.write("""
            The **Consistency Score** measures how often a fund outperforms the benchmark over a given period. 
            It is calculated as the ratio of the number of periods the fund outperformed the benchmark to the total number of periods.

            - **Score of 1.0**: The fund outperformed the benchmark in **100%** of the periods.
            - **Score of 0.5**: The fund outperformed the benchmark in **50%** of the periods.
            - **Score of 0.0**: The fund **never** outperformed the benchmark.

            A higher consistency score indicates that the fund has been more consistent in outperforming the benchmark over time.
            """)

if __name__ == "__main__":
    main()