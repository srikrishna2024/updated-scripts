import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg
from scipy.optimize import newton
import plotly.express as px

def format_indian_number(number):
    """
    Format a number in Indian style (with commas for lakhs, crores)
    Example: 10234567 becomes 1,02,34,567
    """
    if pd.isna(number):
        return "0"
    
    number = float(number)
    is_negative = number < 0
    number = abs(number)
    
    # Convert to string with 2 decimal places
    str_number = f"{number:,.2f}"
    
    # Split the decimal part
    parts = str_number.split('.')
    whole_part = parts[0].replace(',', '')
    decimal_part = parts[1] if len(parts) > 1 else '00'
    
    # Format the whole part in Indian style
    result = ""
    length = len(whole_part)
    
    # Handle numbers less than 1000
    if length <= 3:
        result = whole_part
    else:
        # Add the last 3 digits
        result = whole_part[-3:]
        # Add other digits in groups of 2
        remaining = whole_part[:-3]
        while remaining:
            result = (remaining[-2:] if len(remaining) >= 2 else remaining) + ',' + result
            remaining = remaining[:-2]
    
    # Add decimal part and negative sign if needed
    formatted = f"₹{'-' if is_negative else ''}{result}.{decimal_part}"
    return formatted

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
    """Retrieve all records from portfolio_data table"""
    with connect_to_db() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs from mutual_fund_nav table"""
    with connect_to_db() as conn:
        query = """
            SELECT code, value AS nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav) AS nav_date
                FROM mutual_fund_nav
                GROUP BY code
            )
        """
        return pd.read_sql(query, conn)

def get_goal_mappings():
    """Retrieve goal mappings from the goals table including both MF and debt investments"""
    with connect_to_db() as conn:
        query = """
            WITH mf_latest_values AS (
                SELECT g.goal_name, g.investment_type, g.scheme_name, g.scheme_code,
                       CASE 
                           WHEN g.is_manual_entry THEN g.current_value
                           ELSE COALESCE(p.units * n.value, 0)
                       END as current_value
                FROM goals g
                LEFT JOIN (
                    SELECT scheme_name, code,
                           SUM(CASE 
                               WHEN transaction_type = 'switch' THEN -units
                               WHEN transaction_type = 'redeem' THEN -units
                               ELSE units 
                           END) as units
                    FROM portfolio_data
                    GROUP BY scheme_name, code
                ) p ON g.scheme_code = p.code
                LEFT JOIN (
                    SELECT code, value
                    FROM mutual_fund_nav
                    WHERE (code, nav) IN (
                        SELECT code, MAX(nav)
                        FROM mutual_fund_nav
                        GROUP BY code
                    )
                ) n ON g.scheme_code = n.code
            )
            SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
            FROM mf_latest_values
            ORDER BY goal_name, investment_type, scheme_name
        """
        return pd.read_sql(query, conn)

def prepare_cashflows(df):
    """Prepare cashflow data from portfolio transactions"""
    df['cashflow'] = df.apply(lambda x: 
        -x['amount'] if x['transaction_type'] == 'invest'
        else x['amount'] if x['transaction_type'] == 'redeem'
        else (-x['amount'] if x['transaction_type'] == 'switch' else 0), 
        axis=1
    )
    return df

def xirr(transactions):
    """Calculate XIRR given a set of transactions"""
    if len(transactions) < 2:
        return None

    def xnpv(rate):
        first_date = pd.to_datetime(transactions['date'].min())
        days = [(pd.to_datetime(date) - first_date).days for date in transactions['date']]
        return sum([cf * (1 + rate) ** (-d/365.0) for cf, d in zip(transactions['cashflow'], days)])

    def xnpv_der(rate):
        first_date = pd.to_datetime(transactions['date'].min())
        days = [(pd.to_datetime(date) - first_date).days for date in transactions['date']]
        return sum([cf * (-d/365.0) * (1 + rate) ** (-d/365.0 - 1) 
                    for cf, d in zip(transactions['cashflow'], days)])

    try:
        return newton(xnpv, x0=0.1, fprime=xnpv_der, maxiter=1000)
    except:
        return None

def calculate_portfolio_weights(df, latest_nav, goal_mappings):
    """Calculate current portfolio weights for each scheme including debt investments"""
    # First calculate MF values
    mf_df = df.groupby('scheme_name').agg({
        'units': 'sum',
        'code': 'first'
    }).reset_index()

    mf_df = mf_df.merge(latest_nav, on='code', how='left')
    mf_df['current_value'] = mf_df['units'] * mf_df['nav_value']

    # Get debt investments from goal mappings, excluding any that are already in MF investments
    debt_investments = goal_mappings[
        (goal_mappings['investment_type'] == 'Debt') & 
        (~goal_mappings['scheme_name'].isin(mf_df['scheme_name']))
    ]
    
    # Combine MF and debt investments
    combined_df = pd.concat([
        mf_df[['scheme_name', 'current_value']],
        debt_investments[['scheme_name', 'current_value']]
    ])

    # Group by scheme_name to handle any remaining duplicates
    combined_df = combined_df.groupby('scheme_name')['current_value'].sum().reset_index()

    total_value = combined_df['current_value'].sum()
    combined_df['weight'] = (combined_df['current_value'] / total_value) * 100 if total_value > 0 else 0

    return combined_df

def calculate_xirr(df, latest_nav, goal_mappings):
    """Calculate XIRR for portfolio and individual schemes"""
    schemes = df['scheme_name'].unique()
    xirr_results = {}

    portfolio_growth = []  # To store portfolio value for each date

    for scheme in schemes:
        transactions = df[df['scheme_name'] == scheme].copy()
        # Add the current value as a final cash flow
        if not transactions.empty:
            latest_value = transactions['units'].sum() * latest_nav.loc[latest_nav['code'] == transactions['code'].iloc[0], 'nav_value'].values[0]
            transactions = pd.concat([
                transactions,
                pd.DataFrame({'date': [datetime.now()], 'cashflow': [latest_value]})
            ])
            rate = xirr(transactions)
            xirr_results[scheme] = round(rate * 100, 1) if rate is not None else 0

    # Calculate portfolio growth and overall XIRR
    unique_dates = df['date'].sort_values().unique()

    for date in unique_dates:
        transactions_up_to_date = df[df['date'] <= date].copy()
        transactions_up_to_date = transactions_up_to_date.merge(latest_nav, on='code', how='left')
        transactions_up_to_date['current_value'] = transactions_up_to_date['units'] * transactions_up_to_date['nav_value']
        
        # Include debt investments in the portfolio value
        debt_value_on_date = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
        total_value_on_date = transactions_up_to_date['current_value'].sum() + debt_value_on_date
        
        portfolio_growth.append({'date': date, 'value': total_value_on_date})

    # Calculate overall portfolio XIRR
    total_transactions = df.copy()
    if not total_transactions.empty:
        total_transactions = total_transactions.merge(latest_nav, on='code', how='left')
        total_transactions['current_value'] = total_transactions['units'] * total_transactions['nav_value']
        
        # Include debt investments in the final portfolio value
        debt_final_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
        portfolio_final_value = pd.DataFrame({
            'date': [datetime.now()],
            'cashflow': [total_transactions['current_value'].sum() + debt_final_value]
        })
        
        total_cashflow = total_transactions[['date', 'cashflow']]
        total_transactions = pd.concat([total_cashflow, portfolio_final_value])
        portfolio_xirr = xirr(total_transactions)
        xirr_results['Portfolio'] = round(portfolio_xirr * 100, 1) if portfolio_xirr is not None else 0

    # Calculate Mutual Fund Portfolio XIRR (excluding debt investments)
    mf_transactions = df.copy()
    if not mf_transactions.empty:
        mf_transactions = mf_transactions.merge(latest_nav, on='code', how='left')
        mf_transactions['current_value'] = mf_transactions['units'] * mf_transactions['nav_value']
        
        mf_final_value = pd.DataFrame({
            'date': [datetime.now()],
            'cashflow': [mf_transactions['current_value'].sum()]
        })
        
        mf_cashflow = mf_transactions[['date', 'cashflow']]
        mf_transactions = pd.concat([mf_cashflow, mf_final_value])
        mf_xirr = xirr(mf_transactions)
        xirr_results['Mutual Fund Portfolio'] = round(mf_xirr * 100, 1) if mf_xirr is not None else 0

    return xirr_results, pd.DataFrame(portfolio_growth)

def main():
    st.set_page_config(page_title="Portfolio Analysis", layout="wide")
    st.title("Portfolio Analysis Dashboard")

    df = get_portfolio_data()
    latest_nav = get_latest_nav()
    goal_mappings = get_goal_mappings()

    if df.empty or latest_nav.empty:
        st.warning("No data found. Please ensure portfolio data and NAV data are available.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = prepare_cashflows(df)

    # Calculate XIRR and Portfolio Growth
    xirr_results, portfolio_growth_df = calculate_xirr(df, latest_nav, goal_mappings)

    # Calculate portfolio weights including debt investments
    weights_df = calculate_portfolio_weights(df, latest_nav, goal_mappings)

    # Calculate equity and debt values
    equity_value = weights_df[weights_df['scheme_name'].isin(df['scheme_name'].unique())]['current_value'].sum()
    debt_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
    total_portfolio_value = equity_value + debt_value

    equity_percent = (equity_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
    debt_percent = (debt_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

    # Calculate total invested amount
    mf_invested = df[df['transaction_type'] == 'invest']['amount'].sum()
    debt_invested = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
    total_invested = mf_invested + debt_invested

    # Display Overall Portfolio Metrics - 2 metrics per row
    st.subheader("Overall Portfolio Metrics")
    
    # Row 1: Mutual Fund Portfolio XIRR and Current Portfolio Value
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mutual Fund Portfolio XIRR", f"{xirr_results['Mutual Fund Portfolio']:.1f}%")
    with col2:
        st.metric("Current Portfolio Value ( including EPF and PPF )", format_indian_number(total_portfolio_value))
    
    # Row 2: Equity and Debt Values
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Equity Mutual Funds", f"{format_indian_number(equity_value)} ({equity_percent:.1f}%)")
    with col4:
        st.metric("Debt Funds along with EPF and PPF", f"{format_indian_number(debt_value)} ({debt_percent:.1f}%)")
    
    # Row 3: Total Invested Amount (centered in one column)
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Total Invested Amount", format_indian_number(total_invested))

    # Display Individual Scheme Metrics
    st.subheader("Individual Fund Metrics")
    fund_metrics = weights_df[['scheme_name', 'current_value', 'weight']].copy()
    fund_metrics['Current Value'] = fund_metrics['current_value'].apply(format_indian_number)
    fund_metrics['Weight (%)'] = fund_metrics['weight'].round(2)
    fund_metrics['XIRR (%)'] = fund_metrics['scheme_name'].map(xirr_results)
    
    # Display formatted columns
    display_metrics = fund_metrics[['scheme_name', 'Current Value', 'Weight (%)', 'XIRR (%)']]
    display_metrics.columns = ['Scheme Name', 'Current Value', 'Weight (%)', 'XIRR (%)']
    st.dataframe(display_metrics)

    # Display Portfolio Growth Over Time
    st.subheader("Portfolio Growth Over Time")
    growth_chart_df = portfolio_growth_df.copy()
    growth_chart_df['value'] = growth_chart_df['value'].round(2)
    st.line_chart(growth_chart_df.rename(columns={'value': 'Portfolio Value'}).set_index('date'))

    # Display Goal-wise Equity and Debt Split
    if not goal_mappings.empty:
        st.subheader("Goal-wise Equity and Debt Split")
        
        goals = goal_mappings['goal_name'].unique()
        num_rows = (len(goals) + 1) // 2
        
        for row_idx in range(num_rows):
            col1, col2 = st.columns(2)
            
            with col1:
                if row_idx * 2 < len(goals):
                    goal = goals[row_idx * 2]
                    goal_data = goal_mappings[goal_mappings['goal_name'] == goal]
                    equity_value = goal_data[goal_data['investment_type'] == 'Equity']['current_value'].sum()
                    debt_value = goal_data[goal_data['investment_type'] == 'Debt']['current_value'].sum()
                    total_value = equity_value + debt_value

                    if total_value > 0:
                        equity_percent = (equity_value / total_value) * 100
                        debt_percent = (debt_value / total_value) * 100

                        fig = px.pie(
                            values=[equity_value, debt_value],
                            names=['Equity', 'Debt'],
                            title=f"{goal} - Equity vs Debt Split",
                            height=300
                        )
                        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Total Value:** {format_indian_number(total_value)}")
                        st.write(f"**Equity:** {equity_percent:.1f}% ({format_indian_number(equity_value)})")
                        st.write(f"**Debt:** {debt_percent:.1f}% ({format_indian_number(debt_value)})")

            with col2:
                if row_idx * 2 + 1 < len(goals):
                    goal = goals[row_idx * 2 + 1]
                    goal_data = goal_mappings[goal_mappings['goal_name'] == goal]
                    equity_value = goal_data[goal_data['investment_type'] == 'Equity']['current_value'].sum()
                    debt_value = goal_data[goal_data['investment_type'] == 'Debt']['current_value'].sum()
                    total_value = equity_value + debt_value

                    if total_value > 0:
                        equity_percent = (equity_value / total_value) * 100
                        debt_percent = (debt_value / total_value) * 100

                        fig = px.pie(
                            values=[equity_value, debt_value],
                            names=['Equity', 'Debt'],
                            title=f"{goal} - Equity vs Debt Split",
                            height=300
                        )
                        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Total Value:** {format_indian_number(total_value)}")
                        st.write(f"**Equity:** {equity_percent:.1f}% ({format_indian_number(equity_value)})")
                        st.write(f"**Debt:** {debt_percent:.1f}% ({format_indian_number(debt_value)})")

if __name__ == "__main__":
    main()