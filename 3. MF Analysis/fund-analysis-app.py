import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import newton

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

def get_fund_list():
    """Get list of all mutual funds from the database"""
    with connect_to_db() as conn:
        query = """
            SELECT DISTINCT code, scheme_name
            FROM mutual_fund_nav
            ORDER BY scheme_name
        """
        return pd.read_sql(query, conn)

def get_fund_nav_history(fund_code):
    """Get NAV history for a specific fund"""
    with connect_to_db() as conn:
        query = """
            SELECT code, scheme_name, nav as date, value
            FROM mutual_fund_nav
            WHERE code = %s
            ORDER BY nav
        """
        return pd.read_sql(query, conn, params=(fund_code,))

def get_fund_earliest_date(fund_code):
    """Get the earliest available date for a fund"""
    with connect_to_db() as conn:
        query = """
            SELECT MIN(nav) as earliest_date
            FROM mutual_fund_nav
            WHERE code = %s
        """
        result = pd.read_sql(query, conn, params=(fund_code,))
        return pd.to_datetime(result['earliest_date'].iloc[0])

def calculate_investment_growth(nav_history, investment_amount, start_date):
    """Calculate the growth of an investment from start_date to the present"""
    # Ensure nav_history is sorted by date
    nav_history = nav_history.sort_values('date')
    
    # Find NAV on or after the start date
    start_nav_data = nav_history[nav_history['date'] >= start_date].iloc[0]
    start_nav = start_nav_data['value']
    actual_start_date = pd.to_datetime(start_nav_data['date'])
    
    # Find the latest NAV
    end_nav_data = nav_history.iloc[-1]
    end_nav = end_nav_data['value']
    end_date = pd.to_datetime(end_nav_data['date'])
    
    # Calculate units purchased
    units = investment_amount / start_nav
    
    # Calculate current value
    current_value = units * end_nav
    
    # Calculate growth percentage
    growth_percentage = ((current_value - investment_amount) / investment_amount) * 100
    
    # Create a daily NAV timeseries for plotting
    investment_timeseries = nav_history[nav_history['date'] >= actual_start_date].copy()
    investment_timeseries['units'] = units
    investment_timeseries['investment_value'] = investment_timeseries['value'] * units
    
    # Calculate daily returns
    investment_timeseries['daily_return'] = investment_timeseries['value'].pct_change() * 100
    
    # Add XIRR calculation
    cashflows = pd.DataFrame([
        {'date': actual_start_date, 'cashflow': -investment_amount},
        {'date': end_date, 'cashflow': current_value}
    ])
    
    xirr_rate = calculate_xirr(cashflows)
    
    # Identify high-impact periods
    high_impact_periods = identify_high_impact_periods(investment_timeseries)
    
    return {
        'start_date': actual_start_date,
        'end_date': end_date,
        'investment_amount': investment_amount,
        'start_nav': start_nav,
        'end_nav': end_nav,
        'units': units,
        'current_value': current_value,
        'growth_percentage': growth_percentage,
        'investment_timeseries': investment_timeseries,
        'xirr': xirr_rate * 100,  # Convert to percentage
        'high_impact_periods': high_impact_periods
    }

def identify_high_impact_periods(timeseries, window_size=30, threshold_percentile=90):
    """
    Identify periods of significant market movement that had high impact on returns
    
    Parameters:
    - timeseries: DataFrame containing NAV and investment values
    - window_size: Size of the rolling window (in days) to look for impact periods
    - threshold_percentile: Percentile threshold to consider a period as high impact
    
    Returns:
    - List of dictionaries containing high impact periods info
    """
    if len(timeseries) < window_size + 1:
        return []
    
    # Calculate rolling returns for different windows
    windows = [window_size, 60, 90]  # 30, 60, 90 days windows
    
    high_impact_periods = []
    
    for w in windows:
        if len(timeseries) < w + 1:
            continue
            
        # Calculate rolling return for each window
        timeseries[f'rolling_{w}d_return'] = (
            timeseries['value'].pct_change(periods=w) * 100
        )
        
        # Calculate rolling volatility 
        timeseries[f'rolling_{w}d_volatility'] = (
            timeseries['daily_return'].rolling(window=w).std()
        )
        
        # Calculate impact score (magnitude of return / time period)
        timeseries[f'impact_score_{w}d'] = (
            timeseries[f'rolling_{w}d_return'].abs() / w
        )
        
        # Identify high impact periods
        if not timeseries[f'impact_score_{w}d'].dropna().empty:
            threshold = np.percentile(
                timeseries[f'impact_score_{w}d'].dropna(), 
                threshold_percentile
            )
            
            # Find periods above threshold
            high_impact_mask = timeseries[f'impact_score_{w}d'] > threshold
            
            if high_impact_mask.sum() > 0:
                # Convert index to list to avoid NumPy integer issues
                high_impact_indices = timeseries[high_impact_mask].index.tolist()
                
                # Get the start of each period
                for i in range(len(high_impact_indices)):
                    idx = high_impact_indices[i]
                    # Convert to datetime objects to ensure proper date arithmetic
                    current_day = pd.to_datetime(timeseries.loc[idx, 'date'])
                    prev_day = pd.to_datetime(timeseries.loc[high_impact_indices[i-1], 'date']) if i > 0 else None
                    
                    # Check if this is a new period or continue existing one
                    is_new_period = (i == 0) or (prev_day is None) or ((current_day - prev_day).days > w/2)
                    
                    if is_new_period:  # New period if gap > half window
                        start_idx = max(0, timeseries.index.get_loc(idx) - w + 1)
                        end_idx = timeseries.index.get_loc(idx)
                        
                        start_date = pd.to_datetime(timeseries.iloc[start_idx]['date'])
                        end_date = pd.to_datetime(timeseries.iloc[end_idx]['date'])
                        
                        period_return = timeseries.iloc[end_idx]['value'] / timeseries.iloc[start_idx]['value'] - 1
                        
                        # Ensure date objects for correct date arithmetic
                        date_diff = max(1, (end_date - start_date).days)
                        
                        # Only add if the return is significant
                        if abs(period_return) > 0.05:  # 5% minimum threshold
                            high_impact_periods.append({
                                'start_date': start_date,
                                'end_date': end_date,
                                'period_days': date_diff,
                                'return': period_return * 100,
                                'annualized_return': ((1 + period_return) ** (365 / date_diff) - 1) * 100,
                                'volatility': timeseries.iloc[end_idx][f'rolling_{w}d_volatility']
                            })
    
    # Remove overlapping periods, keeping the ones with highest impact
    if high_impact_periods:
        high_impact_periods.sort(key=lambda x: abs(x['return']), reverse=True)
        
        # Filter out overlapping periods
        filtered_periods = []
        for period in high_impact_periods:
            overlap = False
            for filtered in filtered_periods:
                # Check if periods overlap significantly
                # Ensure proper datetime objects
                p_start = pd.to_datetime(period['start_date'])
                p_end = pd.to_datetime(period['end_date'])
                f_start = pd.to_datetime(filtered['start_date'])
                f_end = pd.to_datetime(filtered['end_date'])
                
                if (max(p_start, f_start) <= min(p_end, f_end)):
                    days_overlap = (min(p_end, f_end) - max(p_start, f_start)).days
                    if days_overlap > min(period['period_days'], filtered['period_days']) * 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered_periods.append(period)
        
        # Sort by date
        filtered_periods.sort(key=lambda x: pd.to_datetime(x['start_date']))
        return filtered_periods
    
    return []

def calculate_xirr(transactions):
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
        if (1 + rate) <= 0:
            return np.inf  # Return a large value to avoid invalid rates
        return sum([cf * (-d/365.0) * (1 + rate) ** (-d/365.0 - 1) 
                    for cf, d in zip(transactions['cashflow'], days)])

    try:
        return newton(xnpv, x0=0.1, fprime=xnpv_der, maxiter=1000)
    except:
        return 0  # Default to 0 in case of errors

def suggest_start_dates(earliest_date, latest_date):
    """Suggest three meaningful start dates for analysis"""
    # Ensure dates are datetime objects
    earliest_date = pd.to_datetime(earliest_date)
    latest_date = pd.to_datetime(latest_date)
    
    date_range = latest_date - earliest_date
    
    # Make sure we have at least 2 years of data to make meaningful suggestions
    if date_range.days < 730:
        # If less than 2 years of data, use simpler splits
        third_point = earliest_date + timedelta(days=date_range.days // 3)
        two_third_point = earliest_date + timedelta(days=(date_range.days * 2) // 3)
        
        return [
            earliest_date.date(),
            third_point.date(),
            two_third_point.date()
        ]
    else:
        # Suggested dates: 1 year ago, 3 years ago, 5 years ago or earliest
        one_year_ago = latest_date - timedelta(days=365)
        three_years_ago = latest_date - timedelta(days=1095)
        five_years_ago = latest_date - timedelta(days=1825)
        
        dates = [
            max(earliest_date, five_years_ago).date(),
            max(earliest_date, three_years_ago).date(),
            max(earliest_date, one_year_ago).date()
        ]
        
        # Make sure dates are unique and sorted
        return sorted(list(set(dates)))

def main():
    st.set_page_config(page_title="Mutual Fund Investment Analysis", layout="wide")
    st.title("Mutual Fund Investment Analysis")
    
    # Load fund list from database
    fund_list = get_fund_list()
    
    if fund_list.empty:
        st.warning("No mutual fund data found in the database.")
        return
    
    # Create sidebar for input controls
    st.sidebar.header("Investment Parameters")
    
    # Fund selection
    selected_fund = st.sidebar.selectbox(
        "Select Mutual Fund",
        options=fund_list['scheme_name'].tolist(),
        index=0
    )
    
    # Get the code for the selected fund
    selected_fund_code = fund_list[fund_list['scheme_name'] == selected_fund]['code'].iloc[0]
    
    # Load NAV history for the selected fund
    nav_history = get_fund_nav_history(selected_fund_code)
    
    if nav_history.empty:
        st.warning(f"No NAV data found for {selected_fund}")
        return
    
    # Format dates
    nav_history['date'] = pd.to_datetime(nav_history['date'])
    earliest_date = nav_history['date'].min()
    latest_date = nav_history['date'].max()
    
    # Suggest 3 start dates
    suggested_dates = suggest_start_dates(earliest_date, latest_date)
    
    # Display date selectors for three different start dates
    st.sidebar.subheader("Select Start Dates for Comparison")
    
    start_date_1 = st.sidebar.date_input(
        "Start Date 1",
        value=suggested_dates[0],
        min_value=earliest_date.date(),
        max_value=latest_date.date()
    )
    
    start_date_2 = st.sidebar.date_input(
        "Start Date 2",
        value=suggested_dates[1] if len(suggested_dates) > 1 else suggested_dates[0],
        min_value=earliest_date.date(),
        max_value=latest_date.date()
    )
    
    start_date_3 = st.sidebar.date_input(
        "Start Date 3",
        value=suggested_dates[2] if len(suggested_dates) > 2 else suggested_dates[0],
        min_value=earliest_date.date(),
        max_value=latest_date.date()
    )
    
    # Investment amount
    investment_amount = st.sidebar.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("Analyze Investment", type="primary")
    
    if analyze_button:
        # Display the selected fund details
        st.header(f"Analysis for {selected_fund}")
        st.write(f"Data available from {earliest_date.date()} to {latest_date.date()}")
        
        # Calculate growth for each start date
        results = []
        for i, start_date in enumerate([start_date_1, start_date_2, start_date_3]):
            result = calculate_investment_growth(
                nav_history, 
                investment_amount, 
                pd.to_datetime(start_date)
            )
            results.append(result)
        
        # Display analysis results in a tabular format
        st.subheader("Investment Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, result in enumerate(results):
            with cols[i]:
                st.markdown(f"**Start Date: {result['start_date'].date()}**")
                st.metric(
                    label="Current Value", 
                    value=format_indian_number(result['current_value']),
                    delta=f"{result['growth_percentage']:.2f}%"
                )
                st.metric(
                    label="XIRR", 
                    value=f"{result['xirr']:.2f}%"
                )
                st.write(f"NAV: {result['start_nav']:.4f} → {result['end_nav']:.4f}")
                st.write(f"Units Purchased: {result['units']:.4f}")
        
        # Plot the investment growth over time
        st.subheader("Investment Value Over Time")
        
        # Create figure
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each line
        names = [f"From {results[i]['start_date'].date()}" for i in range(3)]
        
        for i, result in enumerate(results):
            timeseries = result['investment_timeseries']
            fig.add_trace(go.Scatter(
                x=timeseries['date'],
                y=timeseries['investment_value'],
                mode='lines',
                name=names[i],
                line=dict(color=colors[i], width=2)
            ))
        
        # Add the horizontal line for the investment amount
        fig.add_shape(
            type="line",
            x0=min(pd.to_datetime(r['start_date']) for r in results),
            y0=investment_amount,
            x1=latest_date,
            y1=investment_amount,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        # Add annotation for the investment amount
        fig.add_annotation(
            x=latest_date,
            y=investment_amount,
            text=f"Investment: {format_indian_number(investment_amount)}",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=20
        )
        
        # Highlight high-impact periods
        all_impact_periods = []
        
        for i, result in enumerate(results):
            impact_periods = result['high_impact_periods']
            for period in impact_periods:
                all_impact_periods.append({
                    'start_date': pd.to_datetime(period['start_date']),
                    'end_date': pd.to_datetime(period['end_date']), 
                    'return': period['return'],
                    'color': colors[i],
                    'investment_start': result['start_date'].date()
                })
        
        # Sort and deduplicate impact periods across all investments
        if all_impact_periods:
            all_impact_periods.sort(key=lambda x: (pd.to_datetime(x['start_date']), abs(x['return'])), reverse=True)
            
            # Keep only the most significant periods
            unique_periods = []
            for period in all_impact_periods:
                # Check if this period overlaps with any existing period
                overlaps = False
                for up in unique_periods:
                    # Ensure proper datetime objects
                    p_start = pd.to_datetime(period['start_date'])
                    p_end = pd.to_datetime(period['end_date'])
                    up_start = pd.to_datetime(up['start_date'])
                    up_end = pd.to_datetime(up['end_date'])
                    
                    # If periods overlap significantly
                    if (max(p_start, up_start) <= min(p_end, up_end)):
                        days_overlap = (min(p_end, up_end) - max(p_start, up_start)).days
                        total_days = (p_end - p_start).days
                        if days_overlap > total_days * 0.5:
                            overlaps = True
                            break
                
                if not overlaps:
                    unique_periods.append(period)
            
            # Get the top 5 most impactful periods for highlighting
            unique_periods.sort(key=lambda x: abs(x['return']), reverse=True)
            top_periods = unique_periods[:5]
            
            # Add highlighting for high impact periods
            for period in top_periods:
                # Add a semi-transparent rectangle to highlight the period
                fig.add_shape(
                    type="rect",
                    x0=pd.to_datetime(period['start_date']),
                    y0=0,
                    x1=pd.to_datetime(period['end_date']),
                    y1=max([r['investment_timeseries']['investment_value'].max() for r in results]) * 1.05,
                    fillcolor="rgba(255, 0, 0, 0.1)" if period['return'] < 0 else "rgba(0, 255, 0, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add annotation for the period
                fig.add_annotation(
                    x=pd.to_datetime(period['start_date']) + (pd.to_datetime(period['end_date']) - pd.to_datetime(period['start_date']))/2,
                    y=max([r['investment_timeseries']['investment_value'].max() for r in results]),
                    text=f"{period['return']:.1f}% in {(pd.to_datetime(period['end_date']) - pd.to_datetime(period['start_date'])).days} days",
                    showarrow=True,
                    arrowhead=1,
                    bgcolor="rgba(255,255,255,0.8)"
                )
        
        # Update layout
        fig.update_layout(
            title=f"Growth of {format_indian_number(investment_amount)} in {selected_fund}",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display high-impact periods table
        if all_impact_periods:
            st.subheader("High-Impact Market Periods")
            st.write("These periods significantly affected the investment performance:")
            
            impact_df = pd.DataFrame([
                {
                    'Start Date': pd.to_datetime(p['start_date']).date(),
                    'End Date': pd.to_datetime(p['end_date']).date(),
                    'Duration (Days)': (pd.to_datetime(p['end_date']) - pd.to_datetime(p['start_date'])).days,
                    'Return (%)': round(p['return'], 2),
                    'Annualized (%)': round(((1 + p['return']/100) ** (365 / max(1, (pd.to_datetime(p['end_date']) - pd.to_datetime(p['start_date'])).days)) - 1) * 100, 2),
                    'Investment Start': p['investment_start'],
                    'Impact': 'High Negative' if p['return'] < -15 else 
                             'Negative' if p['return'] < 0 else
                             'High Positive' if p['return'] > 15 else 'Positive'
                } 
                for p in unique_periods[:10]  # Show top 10 impact periods
            ])
            
            # Apply styling
            st.dataframe(
                impact_df.style.apply(
                    lambda x: ['background-color: rgba(255,0,0,0.1)' if x['Return (%)'] < 0 else 'background-color: rgba(0,255,0,0.1)' for _ in x],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("These periods represent market movements with outsized impact on returns. " +
                    "Negative periods are highlighted in red, positive in green.")
        
        # Calculate and display NAV growth chart
        st.subheader("Fund NAV History")
        
        # Create NAV chart
        nav_fig = px.line(
            nav_history, 
            x='date', 
            y='value',
            title=f"NAV History for {selected_fund}"
        )
        
        # Highlight the selected start dates
        for i, start_date in enumerate([start_date_1, start_date_2, start_date_3]):
            closest_date = nav_history[nav_history['date'] >= pd.to_datetime(start_date)].iloc[0]['date']
            closest_nav = nav_history[nav_history['date'] >= pd.to_datetime(start_date)].iloc[0]['value']
            
            nav_fig.add_trace(go.Scatter(
                x=[closest_date],
                y=[closest_nav],
                mode='markers',
                marker=dict(color=colors[i], size=10),
                name=f"Start Date {i+1}"
            ))
        
        # Highlight high-impact periods on NAV chart too
        if all_impact_periods:
            for period in unique_periods[:5]:  # Show top 5 impact periods
                # Add a semi-transparent rectangle to highlight the period
                nav_fig.add_shape(
                    type="rect",
                    x0=pd.to_datetime(period['start_date']),
                    y0=0,
                    x1=pd.to_datetime(period['end_date']),
                    y1=nav_history['value'].max() * 1.05,
                    fillcolor="rgba(255, 0, 0, 0.1)" if period['return'] < 0 else "rgba(0, 255, 0, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
        
        nav_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV (₹)",
            hovermode="x unified"
        )
        
        st.plotly_chart(nav_fig, use_container_width=True)
        
        # Display performance metrics table
        st.subheader("Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            'Start Date': [r['start_date'].date() for r in results],
            'End Date': [r['end_date'].date() for r in results],
            'Duration (Days)': [(r['end_date'] - r['start_date']).days for r in results],
            'Start NAV': [r['start_nav'] for r in results],
            'Current NAV': [r['end_nav'] for r in results],
            'NAV Growth (%)': [(r['end_nav'] / r['start_nav'] - 1) * 100 for r in results],
            'Investment': [investment_amount for r in results],
            'Current Value': [r['current_value'] for r in results],
            'Absolute Return (%)': [r['growth_percentage'] for r in results],
            'XIRR (%)': [r['xirr'] for r in results]
        })
        
        # Format the DataFrame for display
        display_df = metrics_df.copy()
        display_df['Investment'] = display_df['Investment'].apply(lambda x: format_indian_number(x))
        display_df['Current Value'] = display_df['Current Value'].apply(lambda x: format_indian_number(x))
        display_df['Start NAV'] = display_df['Start NAV'].round(4)
        display_df['Current NAV'] = display_df['Current NAV'].round(4)
        display_df['NAV Growth (%)'] = display_df['NAV Growth (%)'].round(2)
        display_df['Absolute Return (%)'] = display_df['Absolute Return (%)'].round(2)
        display_df['XIRR (%)'] = display_df['XIRR (%)'].round(2)
        
        st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()