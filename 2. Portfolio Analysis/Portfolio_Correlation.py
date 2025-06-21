import streamlit as st
import pandas as pd
import numpy as np
import psycopg
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def get_fund_nav_history(time_period='3 years'):
    """Retrieve NAV history for all funds in the portfolio"""
    with connect_to_db() as conn:
        if time_period == 'YTD':
            date_filter = "nav >= date_trunc('year', CURRENT_DATE)"
        elif time_period == '1 year':
            date_filter = "nav >= CURRENT_DATE - INTERVAL '1 year'"
        elif time_period == '3 years':
            date_filter = "nav >= CURRENT_DATE - INTERVAL '3 years'"
        elif time_period == '5 years':
            date_filter = "nav >= CURRENT_DATE - INTERVAL '5 years'"
        elif time_period == '10 years':
            date_filter = "nav >= CURRENT_DATE - INTERVAL '10 years'"
        else:
            date_filter = "nav >= CURRENT_DATE - INTERVAL '3 years'"
            
        query = f"""
            SELECT code, scheme_name, nav, value 
            FROM mutual_fund_nav
            WHERE {date_filter}
            ORDER BY code, nav
        """
        return pd.read_sql(query, conn)

def calculate_correlation_matrix(nav_data):
    """
    Calculate correlation matrix between funds based on NAV returns
    Returns both the correlation matrix and the processed returns data
    """
    # Pivot to get NAVs by date for each fund
    nav_pivot = nav_data.pivot(index='nav', columns='code', values='value')
    
    # Calculate daily returns
    returns = nav_pivot.pct_change().dropna()
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    return corr_matrix, returns

def calculate_rolling_correlation(returns_data, window_days=90):
    """
    Calculate rolling correlation between all fund pairs
    Returns a dictionary with correlation time series for each pair
    """
    rolling_correlations = {}
    fund_codes = returns_data.columns.tolist()
    
    # Calculate rolling correlation for each pair
    for i, fund1 in enumerate(fund_codes):
        for j, fund2 in enumerate(fund_codes):
            if i < j:  # Only calculate for unique pairs
                pair_name = f"{fund1} vs {fund2}"
                rolling_corr = returns_data[fund1].rolling(window=window_days).corr(returns_data[fund2])
                rolling_correlations[pair_name] = rolling_corr.dropna()
    
    return rolling_correlations

def get_top_fund_pairs(corr_matrix, portfolio_funds, top_n=5):
    """Get top N fund pairs by absolute correlation for focused analysis"""
    pairs = []
    fund_codes = corr_matrix.columns.tolist()
    
    for i, fund1 in enumerate(fund_codes):
        for j, fund2 in enumerate(fund_codes):
            if i < j:
                corr_val = corr_matrix.loc[fund1, fund2]
                fund1_name = portfolio_funds[portfolio_funds['code'] == fund1]['scheme_name'].iloc[0]
                fund2_name = portfolio_funds[portfolio_funds['code'] == fund2]['scheme_name'].iloc[0]
                
                pairs.append({
                    'fund1_code': fund1,
                    'fund2_code': fund2,
                    'fund1_name': fund1_name,
                    'fund2_name': fund2_name,
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val),
                    'pair_name': f"{fund1} vs {fund2}",
                    'display_name': f"{fund1_name[:20]}... vs {fund2_name[:20]}..."
                })
    
    # Sort by absolute correlation and return top N
    pairs_df = pd.DataFrame(pairs)
    return pairs_df.nlargest(top_n, 'abs_correlation')

def analyze_correlation_patterns(corr_matrix):
    """
    Analyze correlation patterns and provide recommendations
    Returns a dictionary with analysis results
    """
    # Get upper triangle without diagonal (k=1)
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_triangle = corr_matrix.where(upper_triangle_mask)
    upper_triangle_values = upper_triangle.values[np.triu_indices_from(corr_matrix, k=1)]
    
    analysis = {
        'high_correlation': [],
        'low_correlation': [],
        'negative_correlation': [],
        'average_correlation': np.mean(upper_triangle_values)
    }
    
    # Get fund pairs with correlation > 0.7
    high_corr = (upper_triangle > 0.7) & (upper_triangle != 0)
    for fund1, fund2 in zip(*np.where(high_corr)):
        analysis['high_correlation'].append((
            corr_matrix.index[fund1],
            corr_matrix.columns[fund2],
            corr_matrix.iloc[fund1, fund2]
        ))
    
    # Get fund pairs with correlation < 0.3
    low_corr = (upper_triangle < 0.3) & (upper_triangle > -0.3) & (upper_triangle != 0)
    for fund1, fund2 in zip(*np.where(low_corr)):
        analysis['low_correlation'].append((
            corr_matrix.index[fund1],
            corr_matrix.columns[fund2],
            corr_matrix.iloc[fund1, fund2]
        ))
    
    # Get fund pairs with negative correlation
    neg_corr = (upper_triangle < 0) & (upper_triangle != 0)
    for fund1, fund2 in zip(*np.where(neg_corr)):
        analysis['negative_correlation'].append((
            corr_matrix.index[fund1],
            corr_matrix.columns[fund2],
            corr_matrix.iloc[fund1, fund2]
        ))
    
    return analysis

def get_portfolio_weights():
    """Get current portfolio weights for each fund"""
    with connect_to_db() as conn:
        query = """
            SELECT code, SUM(value) as total_value
            FROM portfolio_data
            WHERE transaction_type IN ('invest', 'switch_in')
            GROUP BY code
        """
        weights = pd.read_sql(query, conn)
        weights['weight'] = weights['total_value'] / weights['total_value'].sum()
        return weights[['code', 'weight']]

def calculate_diversification_score(corr_matrix, weights):
    """
    Calculate portfolio diversification score
    Formula: 1 - (weighted average correlation)
    Higher score = better diversification (0-1 scale)
    """
    # Create weights matrix for pairwise products
    weights_matrix = np.outer(weights, weights)
    
    # Mask upper triangle (excluding diagonal)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    weighted_correlations = corr_matrix.values * weights_matrix * mask
    
    # Calculate weighted average correlation
    sum_weights = weights_matrix[mask].sum()
    if sum_weights > 0:
        weighted_avg_corr = weighted_correlations.sum() / sum_weights
    else:
        weighted_avg_corr = 0
    
    diversification_score = 1 - weighted_avg_corr
    
    return diversification_score, weighted_avg_corr

def calculate_risk_contributions(corr_matrix, returns_data, portfolio_funds):
    """
    Calculate risk contributions for each fund before and after diversification
    Returns a DataFrame with risk contribution metrics
    """
    # Calculate individual volatilities (standard deviations)
    volatilities = returns_data.std()
    fund_codes = returns_data.columns
    
    # Create covariance matrix
    cov_matrix = returns_data.cov()
    
    # Get portfolio weights (or assume equal weights if not available)
    try:
        weights_df = get_portfolio_weights()
        weights = weights_df.set_index('code').reindex(fund_codes)['weight'].values
        weights = np.nan_to_num(weights, nan=1/len(fund_codes))  # Replace NaN with equal weights
        weights = weights / weights.sum()  # Ensure weights sum to 1
    except:
        weights = np.ones(len(fund_codes)) / len(fund_codes)
    
    # Calculate portfolio variance and volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate Marginal Risk Contributions (MRC)
    mrc = np.dot(cov_matrix, weights) / portfolio_volatility
    
    # Calculate % Risk Contribution (RC%)
    rc_pct = (weights * mrc) / portfolio_volatility * 100
    
    # Calculate marginal/undiversified risk contribution (individual variance)
    marginal_risk = weights * volatilities**2
    marginal_risk_pct = marginal_risk / marginal_risk.sum() * 100
    
    # Calculate diversified risk contribution (component of portfolio variance)
    diversified_risk = weights * np.dot(cov_matrix, weights)
    diversified_risk_pct = diversified_risk / portfolio_variance * 100
    
    # Calculate diversification benefit (reduction in risk contribution)
    diversification_benefit = marginal_risk_pct - diversified_risk_pct
    
    # Calculate risk-weight ratio (RWR)
    portfolio_weights_pct = weights * 100
    risk_weight_ratio = diversified_risk_pct / portfolio_weights_pct
    
    # Calculate diversification score
    diversification_score, weighted_avg_corr = calculate_diversification_score(corr_matrix, weights)
    
    # Create DataFrame with results
    risk_data = []
    for i, code in enumerate(fund_codes):
        fund_name = portfolio_funds[portfolio_funds['code'] == code]['scheme_name'].iloc[0]
        risk_data.append({
            'Fund Code': code,
            'Fund Name': fund_name,
            'Portfolio Weight (%)': portfolio_weights_pct[i],
            'Volatility': volatilities[i],
            'Marginal Risk Contribution (MRC)': mrc[i],
            '% Risk Contribution (RC%)': rc_pct[i],
            'Marginal Risk Contribution (%)': marginal_risk_pct[i],
            'Diversified Risk Contribution (%)': diversified_risk_pct[i],
            'Diversification Benefit (%)': diversification_benefit[i],
            'Risk-Weight Ratio (RWR)': risk_weight_ratio[i]
        })
    
    risk_df = pd.DataFrame(risk_data).sort_values('% Risk Contribution (RC%)', ascending=False)
    
    # Add portfolio-level metrics to the risk_df attributes
    risk_df.attrs['diversification_score'] = diversification_score
    risk_df.attrs['weighted_avg_correlation'] = weighted_avg_corr
    risk_df.attrs['portfolio_volatility'] = portfolio_volatility
    
    return risk_df

def plot_risk_contributions(risk_df):
    """
    Create a stacked bar chart showing risk contribution before and after diversification
    """
    # Sort by diversified risk contribution
    risk_df = risk_df.sort_values('Diversified Risk Contribution (%)', ascending=True)
    
    fig = go.Figure()
    
    # Add marginal risk (undiversified)
    fig.add_trace(go.Bar(
        y=risk_df['Fund Name'],
        x=risk_df['Marginal Risk Contribution (%)'],
        name='Undiversified Risk',
        orientation='h',
        marker_color='#EF553B',  # Red
        hovertemplate='<b>%{y}</b><br>' +
                      'Undiversified Risk: %{x:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add diversification benefit (the difference)
    fig.add_trace(go.Bar(
        y=risk_df['Fund Name'],
        x=risk_df['Diversification Benefit (%)'],
        name='Diversification Benefit',
        orientation='h',
        marker_color='#00CC96',  # Green
        hovertemplate='<b>%{y}</b><br>' +
                      'Risk Reduced: %{x:.1f}%<br>' +
                      '<extra></extra>',
        base=risk_df['Diversified Risk Contribution (%)']
    ))
    
    # Add diversified risk
    fig.add_trace(go.Bar(
        y=risk_df['Fund Name'],
        x=risk_df['Diversified Risk Contribution (%)'],
        name='Diversified Risk',
        orientation='h',
        marker_color='#636EFA',  # Blue
        hovertemplate='<b>%{y}</b><br>' +
                      'Diversified Risk: %{x:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Calculate total diversification benefit
    total_benefit = risk_df['Diversification Benefit (%)'].sum()
    
    fig.update_layout(
        title=f"Risk Contribution Analysis (Total Diversification Benefit: {total_benefit:.1f}%)",
        barmode='stack',
        xaxis_title="Risk Contribution (% of Portfolio Variance)",
        yaxis_title="Fund",
        height=600,
        hovermode='y unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_risk_weight_ratio(risk_df):
    """
    Create a bar chart showing Risk-Weight Ratio (RWR) for each fund
    """
    # Sort by RWR
    risk_df = risk_df.sort_values('Risk-Weight Ratio (RWR)', ascending=False)
    
    # Create color scale based on RWR values
    colors = ['#EF553B' if rwr > 1.2 else '#00CC96' if rwr < 0.8 else '#636EFA' 
              for rwr in risk_df['Risk-Weight Ratio (RWR)']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_df['Fund Name'],
        y=risk_df['Risk-Weight Ratio (RWR)'],
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>' +
                      'RWR: %{y:.2f}<br>' +
                      'Weight: %{customdata[0]:.1f}%<br>' +
                      'Risk Contribution: %{customdata[1]:.1f}%<br>' +
                      '<extra></extra>',
        customdata=np.stack((risk_df['Portfolio Weight (%)'], 
                            risk_df['Diversified Risk Contribution (%)']), axis=-1)
    ))
    
    # Add reference lines without annotations (we'll add them separately)
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.add_hline(y=1.2, line_dash="dot", line_color="red")
    fig.add_hline(y=0.8, line_dash="dot", line_color="green")
    
    # Create custom annotations for reference lines
    annotations = [
        dict(x=1, y=1, xref='paper', yref='y',
             text="Balanced (1)", showarrow=False,
             xanchor='left', yanchor='bottom', font=dict(color='gray')),
        dict(x=1, y=1.2, xref='paper', yref='y',
             text="Overweight (1.2)", showarrow=False,
             xanchor='left', yanchor='bottom', font=dict(color='red')),
        dict(x=1, y=0.8, xref='paper', yref='y',
             text="Underweight (0.8)", showarrow=False,
             xanchor='left', yanchor='top', font=dict(color='green')),
        dict(x=0.5, y=-0.25, xref='paper', yref='paper',
             text="RWR = (Risk Contribution %) / (Portfolio Weight %). Values >1 indicate risk-heavy funds.",
             showarrow=False, font=dict(size=12))
    ]
    
    fig.update_layout(
        title="Risk-Weight Ratio (RWR) Analysis",
        xaxis_title="Fund",
        yaxis_title="Risk-Weight Ratio (RWR)",
        height=500,
        hovermode='x unified',
        xaxis=dict(tickangle=45),
        margin=dict(b=120),  # Increase bottom margin for annotation
        annotations=annotations
    )
    
    return fig

def get_portfolio_funds():
    """Get list of funds in the current portfolio"""
    with connect_to_db() as conn:
        query = """
            SELECT DISTINCT code, scheme_name 
            FROM portfolio_data
            WHERE transaction_type IN ('invest', 'switch_in')
        """
        return pd.read_sql(query, conn)

def get_fund_categories():
    """Get fund categories (assuming this exists in your database)"""
    with connect_to_db() as conn:
        query = """
            SELECT code, category 
            FROM fund_categories  -- This table would need to exist
        """
        try:
            return pd.read_sql(query, conn)
        except:
            # Return empty DataFrame if table doesn't exist
            return pd.DataFrame(columns=['code', 'category'])

def get_fund_category(fund_code, fund_categories):
    """Helper function to get fund category"""
    if fund_categories.empty:
        return "Unknown"
    cat = fund_categories[fund_categories['code'] == fund_code]['category']
    return cat.values[0] if len(cat) > 0 else "Unknown"

def generate_allocation_recommendations(risk_df, portfolio_funds, fund_categories):
    """
    Generate specific allocation recommendations based on risk contributions
    """
    recommendations = []
    
    # Sort funds by different metrics for recommendations
    high_rc_funds = risk_df.nlargest(3, '% Risk Contribution (RC%)')
    low_rc_funds = risk_df.nsmallest(3, '% Risk Contribution (RC%)')
    high_rwr_funds = risk_df[risk_df['Risk-Weight Ratio (RWR)'] > 1.2]
    low_rwr_funds = risk_df[risk_df['Risk-Weight Ratio (RWR)'] < 0.8]
    
    # Recommendation 1: Reduce highest risk contributors
    if not high_rc_funds.empty:
        rec = {
            'type': 'Reduce High Risk Contributors',
            'details': 'These funds contribute disproportionately to portfolio risk:',
            'funds': []
        }
        for _, row in high_rc_funds.iterrows():
            category = get_fund_category(row['Fund Code'], fund_categories)
            rec['funds'].append({
                'name': row['Fund Name'],
                'rc_pct': row['% Risk Contribution (RC%)'],
                'weight': row['Portfolio Weight (%)'],
                'category': category
            })
        recommendations.append(rec)
    
    # Recommendation 2: Increase low risk contributors
    if not low_rc_funds.empty:
        rec = {
            'type': 'Increase Low Risk Contributors',
            'details': 'These funds provide good diversification benefits:',
            'funds': []
        }
        for _, row in low_rc_funds.iterrows():
            category = get_fund_category(row['Fund Code'], fund_categories)
            rec['funds'].append({
                'name': row['Fund Name'],
                'rc_pct': row['% Risk Contribution (RC%)'],
                'weight': row['Portfolio Weight (%)'],
                'category': category
            })
        recommendations.append(rec)
    
    # Recommendation 3: Rebalance risk-heavy funds
    if not high_rwr_funds.empty:
        rec = {
            'type': 'Rebalance Risk-Heavy Funds',
            'details': 'These funds are contributing more risk than their allocation:',
            'funds': []
        }
        for _, row in high_rwr_funds.iterrows():
            category = get_fund_category(row['Fund Code'], fund_categories)
            rec['funds'].append({
                'name': row['Fund Name'],
                'rwr': row['Risk-Weight Ratio (RWR)'],
                'weight': row['Portfolio Weight (%)'],
                'category': category
            })
        recommendations.append(rec)
    
    return recommendations

def provide_allocation_recommendations(analysis, portfolio_funds, fund_categories):
    """
    Provide asset allocation recommendations based on correlation analysis
    and fund categories (if available)
    """
    recommendations = []
    
    # 1. For highly correlated funds
    if analysis['high_correlation']:
        rec = {
            'type': 'Reduce overlap',
            'details': 'The following fund pairs show high correlation (>0.7):',
            'fund_pairs': []
        }
        
        for fund1, fund2, corr in analysis['high_correlation']:
            # Try to get categories if available
            cat1 = fund_categories[fund_categories['code'] == fund1]['category'].values
            cat2 = fund_categories[fund_categories['code'] == fund2]['category'].values
            
            cat_info = ""
            if len(cat1) > 0 and len(cat2) > 0:
                cat_info = f" (Both {cat1[0]} funds)" if cat1[0] == cat2[0] else f" ({cat1[0]} vs {cat2[0]})"
            
            fund1_name = portfolio_funds[portfolio_funds['code'] == fund1]['scheme_name'].values[0]
            fund2_name = portfolio_funds[portfolio_funds['code'] == fund2]['scheme_name'].values[0]
            
            rec['fund_pairs'].append(
                f"{fund1_name} & {fund2_name}: {corr:.2f} correlation{cat_info}"
            )
        
        recommendations.append(rec)
    
    # 2. For negatively correlated funds
    if analysis['negative_correlation']:
        rec = {
            'type': 'Diversification opportunities',
            'details': 'These fund pairs show negative correlation, good for diversification:',
            'fund_pairs': []
        }
        
        for fund1, fund2, corr in analysis['negative_correlation']:
            # Try to get categories if available
            cat1 = fund_categories[fund_categories['code'] == fund1]['category'].values
            cat2 = fund_categories[fund_categories['code'] == fund2]['category'].values
            
            cat_info = ""
            if len(cat1) > 0 and len(cat2) > 0:
                cat_info = f" ({cat1[0]} vs {cat2[0]})"
            
            fund1_name = portfolio_funds[portfolio_funds['code'] == fund1]['scheme_name'].values[0]
            fund2_name = portfolio_funds[portfolio_funds['code'] == fund2]['scheme_name'].values[0]
            
            rec['fund_pairs'].append(
                f"{fund1_name} & {fund2_name}: {corr:.2f} correlation{cat_info}"
            )
        
        recommendations.append(rec)
    
    # 3. General recommendations based on average correlation
    if analysis['average_correlation'] > 0.5:
        recommendations.append({
            'type': 'High overall correlation',
            'details': f"Your portfolio has high average correlation ({analysis['average_correlation']:.2f}). Consider adding:",
            'suggestions': [
                "Funds from different categories (e.g., sectoral, international)",
                "Debt funds to balance equity risk",
                "Alternative investments (gold, REITs, etc.) if appropriate"
            ]
        })
    elif analysis['average_correlation'] < 0.2:
        recommendations.append({
            'type': 'Low overall correlation',
            'details': f"Your portfolio has low average correlation ({analysis['average_correlation']:.2f}). This is good for diversification.",
            'suggestions': [
                "Maintain this diversified allocation",
                "Monitor for any style drift in funds",
                "Rebalance periodically to maintain target allocations"
            ]
        })
    else:
        recommendations.append({
            'type': 'Moderate overall correlation',
            'details': f"Your portfolio has moderate average correlation ({analysis['average_correlation']:.2f}).",
            'suggestions': [
                "Consider adding some low-correlation funds for better diversification",
                "Review highly correlated pairs for potential overlap",
                "Ensure you have exposure to different market segments"
            ]
        })
    
    return recommendations

def plot_rolling_correlations(rolling_correlations, selected_pairs, portfolio_funds):
    """Create an interactive plot showing rolling correlations over time"""
    
    if not selected_pairs:
        st.warning("Please select at least one fund pair to display rolling correlations.")
        return
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(selected_pairs)]
    
    for i, pair in enumerate(selected_pairs):
        if pair in rolling_correlations:
            corr_series = rolling_correlations[pair]
            
            # Get display name for the pair
            fund1_code, fund2_code = pair.split(' vs ')
            fund1_name = portfolio_funds[portfolio_funds['code'] == fund1_code]['scheme_name'].iloc[0]
            fund2_name = portfolio_funds[portfolio_funds['code'] == fund2_code]['scheme_name'].iloc[0]
            display_name = f"{fund1_name[:15]}... vs {fund2_name[:15]}..."
            
            fig.add_trace(go.Scatter(
                x=corr_series.index,
                y=corr_series.values,
                mode='lines',
                name=display_name,
                line=dict(color=colors[i], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Correlation: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
    
    # Add horizontal reference lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Correlation (0.7)")
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Correlation (0.3)")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="No Correlation (0)")
    fig.add_hline(y=-0.3, line_dash="dash", line_color="green", 
                  annotation_text="Negative Correlation (-0.3)")
    
    fig.update_layout(
        title="Rolling Correlation Over Time",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    return fig

def main():
    st.set_page_config(page_title="Portfolio Correlation Analysis", layout="wide")
    st.title("🔄 Enhanced Mutual Fund Portfolio Correlation Analysis")
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Analysis Period",
        options=['YTD', '1 year', '3 years', '5 years', '10 years'],
        index=2  # Default to 3 years
    )
    
    # Rolling window selection
    rolling_window_options = {
        '1 Month (30 days)': 30,
        '3 Months (90 days)': 90,
        '6 Months (180 days)': 180,
        '1 Year (250 days)': 250
    }
    
    rolling_window_label = st.sidebar.selectbox(
        "Rolling Window for Correlation",
        options=list(rolling_window_options.keys()),
        index=1  # Default to 3 months
    )
    rolling_window = rolling_window_options[rolling_window_label]
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        options=['Risk Contribution Analysis', 'Rolling Correlation Over Time', 'Both'],
        index=0  # Default to Risk Contribution Analysis
    )
    
    # Load data
    with st.spinner("Loading fund NAV data..."):
        nav_data = get_fund_nav_history(time_period)
        portfolio_funds = get_portfolio_funds()
        fund_categories = get_fund_categories()
    
    if nav_data.empty:
        st.warning("No NAV data found. Please ensure mutual fund NAV data is available.")
        return
    
    if portfolio_funds.empty:
        st.warning("No portfolio funds found. Please ensure you have investment data.")
        return
    
    # Filter NAV data to only include funds in current portfolio
    portfolio_codes = portfolio_funds['code'].unique()
    nav_data = nav_data[nav_data['code'].isin(portfolio_codes)]
    
    if len(nav_data['code'].unique()) < 2:
        st.warning("Need at least 2 funds in portfolio to calculate correlation.")
        return
    
    # Calculate correlation matrix and returns
    with st.spinner("Calculating correlations..."):
        corr_matrix, returns_data = calculate_correlation_matrix(nav_data)
        analysis = analyze_correlation_patterns(corr_matrix)
        recommendations = provide_allocation_recommendations(analysis, portfolio_funds, fund_categories)
        risk_df = calculate_risk_contributions(corr_matrix, returns_data, portfolio_funds)
        allocation_recs = generate_allocation_recommendations(risk_df, portfolio_funds, fund_categories)
    
    # Display results based on selected analysis type
    if analysis_type in ['Risk Contribution Analysis', 'Both']:
        st.subheader("📊 Risk Contribution Analysis")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_benefit = risk_df['Diversification Benefit (%)'].sum()
            st.metric("Total Diversification Benefit", f"{total_benefit:.1f}%")
        
        with col2:
            avg_diversified_risk = risk_df['Diversified Risk Contribution (%)'].mean()
            st.metric("Avg Diversified Risk Contribution", f"{avg_diversified_risk:.1f}%")
        
        with col3:
            max_risk_fund = risk_df.iloc[0]['Fund Name']
            st.markdown(f"""
            <div style="font-size:14px;">
            <b>Highest Risk Contributor</b><br>
            {max_risk_fund}
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            score = risk_df.attrs.get('diversification_score', 0)
            weighted_corr = risk_df.attrs.get('weighted_avg_correlation', 0)
            portfolio_vol = risk_df.attrs.get('portfolio_volatility', 0)
            
            # Color based on score (red < 0.3, yellow < 0.6, green >= 0.6)
            if score < 0.3:
                delta_color = "inverse"
            elif score < 0.6:
                delta_color = "off"
            else:
                delta_color = "normal"
            
            st.metric(
                "Diversification Score", 
                f"{score:.2f}",
                help=f"1 - Weighted Avg Correlation ({weighted_corr:.2f}). Higher = better diversification"
            )
        
        # Display risk contribution chart
        st.plotly_chart(
            plot_risk_contributions(risk_df),
            use_container_width=True
        )
        
        # Display Risk-Weight Ratio plot
        st.plotly_chart(
            plot_risk_weight_ratio(risk_df),
            use_container_width=True
        )
        
        # Show detailed risk contribution table
        with st.expander("📋 Detailed Risk Contribution Data"):
            st.dataframe(risk_df.round(2), use_container_width=True)
        
        # Explanation of risk contribution
        with st.expander("ℹ️ Understanding Risk Contribution"):
            st.write("""
            **Risk Contribution Analysis** shows how each fund contributes to your portfolio's total risk (variance):
            
            - **Undiversified Risk (Red)**: How much risk the fund would contribute if it moved independently
            - **Diversified Risk (Blue)**: Actual risk contribution after accounting for correlations
            - **Diversification Benefit (Green)**: Risk reduction from diversification
            
            **Key Metrics:**
            
            - **Marginal Risk Contribution (MRC)**: How much portfolio risk changes with a small increase in this fund's weight
            - **% Risk Contribution (RC%)**: This fund's contribution to total portfolio risk
            - **Risk-Weight Ratio (RWR)**: Ratio of risk contribution to portfolio weight
            - **Diversification Score**: Overall portfolio diversification quality (0-1 scale)
            
            **Interpretation Guidelines:**
            
            - **RWR > 1.2** 🔴: Fund is overweight in risk (consider reducing allocation)
            - **RWR 0.8-1.2** 🔵: Balanced risk contribution
            - **RWR < 0.8** 🟢: Fund is providing good diversification
            - **Diversification Score:**
              - **0.8-1.0**: Excellent
              - **0.6-0.8**: Good
              - **0.4-0.6**: Moderate
              - **0.2-0.4**: Low
              - **0.0-0.2**: Very poor
            """)
    
    if analysis_type in ['Rolling Correlation Over Time', 'Both']:
        st.subheader("📈 Rolling Correlation Analysis")
        
        # Calculate rolling correlations
        with st.spinner("Calculating rolling correlations..."):
            rolling_correlations = calculate_rolling_correlation(returns_data, rolling_window)
        
        if rolling_correlations:
            # Get top fund pairs for selection
            top_pairs = get_top_fund_pairs(corr_matrix, portfolio_funds, top_n=10)
            
            # Let user select which pairs to display
            st.write("**Select fund pairs to display:**")
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_pairs = st.multiselect(
                    "Choose fund pairs (showing top 10 by absolute correlation):",
                    options=top_pairs['pair_name'].tolist(),
                    default=top_pairs['pair_name'].head(5).tolist(),  # Default to top 5
                    help="Select the fund pairs you want to analyze for rolling correlation"
                )
            
            with col2:
                st.metric("Rolling Window", rolling_window_label)
                st.metric("Data Points", f"{len(returns_data)} days")
            
            # Display the rolling correlation plot
            if selected_pairs:
                fig_rolling = plot_rolling_correlations(rolling_correlations, selected_pairs, portfolio_funds)
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Show statistics for selected pairs
                st.subheader("Rolling Correlation Statistics")
                stats_data = []
                
                for pair in selected_pairs:
                    if pair in rolling_correlations:
                        corr_series = rolling_correlations[pair]
                        stats_data.append({
                            'Fund Pair': pair,
                            'Current Correlation': corr_series.iloc[-1] if len(corr_series) > 0 else np.nan,
                            'Average Correlation': corr_series.mean(),
                            'Max Correlation': corr_series.max(),
                            'Min Correlation': corr_series.min(),
                            'Volatility (Std Dev)': corr_series.std()
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df.round(3), use_container_width=True)
            else:
                st.info("Please select at least one fund pair to display the rolling correlation chart.")
        else:
            st.warning("Not enough data to calculate rolling correlations with the selected window size.")
    
    # Display analysis and recommendations
    st.subheader("🎯 Correlation Analysis & Recommendations")
    
    # Show key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Correlation", f"{analysis['average_correlation']:.3f}")
    
    with col2:
        st.metric("High Correlation Pairs", len(analysis['high_correlation']))
    
    with col3:
        st.metric("Negative Correlation Pairs", len(analysis['negative_correlation']))
    
    with col4:
        st.metric("Portfolio Volatility", f"{risk_df.attrs.get('portfolio_volatility', 0):.4f}")
    
    # Display specific allocation recommendations
    st.subheader("📌 Specific Allocation Recommendations")
    
    for rec in allocation_recs:
        with st.expander(f"{'🔴' if 'Reduce' in rec['type'] else '🟢'} {rec['type']}"):
            st.write(rec['details'])
            
            for fund in rec['funds']:
                if rec['type'] == 'Reduce High Risk Contributors':
                    st.write(f"- **{fund['name']}** (Category: {fund['category']})")
                    st.write(f"  - Current Weight: {fund['weight']:.1f}%")
                    st.write(f"  - Risk Contribution: {fund['rc_pct']:.1f}%")
                    st.write(f"  → Consider reducing allocation by 5-10%")
                    st.write(f"  → Potential replacement: Look for lower-correlation funds in same category")
                
                elif rec['type'] == 'Increase Low Risk Contributors':
                    st.write(f"- **{fund['name']}** (Category: {fund['category']})")
                    st.write(f"  - Current Weight: {fund['weight']:.1f}%")
                    st.write(f"  - Risk Contribution: {fund['rc_pct']:.1f}%")
                    st.write(f"  → Consider increasing allocation by 5-10%")
                    st.write(f"  → Good candidate for additional investments")
                
                elif rec['type'] == 'Rebalance Risk-Heavy Funds':
                    st.write(f"- **{fund['name']}** (Category: {fund['category']})")
                    st.write(f"  - Current Weight: {fund['weight']:.1f}%")
                    st.write(f"  - Risk-Weight Ratio: {fund['rwr']:.2f}")
                    st.write(f"  → Consider reducing allocation to balance risk contribution")
                    st.write(f"  → Rebalance proceeds to funds with RWR < 0.8")
            
            if rec['type'] == 'Reduce High Risk Contributors':
                st.write("\n**Action Plan:**")
                st.write("1. Identify lower-correlation alternatives for these funds")
                st.write("2. Gradually reduce allocations (5-10% per rebalance)")
                st.write("3. Monitor impact on portfolio volatility")
            
            elif rec['type'] == 'Increase Low Risk Contributors':
                st.write("\n**Action Plan:**")
                st.write("1. Prioritize these funds for new investments")
                st.write("2. Consider modest increases in allocation (5-10%)")
                st.write("3. Verify they maintain their diversification characteristics")
            
            elif rec['type'] == 'Rebalance Risk-Heavy Funds':
                st.write("\n**Action Plan:**")
                st.write("1. Reduce allocations to these funds")
                st.write("2. Rebalance proceeds to funds with RWR < 0.8")
                st.write("3. Aim for more balanced risk contributions")
    
    # Display correlation-based recommendations
    st.subheader("🔍 Correlation-Based Recommendations")
    for rec in recommendations:
        with st.expander(f"💡 {rec['type']}"):
            st.write(rec['details'])
            
            if 'fund_pairs' in rec:
                for pair in rec['fund_pairs']:
                    st.write(f"- {pair}")
            
            if 'suggestions' in rec:
                st.write("**Suggestions:**")
                for suggestion in rec['suggestions']:
                    st.write(f"- {suggestion}")
    
    # Additional insights section
    with st.expander("📊 Additional Insights"):
        st.write("**Understanding Risk Contributions:**")
        st.write("- **High RC% Funds**: Contribute disproportionately to portfolio risk - consider reducing")
        st.write("- **Low RC% Funds**: Provide diversification - consider increasing")
        st.write("- **Negative Correlation**: Excellent for risk reduction")
        
        st.write("\n**Understanding Rolling Correlations:**")
        st.write("- **High correlation (>0.7)**: Funds move very similarly - consider reducing overlap")
        st.write("- **Moderate correlation (0.3-0.7)**: Some similarity but still providing diversification")  
        st.write("- **Low correlation (<0.3)**: Good diversification benefits")
        st.write("- **Negative correlation (<0)**: Excellent for risk reduction")
        
        st.write("\n**Time-varying correlations can indicate:**")
        st.write("- Market stress periods (correlations often increase)")
        st.write("- Sector rotation effects")
        st.write("- Fund manager style changes")
        st.write("- Market regime changes")
    
    # Show underlying data
    with st.expander("📋 Underlying Data"):
        tab1, tab2, tab3 = st.tabs(["Returns Data", "Fund Information", "Correlation Matrix"])
        
        with tab1:
            st.write("Daily returns used for correlation calculation:")
            st.dataframe(returns_data.tail(100))  # Show last 100 days
        
        with tab2:
            st.write("Funds in your portfolio:")
            st.dataframe(portfolio_funds)
        
        with tab3:
            st.write("Full correlation matrix:")
            st.dataframe(corr_matrix.round(3))

if __name__ == "__main__":
    main()