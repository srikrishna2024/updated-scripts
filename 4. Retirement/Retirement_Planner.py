# Complete Integrated Retirement Planner with All Features
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg
from scipy.optimize import minimize
from datetime import datetime
import plotly.express as px

# -------------------- DATABASE CONFIG --------------------
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    return psycopg.connect(**DB_PARAMS)

# -------------------- UTILITY FUNCTIONS --------------------
def format_indian_currency(amount):
    """Format numbers in Indian style (lakhs, crores)"""
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    amount = float(amount)
    if amount < 100000:
        return f"₹{amount:,.0f}"
    elif amount < 10000000:
        lakhs = amount / 100000
        return f"₹{lakhs:,.2f} L"
    else:
        crores = amount / 10000000
        return f"₹{crores:,.2f} Cr"

def reset_database():
    """Clear all retirement planning data"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM retirement_progress")
            cur.execute("DELETE FROM retirement_plans")
            conn.commit()
    st.success("All retirement planning data has been reset!")
    st.rerun()

# -------------------- DATABASE SETUP --------------------
def initialize_database():
    """Create required tables if they don't exist"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Retirement plans table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS retirement_plans (
                plan_id SERIAL PRIMARY KEY,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                plan_type VARCHAR(20) CHECK (plan_type IN ('BASELINE', 'UPDATE')),
                current_age INTEGER,
                retirement_age INTEGER,
                life_expectancy INTEGER,
                current_monthly_expenses NUMERIC,
                annual_expenses_growth NUMERIC,
                recurring_annual_expenses NUMERIC,
                post_retirement_inflation NUMERIC,
                equity_return NUMERIC,
                debt_return NUMERIC,
                corpus_needed NUMERIC,
                notes TEXT
            )
            """)
            
            # Retirement progress table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS retirement_progress (
                progress_id SERIAL PRIMARY KEY,
                plan_id INTEGER REFERENCES retirement_plans(plan_id),
                check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_equity NUMERIC,
                current_debt NUMERIC,
                projected_equity NUMERIC,
                projected_debt NUMERIC,
                on_track BOOLEAN,
                variance NUMERIC,
                notes TEXT
            )
            """)
            conn.commit()

# -------------------- PORTFOLIO FUNCTIONS --------------------
def get_portfolio_data():
    """Retrieve all records from portfolio_data table"""
    with get_db_connection() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs from mutual_fund_nav table"""
    with get_db_connection() as conn:
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
    """Retrieve goal mappings from the goals table"""
    with get_db_connection() as conn:
        query = """
            SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
            FROM goals
            ORDER BY goal_name, investment_type, scheme_name
        """
        return pd.read_sql(query, conn)

def get_retirement_investments():
    """Get current investments tagged for retirement goal"""
    with get_db_connection() as conn:
        # Get equity investments for retirement
        equity_query = """
        SELECT COALESCE(SUM(pd.units * mf.value), 0) as equity_value
        FROM portfolio_data pd
        JOIN mutual_fund_nav mf ON pd.code = mf.code
        JOIN goals g ON pd.code = g.scheme_code
        WHERE g.goal_name = 'Retirement'
        AND g.investment_type = 'Equity'
        AND mf.nav = (SELECT MAX(nav) FROM mutual_fund_nav WHERE code = pd.code)
        AND pd.transaction_type IN ('invest', 'switch_in')
        """
        equity_value = pd.read_sql(equity_query, conn).iloc[0,0]
        
        # Get debt investments for retirement
        debt_query = """
        SELECT COALESCE(SUM(current_value), 0) as debt_value
        FROM goals
        WHERE goal_name = 'Retirement'
        AND investment_type = 'Debt'
        """
        debt_value = pd.read_sql(debt_query, conn).iloc[0,0]
        
        return equity_value, debt_value

# -------------------- RETIREMENT PLANNING FUNCTIONS --------------------
def calculate_retirement_corpus(
    current_age,
    retirement_age,
    current_monthly_expenses,
    annual_expenses_growth,
    recurring_annual_expenses,
    post_retirement_inflation,
    equity_return,
    debt_return,
    current_equity=0,
    current_debt=0,
    life_expectancy=90
):
    """
    Calculate retirement corpus needed and annual investment requirements.
    """
    # Validate inputs
    if retirement_age <= current_age:
        st.error("Retirement age must be greater than current age")
        return None, None, None
    
    years_to_retirement = retirement_age - current_age
    retirement_years = life_expectancy - retirement_age
    
    # Calculate current annual expenses
    current_annual_expenses = current_monthly_expenses * 12 + recurring_annual_expenses
    
    # Project expenses at retirement (growing at annual_expenses_growth)
    retirement_annual_expenses = current_annual_expenses * (
        (1 + annual_expenses_growth) ** years_to_retirement
    )
    
    # Calculate corpus needed using present value of retirement expenses
    corpus_needed = 0
    for year in range(1, retirement_years + 1):
        year_expenses = retirement_annual_expenses * ((1 + post_retirement_inflation) ** (year - 1))
        # Discount back to retirement date using conservative post-retirement returns
        post_ret_return = 0.5 * equity_return + 0.5 * debt_return  # Balanced portfolio in retirement
        corpus_needed += year_expenses / ((1 + post_ret_return) ** year)
    
    # Add 10% buffer for unexpected expenses
    corpus_needed *= 1.1
    
    # Calculate annual investments needed with glide path
    projection_data = []
    current_equity_value = current_equity
    current_debt_value = current_debt
    
    for year in range(1, years_to_retirement + 1):
        age = current_age + year
        years_remaining = years_to_retirement - year
        
        # Calculate target allocation (glide path)
        equity_pct = max(0.3, 0.8 - (0.5 * (year / years_to_retirement)))  # Glide from 80% to 30% equity
        debt_pct = 1 - equity_pct
        
        # Calculate investment growth
        equity_growth = current_equity_value * equity_return
        debt_growth = current_debt_value * debt_return
        
        # Update values with growth
        current_equity_value += equity_growth
        current_debt_value += debt_growth
        total_corpus = current_equity_value + current_debt_value
        
        # Calculate required annual investment to reach target
        if year < years_to_retirement:
            # Future value needed at retirement
            fv_needed = corpus_needed - total_corpus
            
            # Calculate required annual investment (growing with income growth)
            growth_rate = min(0.05, annual_expenses_growth)  # Assume salary grows with inflation but cap at 5%
            effective_return = equity_pct * equity_return + debt_pct * debt_return
            if effective_return != growth_rate:
                annuity_factor = (((1 + effective_return) ** years_remaining - 
                                 (1 + growth_rate) ** years_remaining) / 
                                (effective_return - growth_rate))
            else:
                annuity_factor = years_remaining * (1 + effective_return) ** (years_remaining - 1)
            
            required_investment = max(0, fv_needed / annuity_factor)
        else:
            required_investment = 0
        
        # Split investment by allocation
        equity_investment = required_investment * equity_pct
        debt_investment = required_investment * debt_pct
        
        # Update corpus with new investments
        current_equity_value += equity_investment
        current_debt_value += debt_investment
        total_corpus = current_equity_value + current_debt_value
        
        # Record projection data
        projection_data.append({
            'Year': age,
            'Age': age,
            'Years to Retirement': years_remaining,
            'Equity Allocation %': equity_pct * 100,
            'Debt Allocation %': debt_pct * 100,
            'Equity Value': current_equity_value,
            'Debt Value': current_debt_value,
            'Total Corpus': total_corpus,
            'Annual Investment': required_investment,
            'Equity Investment': equity_investment,
            'Debt Investment': debt_investment,
            'Projected Expenses': current_annual_expenses * ((1 + annual_expenses_growth) ** year)
        })
    
    corpus_projection = pd.DataFrame(projection_data)
    
    # Calculate average annual investments
    if not corpus_projection.empty:
        avg_annual_investment = corpus_projection['Annual Investment'].mean()
    else:
        avg_annual_investment = 0
    
    annual_investments = {
        'total': avg_annual_investment,
        'equity': avg_annual_investment * 0.8,  # Start with 80% equity allocation
        'debt': avg_annual_investment * 0.2     # Start with 20% debt allocation
    }
    
    return corpus_needed, annual_investments, corpus_projection

# -------------------- RETIREMENT TRACKING FUNCTIONS --------------------
def save_retirement_plan(plan_data, plan_type="BASELINE"):
    """Save retirement plan to database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO retirement_plans (
                plan_type,
                current_age,
                retirement_age,
                life_expectancy,
                current_monthly_expenses,
                annual_expenses_growth,
                recurring_annual_expenses,
                post_retirement_inflation,
                equity_return,
                debt_return,
                corpus_needed,
                notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING plan_id
            """, (
                plan_type,
                plan_data['current_age'],
                plan_data['retirement_age'],
                plan_data['life_expectancy'],
                plan_data['current_monthly_expenses'],
                plan_data['annual_expenses_growth'],
                plan_data['recurring_annual_expenses'],
                plan_data['post_retirement_inflation'],
                plan_data['equity_return'],
                plan_data['debt_return'],
                plan_data['corpus_needed'],
                plan_data.get('notes', '')
            ))
            plan_id = cur.fetchone()[0]
            conn.commit()
            return plan_id

def save_progress_update(plan_id, progress_data):
    """Save progress update against a plan"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO retirement_progress (
                plan_id,
                current_equity,
                current_debt,
                projected_equity,
                projected_debt,
                on_track,
                variance,
                notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            """, (
                plan_id,
                progress_data['current_equity'],
                progress_data['current_debt'],
                progress_data['projected_equity'],
                progress_data['projected_debt'],
                progress_data['on_track'],
                progress_data['variance'],
                progress_data.get('notes', '')
            ))
            conn.commit()

def get_retirement_plans():
    """Get all retirement plans"""
    with get_db_connection() as conn:
        return pd.read_sql("""
        SELECT * FROM retirement_plans ORDER BY created_date DESC
        """, conn)

def get_plan_progress(plan_id):
    """Get progress updates for a specific plan"""
    with get_db_connection() as conn:
        return pd.read_sql("""
        SELECT * FROM retirement_progress 
        WHERE plan_id = %s 
        ORDER BY check_date
        """, conn, params=(plan_id,))

def plot_glide_path(projection_df):
    """Plot asset allocation glide path"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(projection_df['Year'], projection_df['Equity Allocation %'], label='Equity %', color='blue')
    ax.plot(projection_df['Year'], projection_df['Debt Allocation %'], label='Debt %', color='orange')
    ax.set_title('Asset Allocation Glide Path', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Allocation %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_progress_comparison(progress_data):
    """Create interactive progress comparison chart"""
    progress_data['Total Actual'] = progress_data['current_equity'] + progress_data['current_debt']
    progress_data['Total Projected'] = progress_data['projected_equity'] + progress_data['projected_debt']
    
    fig = px.line(
        progress_data, 
        x='check_date', 
        y=['Total Actual', 'Total Projected'],
        labels={'value': 'Amount (₹)', 'check_date': 'Date'},
        title='Actual vs Projected Retirement Corpus'
    )
    
    # Add markers and improve styling
    fig.update_traces(
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=8)
    )
    
    # Add variance annotations
    for i, row in progress_data.iterrows():
        variance = (row['Total Actual'] - row['Total Projected']) / row['Total Projected'] * 100
        fig.add_annotation(
            x=row['check_date'],
            y=row['Total Actual'],
            text=f"{variance:.1f}%",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        hovermode='x unified',
        legend_title_text='Corpus Type',
        yaxis_tickprefix='₹',
        yaxis_tickformat=',.2r'
    )
    
    return fig

# -------------------- STREAMLIT UI --------------------
def retirement_planner_tab():
    st.header("🏦 Retirement Planner with Progress Tracking")
    
    # Initialize database
    initialize_database()
    
    # Check for existing plans
    existing_plans = get_retirement_plans()
    has_baseline = not existing_plans.empty and 'BASELINE' in existing_plans['plan_type'].values
    
    # Sidebar inputs
    with st.sidebar:
        st.subheader("🔢 Personal Details")
        current_age = st.number_input("Your Current Age", min_value=18, max_value=70, value=30)
        retirement_age = st.number_input("Planned Retirement Age", min_value=current_age+1, max_value=80, value=60)
        life_expectancy = st.number_input("Life Expectancy", min_value=retirement_age+1, max_value=120, value=90)
        
        st.subheader("💰 Expenses")
        current_monthly_expenses = st.number_input("Current Monthly Living Expenses (₹)", 
                                                min_value=10000, value=50000, step=5000)
        recurring_annual_expenses = st.number_input("Annual Recurring Expenses (₹)", 
                                                 min_value=0, value=100000, step=10000)
        annual_expenses_growth = st.slider("Expected Annual Expenses Growth %", 
                                        min_value=0.0, max_value=15.0, value=6.0, step=0.1) / 100
        post_retirement_inflation = st.slider("Post-Retirement Inflation %", 
                                           min_value=0.0, max_value=10.0, value=5.0, step=0.1) / 100
        
        st.subheader("📈 Expected Returns")
        equity_return = st.slider("Post-Tax Equity Return %", 
                               min_value=0.0, max_value=20.0, value=10.0, step=0.1) / 100
        debt_return = st.slider("Post-Tax Debt Return %", 
                             min_value=0.0, max_value=15.0, value=6.0, step=0.1) / 100
        
        st.subheader("💵 Current Investments")
        auto_load = st.checkbox("Auto-load retirement investments", value=True)
        
        if auto_load:
            with st.spinner("Loading retirement investments..."):
                current_equity, current_debt = get_retirement_investments()
            st.metric("Retirement Equity", format_indian_currency(current_equity))
            st.metric("Retirement Debt", format_indian_currency(current_debt))
        else:
            current_equity = st.number_input("Current Equity Investments (₹)", 
                                          min_value=0, value=500000, step=10000)
            current_debt = st.number_input("Current Debt Investments (₹)", 
                                        min_value=0, value=300000, step=10000)
        
        # Reset button
        if st.button("Reset All Data", type="secondary"):
            reset_database()
    
    # Main content
    if not has_baseline:
        st.warning("No baseline retirement plan found. Create your baseline plan first.")
        if st.button("Create Baseline Retirement Plan", type="primary"):
            with st.spinner("Creating your baseline retirement plan..."):
                corpus_needed, annual_investments, projection_df = calculate_retirement_corpus(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    current_monthly_expenses=current_monthly_expenses,
                    annual_expenses_growth=annual_expenses_growth,
                    recurring_annual_expenses=recurring_annual_expenses,
                    post_retirement_inflation=post_retirement_inflation,
                    equity_return=equity_return,
                    debt_return=debt_return,
                    current_equity=current_equity,
                    current_debt=current_debt,
                    life_expectancy=life_expectancy
                )
                
                if corpus_needed is not None:
                    # Save baseline plan
                    plan_id = save_retirement_plan({
                        'current_age': current_age,
                        'retirement_age': retirement_age,
                        'life_expectancy': life_expectancy,
                        'current_monthly_expenses': current_monthly_expenses,
                        'annual_expenses_growth': annual_expenses_growth,
                        'recurring_annual_expenses': recurring_annual_expenses,
                        'post_retirement_inflation': post_retirement_inflation,
                        'equity_return': equity_return,
                        'debt_return': debt_return,
                        'corpus_needed': corpus_needed,
                        'notes': 'Initial baseline plan'
                    }, plan_type="BASELINE")
                    
                    # Save initial progress
                    save_progress_update(plan_id, {
                        'current_equity': current_equity,
                        'current_debt': current_debt,
                        'projected_equity': current_equity,
                        'projected_debt': current_debt,
                        'on_track': True,
                        'variance': 0,
                        'notes': 'Baseline established'
                    })
                    
                    st.success("✅ Baseline retirement plan created successfully!")
                    st.rerun()
    else:
        baseline_plan = existing_plans[existing_plans['plan_type'] == 'BASELINE'].iloc[0]
        
        # Display baseline info
        with st.expander("📋 View Baseline Plan", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Created on", baseline_plan['created_date'].strftime('%Y-%m-%d'))
                st.metric("Corpus Needed", format_indian_currency(baseline_plan['corpus_needed']))
                st.metric("Current Age", baseline_plan['current_age'])
                st.metric("Retirement Age", baseline_plan['retirement_age'])
            with col2:
                st.metric("Equity Return", f"{baseline_plan['equity_return']*100:.1f}%")
                st.metric("Debt Return", f"{baseline_plan['debt_return']*100:.1f}%")
                st.metric("Annual Expense Growth", f"{baseline_plan['annual_expenses_growth']*100:.1f}%")
                st.metric("Post-Retirement Inflation", f"{baseline_plan['post_retirement_inflation']*100:.1f}%")
        
        # Show glide path
        st.subheader("📉 Asset Allocation Glide Path")
        _, annual_investments, projection_df = calculate_retirement_corpus(
            current_age=baseline_plan['current_age'],
            retirement_age=baseline_plan['retirement_age'],
            current_monthly_expenses=baseline_plan['current_monthly_expenses'],
            annual_expenses_growth=baseline_plan['annual_expenses_growth'],
            recurring_annual_expenses=baseline_plan['recurring_annual_expenses'],
            post_retirement_inflation=baseline_plan['post_retirement_inflation'],
            equity_return=baseline_plan['equity_return'],
            debt_return=baseline_plan['debt_return'],
            current_equity=0,
            current_debt=0,
            life_expectancy=baseline_plan['life_expectancy']
        )
        st.pyplot(plot_glide_path(projection_df))
        
        # Check progress button
        if st.button("Check Progress Against Baseline", type="primary"):
            with st.spinner("Calculating progress against baseline..."):
                # Get current values
                current_equity, current_debt = get_retirement_investments()
                
                # Calculate where we should be based on baseline projections
                baseline_progress = get_plan_progress(baseline_plan['plan_id'])
                baseline_date = baseline_plan['created_date']
                days_passed = (datetime.now() - baseline_date).days
                years_passed = days_passed / 365
                
                projected_equity = baseline_progress.iloc[0]['current_equity'] * ((1 + baseline_plan['equity_return']) ** years_passed)
                projected_debt = baseline_progress.iloc[0]['current_debt'] * ((1 + baseline_plan['debt_return']) ** years_passed)
                
                # Calculate variance
                total_current = current_equity + current_debt
                total_projected = projected_equity + projected_debt
                variance = (total_current - total_projected) / total_projected if total_projected > 0 else 0
                on_track = variance >= -0.1  # Within 10% of target
                
                # Save progress update
                save_progress_update(baseline_plan['plan_id'], {
                    'current_equity': current_equity,
                    'current_debt': current_debt,
                    'projected_equity': projected_equity,
                    'projected_debt': projected_debt,
                    'on_track': on_track,
                    'variance': variance,
                    'notes': f'Progress check after {years_passed:.1f} years'
                })
                
                # Display results
                st.subheader("📊 Progress Against Baseline")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Corpus", format_indian_currency(total_current),
                            delta=f"{variance*100:.1f}% vs projected")
                with col2:
                    st.metric("Projected Corpus", format_indian_currency(total_projected))
                with col3:
                    status = "✅ On Track" if on_track else "⚠️ Needs Attention"
                    st.metric("Status", status)
                
                # Plot progress comparison
                progress_data = get_plan_progress(baseline_plan['plan_id'])
                if not progress_data.empty:
                    st.plotly_chart(plot_progress_comparison(progress_data))
                
                # Detailed comparison
                with st.expander("📝 Detailed Comparison", expanded=False):
                    comparison_df = pd.DataFrame({
                        'Metric': ['Equity', 'Debt', 'Total'],
                        'Current': [current_equity, current_debt, total_current],
                        'Projected': [projected_equity, projected_debt, total_projected],
                        'Variance (₹)': [
                            current_equity - projected_equity,
                            current_debt - projected_debt,
                            total_current - total_projected
                        ],
                        'Variance (%)': [
                            (current_equity - projected_equity) / projected_equity * 100 if projected_equity > 0 else 0,
                            (current_debt - projected_debt) / projected_debt * 100 if projected_debt > 0 else 0,
                            variance * 100
                        ]
                    })
                    comparison_df['Current'] = comparison_df['Current'].apply(format_indian_currency)
                    comparison_df['Projected'] = comparison_df['Projected'].apply(format_indian_currency)
                    comparison_df['Variance (₹)'] = comparison_df['Variance (₹)'].apply(format_indian_currency)
                    comparison_df['Variance (%)'] = comparison_df['Variance (%)'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if on_track:
                    st.success("You're on track with your retirement goals! Keep up the good work.")
                    st.markdown("""
                    - Continue with your current investment strategy
                    - Consider reviewing your asset allocation annually
                    - Monitor expense growth to ensure it stays within projections
                    """)
                else:
                    st.warning("You're behind your retirement goals. Consider:")
                    if variance < -0.1:
                        st.markdown("""
                        - **Increase contributions**: Boost monthly investments by 10-20%
                        - **Review allocation**: Ensure you're following the recommended equity/debt split
                        - **Reduce expenses**: Find ways to cut unnecessary spending
                        - **Additional income**: Explore side income opportunities
                        - **Extend timeline**: Consider working 1-2 more years if possible
                        """)
                    
                    # Calculate required increase to get back on track
                    years_remaining = baseline_plan['retirement_age'] - baseline_plan['current_age'] - years_passed
                    if years_remaining > 0:
                        additional_needed = total_projected - total_current
                        monthly_additional = additional_needed / (years_remaining * 12)
                        st.info(f"To get back on track, consider increasing monthly investments by **{format_indian_currency(monthly_additional)}** for the next {years_remaining:.1f} years")

def portfolio_optimizer_tab():
    st.header("📊 Portfolio Optimizer")
    
    # Load categories
    with get_db_connection() as conn:
        categories_df = pd.read_sql(
            "SELECT DISTINCT scheme_category FROM mutual_fund_master_data WHERE scheme_category IS NOT NULL",
            conn
        )
    categories = sorted(categories_df['scheme_category'].dropna().unique().tolist())
    
    # Rolling period selection
    rolling_period_years = st.slider("Analysis Period (Years)", min_value=1, max_value=10, value=5, step=1)
    
    # Main interface with tabs
    subtab1, subtab2, subtab3 = st.tabs(["📈 Analyze Categories", "🔍 Compare Funds", "🔄 Rebalance Portfolio"])
    
    with subtab1:
        st.subheader("Category Analysis")
        selected_categories = st.multiselect(
            "Select up to 3 Fund Categories", 
            categories, 
            max_selections=3
        )
        
        if st.button("Analyze Categories", disabled=len(selected_categories) == 0):
            st.warning("Category analysis would be implemented here")
    
    with subtab2:
        st.subheader("Fund Comparison")
        # Load all funds for selection
        with get_db_connection() as conn:
            query = """
                SELECT DISTINCT code, scheme_name, scheme_category
                FROM mutual_fund_master_data
                ORDER BY scheme_name
            """
            df_all_funds = pd.read_sql(query, conn)
        
        # Create fund options with category info
        fund_options = [f"{row['scheme_name']} ({row['scheme_category']})" for _, row in df_all_funds.iterrows()]
        
        selected_fund_options = st.multiselect(
            "Choose 2-3 Funds to Compare", 
            fund_options, 
            max_selections=3
        )
        
        if st.button("Compare Funds", disabled=len(selected_fund_options) < 2):
            st.warning("Fund comparison would be implemented here")
    
    with subtab3:
        st.subheader("Portfolio Rebalancer")
        if st.button("Analyze Current Allocation"):
            with st.spinner("Loading your portfolio..."):
                try:
                    # Get retirement-specific investments
                    retirement_equity, retirement_debt = get_retirement_investments()
                    
                    # Get all goals
                    goals = get_goal_mappings()
                    
                    # Display retirement allocation
                    st.write("### Retirement Portfolio")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Equity", format_indian_currency(retirement_equity))
                    with col2:
                        st.metric("Total Debt", format_indian_currency(retirement_debt))
                    
                    # Show retirement goal investments
                    retirement_goals = goals[goals['goal_name'] == 'Retirement']
                    if not retirement_goals.empty:
                        st.write("#### Retirement Investments")
                        st.dataframe(retirement_goals[['investment_type', 'scheme_name', 'current_value']]
                                   .assign(current_value=lambda x: x['current_value'].apply(format_indian_currency)))
                    
                except Exception as e:
                    st.error(f"Error loading portfolio: {str(e)}")

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Integrated Financial Planner",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏦 Retirement Planner", "📊 Portfolio Optimizer"])
    
    with tab1:
        retirement_planner_tab()
    
    with tab2:
        portfolio_optimizer_tab()

if __name__ == "__main__":
    main()