#!/usr/bin/env python3
"""
Holiday 2026/2027 ROAS Forecast Dashboard
Interactive dashboard for Pinterest holiday campaign planning
Based on 2025/2026 historical performance data
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Holiday 2026/2027 ROAS Forecast",
    page_icon="🎄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E60023;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E60023;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the CSV file"""
    # Use relative path for deployment
    import os

    # Check which file exists and use it
    possible_files = [
        '2026ROASRAMP_Holiday_weekly.csv',
        '2026ROASRAMP_Holiday_filtered.csv',
        '2026ROASRAMP_Holiday.csv',
        '/Users/nkavanagh/Downloads/2026ROASRAMP_Holiday.csv'
    ]

    file_path = None
    is_preaggregated = False

    for f in possible_files:
        if os.path.exists(f):
            file_path = f
            if 'weekly' in f:
                is_preaggregated = True
            break

    if file_path is None:
        st.error("❌ Data file not found. Please ensure '2026ROASRAMP_Holiday_weekly.csv' is in the repository.")
        st.stop()

    # Read CSV with error handling for malformed lines
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', encoding='utf-8', header=0)

    # Verify required columns exist
    if is_preaggregated and 'week' not in df.columns:
        st.error(f"❌ Expected 'week' column not found in {file_path}. Available columns: {list(df.columns)}")
        st.stop()

    if is_preaggregated:
        # Data is already weekly aggregated
        df['week'] = pd.to_datetime(df['week'])
        # Shift dates forward by 1 year for 2026/2027 forecast
        df['week'] = df['week'] + pd.DateOffset(years=1)
        # Add day column for compatibility
        df['day'] = df['week']
    else:
        # Convert day to datetime
        df['day'] = pd.to_datetime(df['day'])
        # Shift dates forward by 1 year for 2026/2027 forecast
        df['day'] = df['day'] + pd.DateOffset(years=1)
        # Add helper columns
        df['week'] = df['day'].dt.to_period('W').apply(lambda r: r.start_time)

        # Filter extreme outliers for ROAS (keep values between 0 and 99th percentile)
        roas_99th = df['ROAS'].quantile(0.99)
        df = df[df['ROAS'] <= roas_99th]

        # Filter extreme outliers for CPA (keep values <= 99th percentile)
        cpa_99th = df['CPA'].quantile(0.99)
        df = df[df['CPA'] <= cpa_99th]

    df['month'] = df['week'].dt.to_period('M').apply(lambda r: r.start_time)
    df['year_week'] = df['week'].dt.strftime('%Y-W%W')

    # Clean numeric columns (using USD values only)
    numeric_cols = ['CPM (USD)', 'ROAS', 'CPA', 'CTR (%)', 'CVR (%)', 'AOV']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def calculate_budget_allocation(filtered_df, total_budget, objective_splits):
    """Calculate budget allocation based on objective-specific KPIs
    - Awareness: CPM (lower is better)
    - Consideration: CTR (higher is better)
    - Conversion: CVR (higher is better)
    - Shopping: CVR (higher is better)
    """

    # Group by week and calculate median (more robust to outliers)
    weekly_data = filtered_df.groupby('week').agg({
        'CPM (USD)': 'median',
        'CTR (%)': 'median',
        'CVR (%)': 'median',
        'ROAS': 'median',
        'CPA': 'median',
        'AOV': 'median'
    }).reset_index()

    # Calculate allocation indices for each objective
    # AWARENESS: CPM - lower is better, so invert
    if weekly_data['CPM (USD)'].sum() > 0:
        # Invert CPM (1/CPM) so lower CPM = higher allocation
        weekly_data['cpm_inverse'] = 1 / weekly_data['CPM (USD)'].replace(0, 0.01)
        weekly_data['awareness_index'] = weekly_data['cpm_inverse'] / weekly_data['cpm_inverse'].sum()
    else:
        weekly_data['awareness_index'] = 1.0 / len(weekly_data)

    # CONSIDERATION: CTR - higher is better
    if weekly_data['CTR (%)'].sum() > 0:
        weekly_data['consideration_index'] = weekly_data['CTR (%)'] / weekly_data['CTR (%)'].sum()
    else:
        weekly_data['consideration_index'] = 1.0 / len(weekly_data)

    # CONVERSION: CVR - higher is better
    if weekly_data['CVR (%)'].sum() > 0:
        weekly_data['conversion_index'] = weekly_data['CVR (%)'] / weekly_data['CVR (%)'].sum()
    else:
        weekly_data['conversion_index'] = 1.0 / len(weekly_data)

    # SHOPPING: CVR - higher is better
    if weekly_data['CVR (%)'].sum() > 0:
        weekly_data['shopping_index'] = weekly_data['CVR (%)'] / weekly_data['CVR (%)'].sum()
    else:
        weekly_data['shopping_index'] = 1.0 / len(weekly_data)

    # Allocate budget using objective-specific indices
    allocations = []
    for idx, row in weekly_data.iterrows():
        # Each objective gets allocated based on its own KPI
        awareness_weekly = objective_splits['Awareness'] * row['awareness_index']
        consideration_weekly = objective_splits['Consideration'] * row['consideration_index']
        conversion_weekly = objective_splits['Conversion'] * row['conversion_index']
        shopping_weekly = objective_splits['Shopping'] * row['shopping_index']

        week_total = awareness_weekly + consideration_weekly + conversion_weekly + shopping_weekly
        week_pct = (week_total / total_budget * 100) if total_budget > 0 else 0

        allocations.append({
            'week': row['week'],
            'budget_pct': week_pct,
            'awareness_budget': awareness_weekly,
            'consideration_budget': consideration_weekly,
            'conversion_budget': conversion_weekly,
            'shopping_budget': shopping_weekly,
            'total_weekly_budget': week_total,
            'avg_cpm': row['CPM (USD)'],
            'avg_ctr': row['CTR (%)'],
            'avg_cvr': row['CVR (%)'],
            'avg_roas': row['ROAS'],
            'avg_cpa': row['CPA'],
            'avg_aov': row['AOV']
        })

    return pd.DataFrame(allocations)

# Main app
def main():
    st.markdown('<div class="main-header">🎄 Holiday 2026/2027 ROAS Forecast</div>', unsafe_allow_html=True)
    st.markdown("**Optimize your Pinterest holiday campaign budget allocation for 2026/2027 based on 2025/2026 performance data**")
    st.info("💵 All monetary values (CPA, AOV, CPM) are displayed in USD | 📅 Forecast dates based on +1 year from historical data")
    st.markdown("---")

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Date range
    min_date = df['day'].min()
    max_date = df['day'].max()

    # Default to October 5th of the forecast year
    default_start = pd.Timestamp('2026-10-05')
    # If October 5th is before the min date, use min date
    if default_start < min_date:
        default_start = min_date

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Channel filter (moved above countries)
    channels = sorted([c for c in df['pod_channel'].unique() if pd.notna(c)])
    selected_channels = st.sidebar.multiselect(
        "Channels",
        channels,
        default=channels
    )

    # Country filter
    countries = sorted([c for c in df['user_country'].unique() if pd.notna(c)])
    selected_countries = st.sidebar.multiselect(
        "Countries",
        countries,
        default=['US', 'GB', 'CA', 'AU'] if all(c in countries for c in ['US', 'GB', 'CA', 'AU']) else countries[:4]
    )

    # Vertical filter
    verticals = sorted([v for v in df['sfdc_vertical'].unique() if pd.notna(v)])
    selected_verticals = st.sidebar.multiselect(
        "Verticals",
        verticals,
        default=verticals[:3] if len(verticals) >= 3 else verticals
    )

    # Sub-Vertical filter (cascading - only show sub-verticals for selected verticals)
    if selected_verticals:
        available_sub_verticals = df[df['sfdc_vertical'].isin(selected_verticals)]
        sub_verticals = sorted([v for v in available_sub_verticals['sfdc_sub_vertical'].unique() if pd.notna(v)])
    else:
        sub_verticals = sorted([v for v in df['sfdc_sub_vertical'].unique() if pd.notna(v)])

    selected_sub_verticals = st.sidebar.multiselect(
        "Sub-Verticals",
        sub_verticals,
        default=[],
        help="Filtered based on selected Verticals"
    )

    # Micro-Vertical filter (cascading - only show micro-verticals for selected sub-verticals)
    if selected_sub_verticals:
        available_micro_verticals = df[df['sfdc_sub_vertical'].isin(selected_sub_verticals)]
        micro_verticals = sorted([v for v in available_micro_verticals['sfdc_micro_vertical'].unique() if pd.notna(v)])
    elif selected_verticals:
        available_micro_verticals = df[df['sfdc_vertical'].isin(selected_verticals)]
        micro_verticals = sorted([v for v in available_micro_verticals['sfdc_micro_vertical'].unique() if pd.notna(v)])
    else:
        micro_verticals = sorted([v for v in df['sfdc_micro_vertical'].unique() if pd.notna(v)])

    selected_micro_verticals = st.sidebar.multiselect(
        "Micro-Verticals",
        micro_verticals,
        default=[],
        help="Filtered based on selected Sub-Verticals or Verticals"
    )

    st.sidebar.markdown("---")

    # Budget inputs by objective
    st.sidebar.subheader("💰 Budget by Objective")

    awareness_budget = st.sidebar.number_input(
        "Awareness Budget ($)",
        min_value=0,
        max_value=10000000,
        value=30000,
        step=1000
    )

    consideration_budget = st.sidebar.number_input(
        "Consideration Budget ($)",
        min_value=0,
        max_value=10000000,
        value=40000,
        step=1000
    )

    conversion_budget = st.sidebar.number_input(
        "Conversion Budget ($)",
        min_value=0,
        max_value=10000000,
        value=30000,
        step=1000
    )

    shopping_budget = st.sidebar.number_input(
        "Shopping Budget ($)",
        min_value=0,
        max_value=10000000,
        value=20000,
        step=1000
    )

    total_budget = awareness_budget + consideration_budget + conversion_budget + shopping_budget

    objective_splits = {
        'Awareness': awareness_budget,
        'Consideration': consideration_budget,
        'Conversion': conversion_budget,
        'Shopping': shopping_budget
    }

    # Filter data
    filtered_df = df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['day'] >= pd.Timestamp(date_range[0])) &
            (filtered_df['day'] <= pd.Timestamp(date_range[1]))
        ]

    if selected_channels:
        filtered_df = filtered_df[filtered_df['pod_channel'].isin(selected_channels)]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['user_country'].isin(selected_countries)]

    if selected_verticals:
        filtered_df = filtered_df[filtered_df['sfdc_vertical'].isin(selected_verticals)]

    if selected_sub_verticals:
        filtered_df = filtered_df[filtered_df['sfdc_sub_vertical'].isin(selected_sub_verticals)]

    if selected_micro_verticals:
        filtered_df = filtered_df[filtered_df['sfdc_micro_vertical'].isin(selected_micro_verticals)]

    if len(filtered_df) == 0:
        st.warning("No data matches your filters. Please adjust your selections.")
        return

    # Calculate budget allocation first to get metrics
    allocation = calculate_budget_allocation(filtered_df, total_budget, objective_splits)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Budget",
            f"${total_budget:,.0f}",
            help="Your total holiday campaign budget across all objectives"
        )

    with col2:
        start_date = allocation['week'].min().strftime('%b %d')
        end_date = allocation['week'].max().strftime('%b %d, %Y')
        num_weeks = len(allocation)
        st.metric(
            "Campaign Period",
            f"{num_weeks} weeks",
            delta=f"{start_date} - {end_date}",
            help="Duration of your holiday campaign"
        )

    with col3:
        peak_idx = allocation['total_weekly_budget'].idxmax()
        peak_week_num = peak_idx + 1
        peak_amount = allocation['total_weekly_budget'].max()
        peak_pct = (peak_amount / total_budget * 100)
        st.metric(
            f"Peak Week (Week {peak_week_num})",
            f"${peak_amount:,.0f}",
            delta=f"{peak_pct:.1f}% of total",
            help="Week with highest budget allocation"
        )

    with col4:
        avg_weekly = allocation['total_weekly_budget'].mean()
        st.metric(
            "Avg Weekly Spend",
            f"${avg_weekly:,.0f}",
            help="Average budget per week"
        )

    st.markdown("---")

    # Sales Blurb Section
    with st.expander("📋 **Sales Recommendation Summary** (Click to expand)", expanded=False):
        # Calculate key stats for the blurb
        peak_idx = allocation['total_weekly_budget'].idxmax()
        peak_week_num = peak_idx + 1
        peak_date = allocation['week'].iloc[peak_idx].strftime('%B %d')
        peak_amount = allocation['total_weekly_budget'].max()

        # Find which objective has the largest budget
        obj_budgets = {
            'Awareness': awareness_budget,
            'Consideration': consideration_budget,
            'Conversion': conversion_budget,
            'Shopping': shopping_budget
        }
        top_objective = max(obj_budgets, key=obj_budgets.get)
        top_obj_pct = (obj_budgets[top_objective] / total_budget * 100)

        # Calculate budget distribution
        first_4_weeks = allocation.head(4)['total_weekly_budget'].sum()
        last_4_weeks = allocation.tail(4)['total_weekly_budget'].sum()

        if first_4_weeks > last_4_weeks:
            loading_strategy = "front-loaded"
            loading_desc = "concentrating investment in early October to capture high-intent shoppers during the initial holiday planning phase"
        elif last_4_weeks > first_4_weeks:
            loading_strategy = "back-loaded"
            loading_desc = "building momentum toward the peak holiday shopping period to maximize conversions when purchase intent is highest"
        else:
            loading_strategy = "evenly distributed"
            loading_desc = "maintaining consistent presence throughout the entire holiday season to capture shoppers at every stage of their journey"

        # Generate the blurb
        blurb = f"""
### Holiday 2026/2027 Campaign Recommendation

**Investment Overview:**
We recommend a **${total_budget:,}** investment across {len(allocation)} weeks ({allocation['week'].min().strftime('%B %d')} - {allocation['week'].max().strftime('%B %d, %Y')}), strategically allocated using historical performance data to maximize ROI during the critical holiday shopping season.

**Budget Allocation Strategy:**
Based on 2025/2026 performance data, this plan is **{loading_strategy}**, {loading_desc}. Your investment is optimized across four key objectives:

- **{top_objective}**: ${obj_budgets[top_objective]:,} ({top_obj_pct:.0f}%) - Primary focus
- **Awareness**: ${awareness_budget:,} ({(awareness_budget/total_budget*100):.0f}%) - Allocated based on CPM efficiency
- **Consideration**: ${consideration_budget:,} ({(consideration_budget/total_budget*100):.0f}%) - Optimized for click-through performance
- **Conversion + Shopping**: ${conversion_budget + shopping_budget:,} ({((conversion_budget+shopping_budget)/total_budget*100):.0f}%) - Driven by conversion rate data

**Peak Performance Window:**
Week {peak_week_num} ({peak_date}) represents your highest investment week at **${peak_amount:,}**, capturing the optimal performance period identified in historical data. This strategic concentration ensures maximum visibility during proven high-conversion windows.

**Data-Driven Optimization:**
This recommendation leverages objective-specific KPIs from our 2025/2026 performance database:
- Awareness budgets target weeks with lower CPMs for efficient reach
- Consideration budgets follow CTR trends to maximize engagement
- Conversion and Shopping budgets align with peak CVR periods for optimal ROAS

**Average Weekly Investment:** ${allocation['total_weekly_budget'].mean():,.0f}

**Next Steps:**
This allocation provides a strong foundation for holiday success. We recommend reviewing weekly performance during the campaign and remaining flexible to shift budget toward top-performing objectives and weeks as real-time data becomes available.
"""

        st.markdown(blurb)

        st.caption("💡 Tip: Select and copy the text above to use in your sales materials")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Weekly Allocation", "📈 Performance Trends", "🌍 Geographic Analysis", "📊 Budget Summary"])

    with tab1:
        st.subheader("Weekly Budget Allocation")
        st.markdown("**Budget distribution based on historical ROAS performance**")

        # Weekly budget bar chart (stacked)
        fig_weekly = go.Figure()

        # Create week labels with both week number and date
        week_labels = [f"Week {i+1}<br>{allocation['week'].iloc[i].strftime('%Y-%m-%d')}" for i in range(len(allocation))]

        fig_weekly.add_trace(go.Bar(
            x=week_labels,
            y=allocation['awareness_budget'],
            name='Awareness',
            marker_color='#667eea',
            hovertemplate='<b>Awareness</b><br>$%{y:,.0f}<extra></extra>',
            text=allocation['awareness_budget'],
            textposition='inside',
            texttemplate='$%{text:,.0f}',
            insidetextanchor='middle'
        ))

        fig_weekly.add_trace(go.Bar(
            x=week_labels,
            y=allocation['consideration_budget'],
            name='Consideration',
            marker_color='#764ba2',
            hovertemplate='<b>Consideration</b><br>$%{y:,.0f}<extra></extra>',
            text=allocation['consideration_budget'],
            textposition='inside',
            texttemplate='$%{text:,.0f}',
            insidetextanchor='middle'
        ))

        fig_weekly.add_trace(go.Bar(
            x=week_labels,
            y=allocation['conversion_budget'],
            name='Conversion',
            marker_color='#E60023',
            hovertemplate='<b>Conversion</b><br>$%{y:,.0f}<extra></extra>',
            text=allocation['conversion_budget'],
            textposition='inside',
            texttemplate='$%{text:,.0f}',
            insidetextanchor='middle'
        ))

        fig_weekly.add_trace(go.Bar(
            x=week_labels,
            y=allocation['shopping_budget'],
            name='Shopping',
            marker_color='#ff9800',
            hovertemplate='<b>Shopping</b><br>$%{y:,.0f}<extra></extra>',
            text=allocation['shopping_budget'],
            textposition='inside',
            texttemplate='$%{text:,.0f}',
            insidetextanchor='middle'
        ))

        fig_weekly.update_layout(
            barmode='stack',
            title='Weekly Budget by Objective',
            xaxis_title='Week',
            yaxis_title='Budget ($)',
            hovermode='x unified',
            height=400,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_weekly, use_container_width=True)

        st.markdown("---")

        # Week-over-week percentage change table
        st.markdown("#### 📊 Weekly Spend vs. Week 1 Baseline")

        # Calculate percentage change from Week 1
        week1_budget = allocation['total_weekly_budget'].iloc[0]

        wow_data = []
        for idx, row in allocation.iterrows():
            week_num = idx + 1
            current_budget = row['total_weekly_budget']
            pct_change = ((current_budget - week1_budget) / week1_budget * 100) if week1_budget > 0 else 0

            if week_num == 1:
                delivery_msg = f"${current_budget:,.0f}\nBaseline"
                arrow = ""
            elif pct_change > 0:
                delivery_msg = f"${current_budget:,.0f}\n↑ UP {pct_change:.0f}%"
                arrow = "↑"
            elif pct_change < 0:
                delivery_msg = f"${current_budget:,.0f}\n↓ DOWN {abs(pct_change):.0f}%"
                arrow = "↓"
            else:
                delivery_msg = f"${current_budget:,.0f}\nNo Change"
                arrow = "→"

            wow_data.append({
                'Week': f'Week {week_num}',
                'Delivery': delivery_msg
            })

        wow_df = pd.DataFrame(wow_data)

        # Create colored table using columns for better display
        num_weeks = len(wow_df)
        cols_per_row = 7
        num_rows = (num_weeks + cols_per_row - 1) // cols_per_row

        for row_idx in range(num_rows):
            cols = st.columns(cols_per_row)
            start_idx = row_idx * cols_per_row
            end_idx = min(start_idx + cols_per_row, num_weeks)

            for col_idx, (col, week_idx) in enumerate(zip(cols, range(start_idx, end_idx))):
                week_data = wow_df.iloc[week_idx]
                pct_change_value = ((allocation['total_weekly_budget'].iloc[week_idx] - week1_budget) / week1_budget * 100) if week1_budget > 0 else 0

                # Color based on percentage change
                # SPENDING MORE = Blue shades (ramping up)
                # SPENDING LESS = Orange/Yellow shades (ramping down)
                if week_idx == 0:
                    # Week 1 baseline
                    bg_color = "#4A4A4A"  # Dark Gray
                    text_color = "#FFFFFF"
                    border = "3px solid #FFD700"
                elif pct_change_value >= 50:
                    bg_color = "#003366"  # Dark blue (high spend increase)
                    text_color = "#FFFFFF"
                    border = "2px solid #0066CC"
                elif pct_change_value >= 30:
                    bg_color = "#0066CC"  # Royal blue
                    text_color = "#FFFFFF"
                    border = "2px solid #0099FF"
                elif pct_change_value >= 10:
                    bg_color = "#3399FF"  # Sky blue
                    text_color = "#FFFFFF"
                    border = "2px solid #66B2FF"
                elif pct_change_value > 0:
                    bg_color = "#66B2FF"  # Light blue
                    text_color = "#000000"
                    border = "2px solid #99CCFF"
                elif pct_change_value <= -30:
                    bg_color = "#CC5500"  # Dark orange (big decrease)
                    text_color = "#FFFFFF"
                    border = "2px solid #FF6600"
                elif pct_change_value <= -10:
                    bg_color = "#FF8C00"  # Orange
                    text_color = "#FFFFFF"
                    border = "2px solid #FFA500"
                else:
                    bg_color = "#FFD700"  # Gold/Yellow (small decrease)
                    text_color = "#000000"
                    border = "2px solid #FFA500"

                with col:
                    st.markdown(f"""
                        <div style='background-color: {bg_color}; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem; border: {border}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <div style='color: {text_color}; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;'>{week_data['Week']}</div>
                            <div style='color: {text_color}; font-size: 0.9rem; line-height: 1.4; white-space: pre-line;'>{week_data['Delivery']}</div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Format for display
        display_df = allocation.copy()
        display_df['week'] = display_df['week'].dt.strftime('%Y-%m-%d')
        display_df['budget_pct'] = display_df['budget_pct'].round(2)

        # Round budget columns
        budget_cols = ['awareness_budget', 'consideration_budget', 'conversion_budget', 'shopping_budget', 'total_weekly_budget']
        for col in budget_cols:
            display_df[col] = display_df[col].round(0)

        # Select columns (removed avg_roas, avg_cpa, avg_ctr, avg_cvr)
        display_columns = [
            'week', 'budget_pct', 'awareness_budget', 'consideration_budget',
            'conversion_budget', 'shopping_budget', 'total_weekly_budget'
        ]

        display_df = display_df[display_columns].copy()
        display_df.columns = [
            'Week Start', '% of Budget', 'Awareness ($)', 'Consideration ($)',
            'Conversion ($)', 'Shopping ($)', 'Total ($)'
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_week = display_df.loc[display_df['Total ($)'].idxmax()]
            st.info(f"🔥 **Peak Week:** {peak_week['Week Start']} (${peak_week['Total ($)']:,.0f})")

        with col2:
            avg_weekly = display_df['Total ($)'].mean()
            st.info(f"📊 **Avg Weekly Budget:** ${avg_weekly:,.0f}")

        with col3:
            total_weeks = len(display_df)
            st.info(f"📅 **Campaign Duration:** {total_weeks} weeks")

    with tab2:
        st.subheader("Performance Trends")

        # Budget curve - stacked area
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=allocation['week'],
            y=allocation['awareness_budget'],
            mode='lines',
            name='Awareness',
            line=dict(width=0.5),
            stackgroup='one',
            fillcolor='rgba(102, 126, 234, 0.6)'
        ))

        fig.add_trace(go.Scatter(
            x=allocation['week'],
            y=allocation['consideration_budget'],
            mode='lines',
            name='Consideration',
            line=dict(width=0.5),
            stackgroup='one',
            fillcolor='rgba(118, 75, 162, 0.6)'
        ))

        fig.add_trace(go.Scatter(
            x=allocation['week'],
            y=allocation['conversion_budget'],
            mode='lines',
            name='Conversion',
            line=dict(width=0.5),
            stackgroup='one',
            fillcolor='rgba(230, 0, 35, 0.6)'
        ))

        fig.add_trace(go.Scatter(
            x=allocation['week'],
            y=allocation['shopping_budget'],
            mode='lines',
            name='Shopping',
            line=dict(width=0.5),
            stackgroup='one',
            fillcolor='rgba(255, 152, 0, 0.6)'
        ))

        fig.update_layout(
            title='Weekly Budget Allocation Over Holiday Season',
            xaxis_title='Week',
            yaxis_title='Budget ($)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # ROAS trend
        st.markdown("---")
        st.markdown("#### ROAS Performance by Week")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=allocation['week'],
            y=allocation['avg_roas'],
            mode='lines+markers',
            name='Avg ROAS',
            line=dict(color='#E60023', width=3),
            marker=dict(size=10)
        ))

        fig2.update_layout(
            xaxis_title='Week',
            yaxis_title='ROAS',
            height=400,
            template='plotly_white'
        )

        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Geographic Analysis")

        # Performance by country (using median)
        country_perf = filtered_df.groupby('user_country').agg({
            'ROAS': 'median',
            'CPA': 'median',
            'CTR (%)': 'median',
            'CVR (%)': 'median',
            'AOV': 'median'
        }).reset_index()

        country_perf = country_perf.sort_values('ROAS', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Countries by ROAS")

            fig = px.bar(
                country_perf.head(10),
                x='user_country',
                y='ROAS',
                text='ROAS',
                color='ROAS',
                color_continuous_scale='Reds'
            )

            fig.update_traces(
                texttemplate='%{text:.2f}x',
                textposition='outside'
            )

            fig.update_layout(
                xaxis_title='Country',
                yaxis_title='Median ROAS',
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Performance Metrics by Country")

            display_country = country_perf.head(10).copy()
            display_country['ROAS'] = display_country['ROAS'].round(2)
            display_country['CPA'] = display_country['CPA'].round(2)
            display_country['CTR (%)'] = display_country['CTR (%)'].round(3)
            display_country['CVR (%)'] = display_country['CVR (%)'].round(3)
            display_country['AOV'] = display_country['AOV'].round(2)

            st.dataframe(
                display_country,
                use_container_width=True,
                height=400
            )

        # Vertical performance
        st.markdown("---")
        st.markdown("#### Performance by Vertical")

        vertical_perf = filtered_df.groupby('sfdc_vertical').agg({
            'ROAS': 'median',
            'CPA': 'median',
            'CTR (%)': 'median',
            'CVR (%)': 'median'
        }).reset_index()

        vertical_perf = vertical_perf.sort_values('ROAS', ascending=False)

        fig3 = px.bar(
            vertical_perf,
            x='sfdc_vertical',
            y='ROAS',
            text='ROAS',
            color='ROAS',
            color_continuous_scale='Purples'
        )

        fig3.update_traces(
            texttemplate='%{text:.2f}x',
            textposition='outside'
        )

        fig3.update_layout(
            xaxis_title='Vertical',
            yaxis_title='Median ROAS',
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Budget Summary by Objective")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Total Budget by Objective")

            objective_totals = pd.DataFrame({
                'Objective': ['Awareness', 'Consideration', 'Conversion', 'Shopping'],
                'Budget': [awareness_budget, consideration_budget, conversion_budget, shopping_budget]
            })

            fig = px.pie(
                objective_totals,
                values='Budget',
                names='Objective',
                color='Objective',
                color_discrete_map={
                    'Awareness': '#667eea',
                    'Consideration': '#764ba2',
                    'Conversion': '#E60023',
                    'Shopping': '#ff9800'
                },
                hole=0.4
            )

            fig.update_traces(
                textposition='inside',
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Budget: $%{value:,.0f}<extra></extra>'
            )

            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Weekly Average by Objective")

            avg_weekly = pd.DataFrame({
                'Objective': ['Awareness', 'Consideration', 'Conversion', 'Shopping'],
                'Avg Weekly Budget': [
                    allocation['awareness_budget'].mean(),
                    allocation['consideration_budget'].mean(),
                    allocation['conversion_budget'].mean(),
                    allocation['shopping_budget'].mean()
                ]
            })

            fig2 = px.bar(
                avg_weekly,
                x='Objective',
                y='Avg Weekly Budget',
                text='Avg Weekly Budget',
                color='Objective',
                color_discrete_map={
                    'Awareness': '#667eea',
                    'Consideration': '#764ba2',
                    'Conversion': '#E60023',
                    'Shopping': '#ff9800'
                }
            )

            fig2.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            )

            fig2.update_layout(
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.markdown("---")
        st.markdown("#### Campaign Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Budget", f"${total_budget:,.0f}")

        with col2:
            st.metric("Number of Weeks", f"{len(allocation)}")

        with col3:
            st.metric("Peak Week Budget", f"${allocation['total_weekly_budget'].max():,.0f}")

        with col4:
            st.metric("Avg Weekly Budget", f"${allocation['total_weekly_budget'].mean():,.0f}")

        # Data quality info
        st.markdown("---")
        st.markdown("#### Data Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"📊 **Total Records:** {len(filtered_df):,}")

        with col2:
            st.info(f"🌍 **Countries:** {filtered_df['user_country'].nunique()}")

        with col3:
            st.info(f"📅 **Date Range:** {filtered_df['day'].min().strftime('%Y-%m-%d')} to {filtered_df['day'].max().strftime('%Y-%m-%d')}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Holiday 2026/2027 ROAS Forecast Dashboard</strong></p>
            <p>Forecast based on 2025/2026 historical performance data | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
