import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="E-Commerce Revenue Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS - Premium Dark Theme
# =============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.35);
        animation: pulse-glow 2s ease-in-out infinite alternate;
    }
    @keyframes pulse-glow {
        from { box-shadow: 0 10px 40px rgba(102, 126, 234, 0.35); }
        to { box-shadow: 0 10px 50px rgba(118, 75, 162, 0.5); }
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .prediction-label {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #a5b4fc;
    }

    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .info-box p {
        color: #a5b4fc;
        margin: 0;
        font-size: 0.9rem;
    }

    /* Comparison table */
    .model-badge-best {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================
# LOAD MODEL & DATA
# =============================================
@st.cache_resource
def load_model_artifacts():
    """Load all saved model artifacts."""
    model = joblib.load('revenue_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, scaler, label_encoders, feature_cols, metadata

@st.cache_data
def load_data():
    """Load the processed dataset."""
    return pd.read_csv('processed_data.csv')

# Check if model files exist
required_files = ['revenue_model.pkl', 'scaler.pkl', 'label_encoders.pkl',
                  'feature_columns.pkl', 'model_metadata.pkl', 'processed_data.csv']
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"⚠️ Missing model files: {', '.join(missing)}")
    st.info("👉 Run `python train_model.py` first to generate the model artifacts.")
    st.stop()

model, scaler, label_encoders, feature_cols, metadata = load_model_artifacts()
df = load_data()


# =============================================
# HEADER
# =============================================
st.markdown("""
<div class="main-header">
    <h1>🛒 E-Commerce Revenue Predictor</h1>
    <p>AI-powered revenue forecasting using machine learning • Trained on {:,} transactions</p>
</div>
""".format(metadata['data_stats']['total_records']), unsafe_allow_html=True)


# =============================================
# SIDEBAR - Prediction Inputs
# =============================================
with st.sidebar:
    st.markdown("## 🎯 Predict Revenue")
    st.markdown("---")
    
    # Product Category
    product_category = st.selectbox(
        "📦 Product Category",
        options=sorted(metadata['category_values']),
        index=0
    )
    
    # Region
    region = st.selectbox(
        "🌍 Region",
        options=sorted(metadata['region_values']),
        index=0
    )
    
    # Payment Method
    payment_method = st.selectbox(
        "💳 Payment Method",
        options=sorted(metadata['payment_values']),
        index=0
    )
    
    st.markdown("---")
    
    # Quantity
    quantity = st.slider("📊 Quantity", min_value=1, max_value=10, value=3, step=1)
    
    # Unit Price
    unit_price = st.number_input("💰 Unit Price ($)", min_value=10.0, max_value=1000.0,
                                  value=250.0, step=10.0)
    
    # Discount
    discount = st.slider("🏷️ Discount (%)", min_value=0, max_value=50, value=15, step=1)
    discount_decimal = discount / 100
    
    # Delivery Days
    delivery_days = st.slider("🚚 Delivery Days", min_value=1, max_value=15, value=5, step=1)
    
    # Customer Rating
    customer_rating = st.slider("⭐ Customer Rating", min_value=1.0, max_value=5.0,
                                 value=3.5, step=0.1)
    
    st.markdown("---")
    
    # Time features
    order_month = st.selectbox("📅 Month", options=list(range(1, 13)),
                                index=0, format_func=lambda x: 
                                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
    
    order_quarter = (order_month - 1) // 3 + 1
    order_day_of_week = st.selectbox("📅 Day of Week", options=list(range(7)),
                                      format_func=lambda x: 
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                       'Friday', 'Saturday', 'Sunday'][x])
    
    # Predict button
    st.markdown("---")
    predict_clicked = st.button("🚀 Predict Revenue", use_container_width=True, type="primary")


# =============================================
# MAIN CONTENT - Tabs
# =============================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Prediction", "📈 Model Performance", "🔍 Data Explorer"])

# =============================================
# TAB 1: DASHBOARD
# =============================================
with tab1:
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metadata['data_stats']['total_records']:,}</p>
            <p class="metric-label">Total Orders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_rev = df['revenue'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${total_rev/1e6:.1f}M</p>
            <p class="metric-label">Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_rev = df['revenue'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${avg_rev:,.0f}</p>
            <p class="metric-label">Avg Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = df['customer_rating'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_rating:.1f}⭐</p>
            <p class="metric-label">Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        model_r2 = metadata['metrics']['R2']
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{model_r2:.1%}</p>
            <p class="metric-label">Model Accuracy (R²)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by Category
        cat_rev = df.groupby('product_category')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
        fig = px.bar(cat_rev, x='product_category', y='mean',
                     color='product_category',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                     title='Average Revenue by Product Category',
                     labels={'mean': 'Average Revenue ($)', 'product_category': 'Category'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by Region
        region_rev = df.groupby('region')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
        fig = px.bar(region_rev, x='region', y='mean',
                     color='region',
                     color_discrete_sequence=['#34d399', '#10b981', '#059669', '#047857'],
                     title='Average Revenue by Region',
                     labels={'mean': 'Average Revenue ($)', 'region': 'Region'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment Method Distribution
        payment_dist = df['payment_method'].value_counts().reset_index()
        payment_dist.columns = ['payment_method', 'count']
        fig = px.pie(payment_dist, values='count', names='payment_method',
                     title='Payment Method Distribution',
                     color_discrete_sequence=['#667eea', '#f43f5e', '#f59e0b'],
                     hole=0.45)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue Distribution
        fig = px.histogram(df, x='revenue', nbins=50,
                           title='Revenue Distribution',
                           color_discrete_sequence=['#667eea'],
                           labels={'revenue': 'Revenue ($)', 'count': 'Frequency'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        # Discount vs Revenue scatter
        fig = px.scatter(df.sample(min(1000, len(df)), random_state=42),
                         x='discount', y='revenue',
                         color='product_category',
                         title='Discount vs Revenue',
                         color_discrete_sequence=['#667eea', '#f43f5e', '#10b981', '#f59e0b'],
                         opacity=0.6,
                         labels={'discount': 'Discount (%)', 'revenue': 'Revenue ($)'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly revenue trend
        if 'order_month' in df.columns:
            monthly = df.groupby('order_month')['revenue'].mean().reset_index()
            fig = px.line(monthly, x='order_month', y='revenue',
                          title='Average Revenue by Month',
                          markers=True,
                          color_discrete_sequence=['#667eea'],
                          labels={'order_month': 'Month', 'revenue': 'Avg Revenue ($)'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                title_font_size=16
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)',
                            tickvals=list(range(1, 13)),
                            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig, use_container_width=True)

# =============================================
# TAB 2: PREDICTION
# =============================================
with tab2:
    st.markdown('<p class="section-header">🔮 Revenue Prediction Engine</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>👈 Configure your order parameters in the sidebar, then click <b>Predict Revenue</b> to get an AI-powered estimate.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current input summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📦 Order Details**")
        st.write(f"- Category: **{product_category}**")
        st.write(f"- Quantity: **{quantity}**")
        st.write(f"- Unit Price: **${unit_price:.2f}**")
    with col2:
        st.markdown("**🏷️ Pricing & Delivery**")
        st.write(f"- Discount: **{discount}%**")
        st.write(f"- Delivery Days: **{delivery_days}**")
        st.write(f"- Customer Rating: **{customer_rating}⭐**")
    with col3:
        st.markdown("**🌍 Location & Payment**")
        st.write(f"- Region: **{region}**")
        st.write(f"- Payment: **{payment_method}**")
        st.write(f"- Month: **{order_month}** | Quarter: **Q{order_quarter}**")
    
    st.markdown("---")
    
    if predict_clicked:
        # Prepare input data
        input_data = {
            'quantity': quantity,
            'unit_price': unit_price,
            'discount': discount_decimal,
            'delivery_days': delivery_days,
            'customer_rating': customer_rating,
            'order_month': order_month,
            'order_quarter': order_quarter,
            'order_day_of_week': order_day_of_week,
        }
        
        # Encode categorical variables
        for col in ['product_category', 'region', 'payment_method']:
            le = label_encoders[col]
            val = {'product_category': product_category, 'region': region, 'payment_method': payment_method}[col]
            input_data[f'{col}_encoded'] = le.transform([val])[0]
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_cols]
        
        # Make prediction (use appropriate input based on model type)
        model_name = metadata['best_model_name']
        if model_name == 'Linear Regression':
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <p class="prediction-label">Predicted Revenue</p>
            <p class="prediction-value">${prediction:,.2f}</p>
            <p class="prediction-label">Model: {model_name} • R² Score: {metadata['metrics']['R2']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison with averages
        st.markdown("#### 📊 How does this compare?")
        col1, col2, col3 = st.columns(3)
        
        overall_avg = df['revenue'].mean()
        cat_avg = df[df['product_category'] == product_category]['revenue'].mean()
        region_avg = df[df['region'] == region]['revenue'].mean()
        
        with col1:
            diff_overall = ((prediction - overall_avg) / overall_avg) * 100
            emoji = "📈" if diff_overall > 0 else "📉"
            st.metric("vs Overall Average", f"${overall_avg:,.2f}", f"{diff_overall:+.1f}% {emoji}")
        
        with col2:
            diff_cat = ((prediction - cat_avg) / cat_avg) * 100
            emoji = "📈" if diff_cat > 0 else "📉"
            st.metric(f"vs {product_category} Average", f"${cat_avg:,.2f}", f"{diff_cat:+.1f}% {emoji}")
        
        with col3:
            diff_region = ((prediction - region_avg) / region_avg) * 100
            emoji = "📈" if diff_region > 0 else "📉"
            st.metric(f"vs {region} Average", f"${region_avg:,.2f}", f"{diff_region:+.1f}% {emoji}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.4);">
            <p style="font-size: 3rem; margin-bottom: 1rem;">🔮</p>
            <p style="font-size: 1.1rem;">Configure parameters in the sidebar and click <b>Predict Revenue</b></p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# TAB 3: MODEL PERFORMANCE
# =============================================
with tab3:
    st.markdown('<p class="section-header">📈 Model Performance & Metrics</p>', unsafe_allow_html=True)
    
    # Best model highlight
    st.success(f"🏆 **Best Model: {metadata['best_model_name']}** — R² Score: {metadata['metrics']['R2']:.4f}")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metadata['metrics']['R2']:.4f}</p>
            <p class="metric-label">R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${metadata['metrics']['MAE']:,.0f}</p>
            <p class="metric-label">Mean Abs Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${metadata['metrics']['RMSE']:,.0f}</p>
            <p class="metric-label">Root Mean Sq Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metadata['metrics']['CV_R2_mean']:.4f}</p>
            <p class="metric-label">Cross-Val R² (5-fold)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model comparison table
    st.markdown("#### 🔄 All Models Comparison")
    comparison_data = []
    for name, metrics in metadata['all_model_results'].items():
        is_best = "🏆" if name == metadata['best_model_name'] else ""
        comparison_data.append({
            '': is_best,
            'Model': name,
            'R² Score': f"{metrics['R2']:.4f}",
            'MAE ($)': f"${metrics['MAE']:,.2f}",
            'RMSE ($)': f"${metrics['RMSE']:,.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Model comparison chart
    model_names = list(metadata['all_model_results'].keys())
    r2_scores = [metadata['all_model_results'][m]['R2'] for m in model_names]
    mae_scores = [metadata['all_model_results'][m]['MAE'] for m in model_names]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=r2_scores,
                   marker_color=['#667eea' if n == metadata['best_model_name'] else '#4a4a6a' 
                                for n in model_names],
                   text=[f'{s:.4f}' for s in r2_scores],
                   textposition='outside')
        ])
        fig.update_layout(
            title='R² Score Comparison',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            yaxis_range=[0, max(r2_scores) * 1.15],
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=mae_scores,
                   marker_color=['#f43f5e' if n == metadata['best_model_name'] else '#4a4a6a'
                                for n in model_names],
                   text=[f'${s:,.0f}' for s in mae_scores],
                   textposition='outside')
        ])
        fig.update_layout(
            title='MAE Comparison (lower is better)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            yaxis_range=[0, max(mae_scores) * 1.15],
            title_font_size=16
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available from model)
    st.markdown("#### 🎯 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance',
                     color_continuous_scale=['#4a4a6a', '#667eea', '#764ba2'],
                     title=f'Feature Importance ({metadata["best_model_name"]})')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            coloraxis_showscale=False,
            title_font_size=16,
            height=450
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for the selected model type.")
    
    # Training details
    st.markdown("#### ℹ️ Training Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Training samples:** {metadata['data_stats']['train_size']:,}")
        st.write(f"- **Testing samples:** {metadata['data_stats']['test_size']:,}")
        st.write(f"- **Total features:** {len(feature_cols)}")
    with col2:
        st.write(f"- **Revenue Mean:** ${metadata['data_stats']['revenue_mean']:,.2f}")
        st.write(f"- **Revenue Std Dev:** ${metadata['data_stats']['revenue_std']:,.2f}")
        st.write(f"- **CV R² Std:** ±{metadata['metrics']['CV_R2_std']:.4f}")

# =============================================
# TAB 4: DATA EXPLORER
# =============================================
with tab4:
    st.markdown('<p class="section-header">🔍 Interactive Data Explorer</p>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_category = st.multiselect("Filter by Category",
                                          options=sorted(df['product_category'].unique()),
                                          default=sorted(df['product_category'].unique()))
    with col2:
        filter_region = st.multiselect("Filter by Region",
                                        options=sorted(df['region'].unique()),
                                        default=sorted(df['region'].unique()))
    with col3:
        revenue_range = st.slider("Revenue Range ($)",
                                   min_value=float(df['revenue'].min()),
                                   max_value=float(df['revenue'].max()),
                                   value=(float(df['revenue'].min()), float(df['revenue'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['product_category'].isin(filter_category)) &
        (df['region'].isin(filter_region)) &
        (df['revenue'] >= revenue_range[0]) &
        (df['revenue'] <= revenue_range[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
    
    # Display columns to show
    display_cols = ['order_id', 'order_date', 'product_category', 'region', 'quantity',
                    'unit_price', 'discount', 'payment_method', 'delivery_days',
                    'customer_rating', 'revenue']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(filtered_df[available_cols].head(500), use_container_width=True, hide_index=True)
    
    # Download filtered data
    csv = filtered_df[available_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_ecommerce_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Summary statistics for filtered data
    st.markdown("#### 📊 Filtered Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(filtered_df[['revenue', 'quantity', 'unit_price', 'discount',
                               'delivery_days', 'customer_rating']].describe().round(2))
    with col2:
        # Filtered revenue by category
        if len(filter_category) > 0:
            fig = px.box(filtered_df, x='product_category', y='revenue',
                         color='product_category',
                         color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                         title='Revenue Distribution by Category')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                showlegend=False,
                title_font_size=16
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig, use_container_width=True)
