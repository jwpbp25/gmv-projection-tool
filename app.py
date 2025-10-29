import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import anthropic
import os
import json
import re
from datetime import datetime

# Page config
st.set_page_config(
    page_title="GMV Scenario Modeler",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Load and prepare data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("gmv_data.csv")
    return df

df = load_data()

feature_cols = ['makers', 'active_skus', 'orders', 'units',
                'total_discounted_skus',
                'discount_skus_10_to_20_percent', 'discount_days_10_to_20_percent',
                'discount_skus_30_to_40_percent', 'discount_days_30_to_40_percent']

X = df[feature_cols]
y = df['gmv']

# ======================
# Hybrid Model Class
# ======================
class HybridGMVModel:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.linear_model.fit(X, y)
        
        self.ml_model = XGBRegressor(n_estimators=200, random_state=42, learning_rate=0.1)
        self.ml_model.fit(X, y)
        
        self.X_min = X.min()
        self.X_max = X.max()
        self.feature_names = X.columns
        
        self.calculate_unit_economics()
    
    def calculate_unit_economics(self):
        self.unit_impacts = {}
        for i, feature in enumerate(self.feature_names):
            self.unit_impacts[feature] = self.linear_model.coef_[i]
    
    def predict(self, input_data):
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        
        out_of_range = {}
        for col in self.feature_names:
            if input_df[col].values[0] > self.X_max[col]:
                out_of_range[col] = input_df[col].values[0] - self.X_max[col]
        
        if len(out_of_range) <= 2:
            clipped_input = input_df.copy()
            for col in self.feature_names:
                clipped_input[col] = clipped_input[col].clip(upper=self.X_max[col] * 1.5)
            base_prediction = self.ml_model.predict(clipped_input)[0]
            
            adjustment = 0
            for col, excess in out_of_range.items():
                adjustment += self.unit_impacts[col] * excess
            
            return base_prediction + adjustment
        else:
            return self.linear_model.predict(input_df)[0]

# ======================
# Train Models and Calculate Metrics
# ======================
@st.cache_resource
def train_models():
    hybrid_model = HybridGMVModel()
    
    models = {
        "Hybrid (Recommended)": hybrid_model,
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42, learning_rate=0.1),
        "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, learning_rate=0.1, verbose=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }
    
    for name, model in models.items():
        if name != "Hybrid (Recommended)":
            model.fit(X, y)
    
    # Calculate metrics for all models
    metrics = {}
    for name, model in models.items():
        if name == "Hybrid (Recommended)":
            y_pred = np.array([model.predict(X.iloc[i].values) for i in range(len(X))])
        else:
            y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        metrics[name] = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    return models, metrics

models, model_metrics = train_models()

# ======================
# Calculate Baseline Values from Data Averages
# ======================
BASELINE_VALUES = {
    'makers': int(df['makers'].mean()),
    'active_skus': int(df['active_skus'].mean()),
    'orders': int(df['orders'].mean()),
    'units': int(df['units'].mean()),
    'discounted_skus': int(df['total_discounted_skus'].mean()),
    'discount_skus_10_to_20': int(df['discount_skus_10_to_20_percent'].mean()),
    'discount_days_10_to_20': int(df['discount_days_10_to_20_percent'].mean()),
    'discount_skus_30_to_40': int(df['discount_skus_30_to_40_percent'].mean()),
    'discount_days_30_to_40': int(df['discount_days_30_to_40_percent'].mean())
}

# ======================
# LLM Chat Function
# ======================
def parse_chat_query(user_message, current_values, baseline_values):
    """Use Claude to interpret user query and suggest parameter changes"""
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {
            'success': False,
            'message': '⚠️ Please set ANTHROPIC_API_KEY environment variable to use chat features.',
            'params': current_values
        }
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""You are a retail analytics assistant helping with GMV scenario modeling.

BASELINE VALUES (use these as reference for relative changes):
- Makers: {baseline_values['makers']}
- Active SKUs: {baseline_values['active_skus']}
- Orders: {baseline_values['orders']}
- Units: {baseline_values['units']}
- Total Discounted SKUs: {baseline_values['discounted_skus']}
- Discount SKUs (10-20%): {baseline_values['discount_skus_10_to_20']}
- Discount Days (10-20%): {baseline_values['discount_days_10_to_20']}
- Discount SKUs (30-40%): {baseline_values['discount_skus_30_to_40']}
- Discount Days (30-40%): {baseline_values['discount_days_30_to_40']}

CURRENT VALUES:
- Makers: {current_values['makers']}
- Active SKUs: {current_values['active_skus']}
- Orders: {current_values['orders']}
- Units: {current_values['units']}
- Total Discounted SKUs: {current_values['discounted_skus']}
- Discount SKUs (10-20%): {current_values['discount_skus_10_to_20']}
- Discount Days (10-20%): {current_values['discount_days_10_to_20']}
- Discount SKUs (30-40%): {current_values['discount_skus_30_to_40']}
- Discount Days (30-40%): {current_values['discount_days_30_to_40']}

Valid ranges:
- Makers: 5-500
- Active SKUs: 10-10,000
- Orders: 1-10,000
- Units: 1-10,000
- Total Discounted SKUs: 0-10,000
- Discount SKUs: 0-10,000
- Discount Days: 0-31

User query: "{user_message}"

IMPORTANT: When user says "double" or uses percentages, apply them to the BASELINE VALUES, not current values.
For example, "double the SKUs" means active_skus should be {baseline_values['active_skus'] * 2}.

Interpret the user's request and provide updated parameter values (only change what user mentions), a brief explanation, and business insights.

Respond in JSON format:
{{
  "params": {{
    "makers": number,
    "active_skus": number,
    "orders": number,
    "units": number,
    "discounted_skus": number,
    "discount_skus_10_to_20": number,
    "discount_days_10_to_20": number,
    "discount_skus_30_to_40": number,
    "discount_days_30_to_40": number
  }},
  "explanation": "What changed and why",
  "insight": "Business recommendation or insight"
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'success': True,
                'params': result.get('params', current_values),
                'explanation': result.get('explanation', ''),
                'insight': result.get('insight', ''),
                'message': f"✅ {result.get('explanation', '')}\n\n💡 {result.get('insight', '')}"
            }
        else:
            return {
                'success': False,
                'message': '❌ Could not parse response. Please try rephrasing.',
                'params': current_values
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f'❌ Error: {str(e)}',
            'params': current_values
        }

# ======================
# Initialize Session State
# ======================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'params' not in st.session_state:
    st.session_state.params = BASELINE_VALUES.copy()

# ======================
# UI Layout
# ======================
st.markdown('<div class="main-header">🛍️ GMV Scenario Modeler with AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict your Gross Merchandise Value using machine learning and natural language</div>', unsafe_allow_html=True)

# ======================
# Sidebar - Model Selection
# ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox(
        "🤖 Select Model",
        ["Hybrid (Recommended)", "XGBoost", "LightGBM", "Gradient Boosting", 
         "Random Forest", "Ridge Regression", "Linear Regression"],
        help="Hybrid model is best for extrapolation"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Average Baseline Values")
    st.caption("*(Calculated from training data)*")
    st.caption(f"**Makers:** {BASELINE_VALUES['makers']}")
    st.caption(f"**Active SKUs:** {BASELINE_VALUES['active_skus']}")
    st.caption(f"**Orders:** {BASELINE_VALUES['orders']}")
    st.caption(f"**Units:** {BASELINE_VALUES['units']}")
    st.caption(f"**Discounted SKUs:** {BASELINE_VALUES['discounted_skus']}")
    
    st.markdown("---")
    st.markdown("### 📊 Training Data Ranges")
    st.caption(f"**Active SKUs:** {X['active_skus'].min():.0f} - {X['active_skus'].max():.0f}")
    st.caption(f"**Orders:** {X['orders'].min():.0f} - {X['orders'].max():.0f}")
    st.caption(f"**Units:** {X['units'].min():.0f} - {X['units'].max():.0f}")
    
    if model_name == "Hybrid (Recommended)":
        st.info("✅ Uses ML for in-range predictions and linear extrapolation for out-of-range values.")
    elif model_name in ["Linear Regression", "Ridge Regression"]:
        st.info("📈 Good for extrapolation but may miss complex patterns.")
    else:
        st.warning("⚠️ Not recommended for extrapolation beyond training range.")
    
    # Show model equation/details
    st.markdown("---")
    st.markdown("### 📐 Model Equation")
    
    if model_name in ["Linear Regression", "Ridge Regression"]:
        model = models[model_name]
        intercept = model.intercept_
        coefficients = model.coef_
        
        if model_name == "Ridge Regression":
            st.markdown("**GMV = Intercept + Σ(Coefficient × Feature)**")
            st.caption("*(With L2 regularization to prevent overfitting)*")
        else:
            st.markdown("**GMV = Intercept + Σ(Coefficient × Feature)**")
        
        st.caption(f"Intercept: ${intercept:,.2f}")
        st.caption("**Coefficients:**")
        for i, col in enumerate(feature_cols):
            coef_sign = "+" if coefficients[i] >= 0 else ""
            st.caption(f"  • {col}: {coef_sign}{coefficients[i]:,.2f}")
    
    elif model_name == "Hybrid (Recommended)":
        hybrid = models[model_name]
        intercept = hybrid.linear_model.intercept_
        coefficients = hybrid.linear_model.coef_
        
        st.markdown("**If extrapolating:**")
        st.caption("GMV = XGBoost(clipped) + Linear adjustments")
        st.markdown("**If interpolating:**")
        st.caption("GMV = XGBoost prediction")
        
        with st.expander("View Linear Coefficients"):
            st.caption(f"Intercept: ${intercept:,.2f}")
            for i, col in enumerate(feature_cols):
                coef_sign = "+" if coefficients[i] >= 0 else ""
                st.caption(f"  • {col}: {coef_sign}{coefficients[i]:,.2f}")
    
    elif model_name in ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]:
        model = models[model_name]
        
        st.markdown("**Ensemble Model**")
        st.caption(f"Uses {model.n_estimators} decision trees")
        
        if model_name == "LightGBM":
            st.caption("*(Optimized for speed and efficiency)*")
        elif model_name == "XGBoost":
            st.caption("*(Extreme Gradient Boosting)*")
        elif model_name == "Gradient Boosting":
            st.caption("*(Sequential tree building)*")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.caption("**Feature Importances:**")
            for _, row in importance_df.iterrows():
                st.caption(f"  • {row['Feature']}: {row['Importance']*100:.1f}%")

# ======================
# Create Tabs
# ======================
tab1, tab2 = st.tabs(["💬 AI Chat", "🎛️ Manual Controls"])

with tab1:
    # ======================
    # AI Chat Interface
    # ======================
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("💬 AI Assistant")
    
    with st.expander("ℹ️ What can I ask?", expanded=False):
        st.markdown(f"""
        - **"Double the SKUs"** (uses average baseline of {BASELINE_VALUES['active_skus']} → {BASELINE_VALUES['active_skus']*2})
        - **"Set active SKUs to 5000"**
        - **"Increase orders by 50%"** (uses average baseline of {BASELINE_VALUES['orders']})
        - **"Show me a high-discount scenario"**
        - **"How do I reach $1M GMV?"**
        - **"Reset to average"**
        """)
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your GMV scenario...")
    
    if user_input:
        current_values = st.session_state.params.copy()
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        result = parse_chat_query(user_input, current_values, BASELINE_VALUES)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result['message']
        })
        
        if result['success']:
            st.session_state.params = result['params']
        
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show current prediction in chat tab
    st.markdown("---")
    st.header("💵 Current Prediction")
    
    # Get current parameters
    params = st.session_state.params
    input_data = np.array([[params['makers'], params['active_skus'], params['orders'], 
                           params['units'], params['discounted_skus'],
                           params['discount_skus_10_to_20'], params['discount_days_10_to_20'],
                           params['discount_skus_30_to_40'], params['discount_days_30_to_40']]])
    
    model = models[model_name]
    if model_name == "Hybrid (Recommended)":
        predicted_gmv = model.predict(input_data[0])
    else:
        predicted_gmv = model.predict(input_data)[0]
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${predicted_gmv:,.2f}</div></div>', unsafe_allow_html=True)
    
    with col_pred2:
        st.markdown("**Current Parameters:**")
        st.caption(f"Makers: {params['makers']}")
        st.caption(f"Active SKUs: {params['active_skus']}")
        st.caption(f"Orders: {params['orders']}")
        st.caption(f"Units: {params['units']}")

with tab2:
    # ======================
    # Main Content - Input Controls
    # ======================
    st.header("📊 Scenario Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Business Metrics")
        makers = st.number_input("👥 Makers", min_value=5, max_value=500, value=st.session_state.params['makers'], step=1)
        active_skus = st.number_input("📦 Active SKUs", min_value=10, max_value=10000, value=st.session_state.params['active_skus'], step=10)
        orders = st.number_input("🛒 Orders", min_value=1, max_value=10000, value=st.session_state.params['orders'], step=10)
        units = st.number_input("📊 Units", min_value=1, max_value=10000, value=st.session_state.params['units'], step=10)
        discounted_skus = st.number_input("💰 Total Discounted SKUs", min_value=0, max_value=10000, value=st.session_state.params['discounted_skus'], step=10)
    
    with col2:
        st.markdown("### 🎯 10-20% Discount Range")
        discount_skus_10_to_20 = st.number_input("🏷️ Discount SKUs", min_value=0, max_value=10000, value=st.session_state.params['discount_skus_10_to_20'], step=10, key="d10_20_skus")
        discount_days_10_to_20 = st.number_input("📅 Discount Days", min_value=0, max_value=31, value=st.session_state.params['discount_days_10_to_20'], step=1, key="d10_20_days")
        
        st.markdown("### 🎯 30-40% Discount Range")
        discount_skus_30_to_40 = st.number_input("🏷️ Discount SKUs", min_value=0, max_value=10000, value=st.session_state.params['discount_skus_30_to_40'], step=10, key="d30_40_skus")
        discount_days_30_to_40 = st.number_input("📅 Discount Days", min_value=0, max_value=31, value=st.session_state.params['discount_days_30_to_40'], step=1, key="d30_40_days")
    
    # Update session state with input values
    st.session_state.params = {
        'makers': makers,
        'active_skus': active_skus,
        'orders': orders,
        'units': units,
        'discounted_skus': discounted_skus,
        'discount_skus_10_to_20': discount_skus_10_to_20,
        'discount_days_10_to_20': discount_days_10_to_20,
        'discount_skus_30_to_40': discount_skus_30_to_40,
        'discount_days_30_to_40': discount_days_30_to_40
    }
    
    # ======================
    # Prediction
    # ======================
    model = models[model_name]
    
    input_data = np.array([[makers, active_skus, orders, units, discounted_skus,
                            discount_skus_10_to_20, discount_days_10_to_20,
                            discount_skus_30_to_40, discount_days_30_to_40]])
    
    # Make prediction
    if model_name == "Hybrid (Recommended)":
        predicted_gmv = model.predict(input_data[0])
    else:
        predicted_gmv = model.predict(input_data)[0]
    
    # Check extrapolation
    input_df = pd.DataFrame(input_data, columns=X.columns)
    extrapolating = False
    out_of_range_features = []
    
    for col in X.columns:
        if input_df[col].values[0] > X[col].max():
            extrapolating = True
            out_of_range_features.append(f"{col}")
    
    # ======================
    # Display Results
    # ======================
    with col3:
        st.markdown("### 💵 Predicted GMV")
        st.markdown(f'<div class="metric-card"><div class="metric-value">${predicted_gmv:,.2f}</div></div>', unsafe_allow_html=True)
        
        if extrapolating:
            st.warning(f"⚠️ **Extrapolating** beyond training data: {', '.join(out_of_range_features)}")
        else:
            st.success("✅ **Interpolating** within training data range")

# ======================
# Visualization (Shared across both tabs)
# ======================
st.markdown("---")
st.header("📊 Model Performance")

# Generate predictions for visualization
if model_name == "Hybrid (Recommended)":
    y_pred = np.array([models[model_name].predict(X.iloc[i].values) for i in range(len(X))])
else:
    y_pred = models[model_name].predict(X)

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=y, 
    mode='lines+markers', 
    name='Actual GMV',
    line=dict(color='#3498db', width=3),
    marker=dict(size=6)
))
fig.add_trace(go.Scatter(
    y=y_pred, 
    mode='lines+markers', 
    name=f'Predicted GMV ({model_name})',
    line=dict(color='#e74c3c', width=3, dash='dash'),
    marker=dict(size=6)
))

fig.update_layout(
    title=f"Actual vs Predicted GMV - {model_name}",
    xaxis_title="Data Point",
    yaxis_title="GMV ($)",
    template="plotly_white",
    height=500,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# Model Performance Metrics Table
# ======================
st.markdown("### 📈 Model Performance Metrics")

metrics_df = pd.DataFrame({
    'Model': list(model_metrics.keys()),
    'R² Score': [f"{model_metrics[m]['r2']:.4f}" for m in model_metrics.keys()],
    'RMSE': [f"${model_metrics[m]['rmse']:,.2f}" for m in model_metrics.keys()],
    'MSE': [f"${model_metrics[m]['mse']:,.2f}" for m in model_metrics.keys()],
    'MAE': [f"${model_metrics[m]['mae']:,.2f}" for m in model_metrics.keys()]
})

# Highlight the selected model
def highlight_selected(row):
    if row['Model'] == model_name:
        return ['background-color: #e3f2fd'] * len(row)
    return [''] * len(row)

st.dataframe(
    metrics_df.style.apply(highlight_selected, axis=1),
    use_container_width=True,
    hide_index=True
)

st.caption("**R² Score**: Proportion of variance explained (closer to 1.0 is better)")
st.caption("**RMSE**: Root Mean Squared Error (lower is better)")
st.caption("**MSE**: Mean Squared Error (lower is better)")
st.caption("**MAE**: Mean Absolute Error (lower is better)")

# ======================
# Footer
# ======================
st.markdown("---")
st.markdown("### 🚀 Quick Actions")
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔄 Reset to Averages"):
        st.session_state.params = BASELINE_VALUES.copy()
        st.rerun()

with col_b:
    # Create CSV data
    scenario_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model': [model_name],
        'makers': [st.session_state.params['makers']],
        'active_skus': [st.session_state.params['active_skus']],
        'orders': [st.session_state.params['orders']],
        'units': [st.session_state.params['units']],
        'discounted_skus': [st.session_state.params['discounted_skus']],
        'discount_skus_10_to_20': [st.session_state.params['discount_skus_10_to_20']],
        'discount_days_10_to_20': [st.session_state.params['discount_days_10_to_20']],
        'discount_skus_30_to_40': [st.session_state.params['discount_skus_30_to_40']],
        'discount_days_30_to_40': [st.session_state.params['discount_days_30_to_40']],
        'predicted_gmv': [float(predicted_gmv)],
        'r2_score': [model_metrics[model_name]['r2']],
        'rmse': [model_metrics[model_name]['rmse']],
        'mse': [model_metrics[model_name]['mse']]
    }
    
    scenario_df = pd.DataFrame(scenario_data)
    csv_data = scenario_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Export Scenario (CSV)",
        data=csv_data,
        file_name=f"gmv_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col_c:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
