import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pipeline_utils
import os


# Setup page config
st.set_page_config(
    page_title="SafeClinical - Leakage-Free Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL - read from secrets or environment, default to local for development
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))

# Inject Custom CSS for Premium Slate/Glassmorphism Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8da2fb;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(25, 30, 45, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .glass-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Metrics panel styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-sec {
        color: #00f2fe;
    }
    .metric-leak {
        color: #ff3366;
    }
    
    /* Concept boxes */
    .secure-box {
        background: rgba(0, 242, 254, 0.05);
        border-left: 4px solid #00f2fe;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .leaky-box {
        background: rgba(255, 51, 102, 0.05);
        border-left: 4px solid #ff3366;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    /* Risk Levels */
    .risk-high {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.15) 0%, rgba(255, 102, 102, 0.05) 100%);
        border: 1px solid rgba(255, 51, 102, 0.3);
        color: #ff3366;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .risk-med {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.15) 0%, rgba(255, 200, 100, 0.05) 100%);
        border: 1px solid rgba(255, 165, 0, 0.3);
        color: #ffa500;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.05) 100%);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Step Trace styling */
    .trace-step {
        background: rgba(30, 35, 55, 0.7);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .trace-step-title {
        font-weight: 600;
        color: #8da2fb;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🛡️ SafeClinical Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Medical Diagnostic Pipeline with Zero-Leakage Preprocessing & Validation</div>', unsafe_allow_html=True)

# -----------------
# SIDEBAR CONTROLS
# -----------------
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

estimator = st.sidebar.selectbox(
    "1. Estimator Classifier",
    options=["rf", "gb", "lr"],
    format_func=lambda x: "Random Forest Classifier" if x=="rf" else "Gradient Boosting" if x=="gb" else "Logistic Regression",
    help="Select the classification algorithm."
)

imputer = st.sidebar.selectbox(
    "2. Numerical Imputer Strategy",
    options=["mean", "median", "most_frequent"],
    help="Strategy used to fill missing values (which we artificially inject in 10% of serum sodium and platelet records)."
)

scaler = st.sidebar.selectbox(
    "3. Feature Scaling",
    options=["standard", "minmax"],
    format_func=lambda x: "StandardScaler (μ=0, σ=1)" if x=="standard" else "MinMaxScaler (Range 0-1)",
    help="Scaling method for continuous variables."
)

cv_folds = st.sidebar.slider(
    "4. Cross-Validation Folds (K)",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

test_size = st.sidebar.slider(
    "5. Holdout Test Set Fraction",
    min_value=0.15,
    max_value=0.40,
    value=0.20,
    step=0.05
)

train_button = st.sidebar.button(
    "🚀 Train & Compare Pipelines",
    use_container_width=True,
    type="primary"
)

# -----------------
# TAB LAYOUT SETUP
# -----------------
tab_concept, tab_data, tab_metrics, tab_predict = st.tabs([
    "💡 Data Leakage & Security",
    "📊 Dataset Statistics",
    "📈 Model Comparison",
    "🩺 Patient Risk Calculator"
])

# Get Dataset statistics immediately for other tabs
@st.cache_data
def fetch_dataset_info():
    try:
        response = requests.get(f"{API_URL}/data-info")
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

data_info = fetch_dataset_info()

# ---------------------------------------------
# TAB 1: DATA LEAKAGE & SECURITY CONCEPT
# ---------------------------------------------
with tab_concept:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>Data Leakage: The Silent Model Killer in Medicine</h3>', unsafe_allow_html=True)
    st.markdown("""
    In clinical machine learning, preparing medical data incorrectly leads to **Data Leakage**. This occurs when information from 
    outside the training dataset (specifically, the validation or holdout test folds) is inadvertently used to train the model. 
    The result? A model that achieves seemingly **superb validation accuracy** during training but **fails completely** when deployed on real patients.
    """, unsafe_allow_html=True)
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown('<div class="leaky-box">', unsafe_allow_html=True)
        st.markdown('<h4>🚫 The Naive / Leaky Preprocessing Flow</h4>', unsafe_allow_html=True)
        st.markdown("""
        **What happens:**
        1. Imputation values (e.g. mean) or scaling parameters (e.g. min, max, variance) are computed globally across the **entire dataset**.
        2. Categorical encoders are fit globally, learning all categorical boundaries including rare levels in the test set.
        3. The preprocessed data is then split into training and validation folds.
        
        **Why it leaks:**
        The training data contains embedded statistics (means, scales, class bounds) from the validation fold. The validation fold 
        is no longer "unseen." This artificially inflates cross-validation performance.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_l2:
        st.markdown('<div class="secure-box">', unsafe_allow_html=True)
        st.markdown('<h4>🛡️ The Secure Pipeline Flow</h4>', unsafe_allow_html=True)
        st.markdown("""
        **What happens:**
        1. Preprocessing transformations are wrapped into a Scikit-Learn `Pipeline`.
        2. During Cross-Validation, the preprocessing parameters (imputer mean, scaler variance, one-hot categories) are computed **solely from the training split** of that specific fold.
        3. The transformations are then applied to the validation split using the parameters fit from the training split.
        
        **Why it is secure:**
        It mathematically guarantees that **no information** from the validation fold is visible during model training, mirroring real-life clinical application.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Interactive Preprocessing Flow Visualizer")
    
    # Custom SVG / Flow Diagram
    st.write("Below is a visual representation of how the secure diagnostic pipeline isolates the training phase:")
    st.markdown("""
    <div style="background-color:#0f111a; padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align:center;">
        <span style="background-color:#1e293b; padding:8px 12px; border-radius:6px; color:#ffffff; font-weight:600; border: 1px solid #4facfe;">Raw Patient Input</span>
        <span style="color:#4facfe; font-size:1.5rem;"> ➔ </span>
        <span style="background-color:#1e293b; padding:8px 12px; border-radius:6px; color:#ffffff; font-weight:600; border: 1px solid #4facfe;">Numerical Imputer (Fitted on Train Only)</span>
        <span style="color:#4facfe; font-size:1.5rem;"> ➔ </span>
        <span style="background-color:#1e293b; padding:8px 12px; border-radius:6px; color:#ffffff; font-weight:600; border: 1px solid #4facfe;">Standardizer (Fitted on Train Only)</span>
        <span style="color:#4facfe; font-size:1.5rem;"> ➔ </span>
        <span style="background-color:#1e293b; padding:8px 12px; border-radius:6px; color:#ffffff; font-weight:600; border: 1px solid #4facfe;">Classifier Estimator</span>
        <span style="color:#00f2fe; font-size:1.5rem;"> ➔ </span>
        <span style="background-color:#0f2e30; padding:8px 12px; border-radius:6px; color:#00f2fe; font-weight:600; border: 1px solid #00f2fe;">Safe Risk Score (%)</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------
# TAB 2: DATASET STATISTICS
# ---------------------------------------------
with tab_data:
    if data_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card-header">📊 Heart Failure Clinical Records Overview</div>', unsafe_allow_html=True)
        
        col_d1, col_d2, col_d3 = st.columns([1, 1.2, 1.8])
        
        with col_d1:
            st.metric("Total Records", data_info["total_records"])
            st.metric("Features Analyzed", len(data_info["columns"]) - 1)
            
            # Target class distribution chart
            dist_data = data_info["class_distribution"]
            labels = ["Surviving (0)", "Deceased (1)"]
            values = [dist_data["0"]["count"], dist_data["1"]["count"]]
            fig_target = px.pie(
                names=labels, 
                values=values,
                title="Target Variable Distribution (DEATH_EVENT)",
                color_discrete_sequence=['#00f2fe', '#ff3366'],
                hole=0.4
            )
            fig_target.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Outfit'),
                margin=dict(t=40, b=0, l=0, r=0),
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col_d2:
            st.markdown("#### 🧪 Injected Missing Values (10% Sample)")
            st.markdown("""
            Standard medical datasets can be pristine. To properly demonstrate the clinical imputation pipeline, we have injected 
            **10% missing values (NaNs)** into `serum_sodium` and `platelets`.
            """)
            
            # Draw missing values dataframe/chart
            missing_df = pd.DataFrame({
                "Feature": list(data_info["missing_values"].keys()),
                "Missing Count": list(data_info["missing_values"].values())
            }).sort_values("Missing Count", ascending=False)
            
            # Only show variables with missing values
            fig_missing = px.bar(
                missing_df[missing_df["Missing Count"] > 0],
                x="Feature",
                y="Missing Count",
                color="Missing Count",
                color_continuous_scale="Viridis",
                title="Count of Missing Records per Feature"
            )
            fig_missing.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Outfit'),
                margin=dict(t=40, b=20, l=20, r=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_missing, use_container_width=True)
            
        with col_d3:
            st.markdown("#### 🧮 Numerical Features Summary Statistics")
            summary_stats = pd.DataFrame(data_info["summary_stats"])
            # Format and show subset
            st.dataframe(summary_stats.loc[['mean', 'std', 'min', 'max'], pipeline_utils.NUM_COLS].T, height=350)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Could not contact the FastAPI backend to retrieve dataset information. Make sure it is running on port 8000.")


# ---------------------------------------------
# TAB 3: MODEL COMPARISON & LEAKAGE PROOF
# ---------------------------------------------
with tab_metrics:
    st.markdown("Configure settings in the sidebar and click **Train & Compare Pipelines** to compute statistics.")
    
    # Check if we have results in session state
    if 'train_results' not in st.session_state:
        # Trigger default training on first load
        with st.spinner("Initializing default model training..."):
            try:
                payload = {
                    "estimator_type": estimator,
                    "imputer_strategy": imputer,
                    "scaling_method": scaler,
                    "cv_folds": cv_folds,
                    "test_size": test_size
                }
                res = requests.post(f"{API_URL}/train", json=payload)
                if res.status_code == 200:
                    st.session_state['train_results'] = res.json()
                    st.session_state['trained_config'] = payload
            except Exception as e:
                st.error(f"Failed to connect to backend for training: {str(e)}")
                
    if train_button:
        with st.spinner("Fitting pipelines and running cross-validation splits..."):
            try:
                payload = {
                    "estimator_type": estimator,
                    "imputer_strategy": imputer,
                    "scaling_method": scaler,
                    "cv_folds": cv_folds,
                    "test_size": test_size
                }
                res = requests.post(f"{API_URL}/train", json=payload)
                if res.status_code == 200:
                    st.session_state['train_results'] = res.json()
                    st.session_state['trained_config'] = payload
                    st.toast("Model updated successfully!", icon="🔥")
                else:
                    st.error(f"Training failed: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
                
    # Display results if available
    if 'train_results' in st.session_state:
        res_data = st.session_state['train_results']
        conf = st.session_state['trained_config']
        
        st.info(f"Showing results for **{conf['estimator_type'].upper()}** classifier with **{conf['imputer_strategy']}** imputation and **{conf['scaling_method']}** scaling.")
        
        # 1. Performance Overview Grid
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">🛡️ Secure Pipeline Results</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-value metric-sec">{res_data["secure_cv"]["accuracy"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("5-Fold CV Accuracy")
            with c2:
                st.markdown(f'<div class="metric-value metric-sec">{res_data["secure_cv"]["roc_auc"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("5-Fold CV ROC AUC")
            with c3:
                st.markdown(f'<div class="metric-value metric-sec">{res_data["secure_holdout"]["accuracy"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("True Holdout Accuracy")
                
            st.markdown("""
            *Notice that the cross-validation score is a realistic estimate of the holdout test score. 
            Imputation & scaling statistics are strictly isolated in each fold.*
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">🚫 Leaky Pipeline Results</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-value metric-leak">{res_data["leaky_cv"]["accuracy"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("Leaky CV Accuracy")
            with c2:
                st.markdown(f'<div class="metric-value metric-leak">{res_data["leaky_cv"]["roc_auc"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("Leaky CV ROC AUC")
            with c3:
                st.markdown(f'<div class="metric-value metric-leak">{res_data["leaky_holdout"]["accuracy"]:.3f}</div>', unsafe_allow_html=True)
                st.caption("Leaky Holdout Accuracy")
                
            st.markdown("""
            *Leaky validation often yields an overly optimistic cross-validation score. However, when 
            evaluated on a truly clean holdout, the actual generalization capability drops.*
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 2. Plotly Charts comparing metrics
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card-header">📊 Cross-Validation vs. Holdout Test Generalization Gap</div>', unsafe_allow_html=True)
        
        metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        
        comparison_df = pd.DataFrame([
            {"Metric": m, "Score": res_data["secure_cv"][m.lower().replace(" ", "_")], "Flow": "Secure (CV)"} for m in metrics_names
        ] + [
            {"Metric": m, "Score": res_data["secure_holdout"][m.lower().replace(" ", "_")], "Flow": "Secure (Holdout Test)"} for m in metrics_names
        ] + [
            {"Metric": m, "Score": res_data["leaky_cv"][m.lower().replace(" ", "_")], "Flow": "Leaky (CV)"} for m in metrics_names
        ] + [
            {"Metric": m, "Score": res_data["leaky_holdout"][m.lower().replace(" ", "_")], "Flow": "Leaky (Holdout Test)"} for m in metrics_names
        ])
        
        fig_comp = px.bar(
            comparison_df,
            x="Metric",
            y="Score",
            color="Flow",
            barmode="group",
            color_discrete_sequence=['#00f2fe', '#4facfe', '#ff3366', '#ffa500'],
            labels={"Score": "Model Score"}
        )
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Outfit'),
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 3. ROC Curve and Confusion Matrix Row
        col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
        
        with col_c1:
            st.markdown('<div class="glass-card" style="height:500px;">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">📈 Holdout ROC Curve comparison</div>', unsafe_allow_html=True)
            
            fig_roc = go.Figure()
            
            # Secure
            roc_sec = res_data["roc_secure"]
            fig_roc.add_trace(go.Scatter(
                x=roc_sec['fpr'], y=roc_sec['tpr'],
                mode='lines',
                name=f'Secure Pipeline (AUC = {roc_sec["auc"]:.3f})',
                line=dict(color='#00f2fe', width=3)
            ))
            
            # Leaky
            roc_leak = res_data["roc_leaky"]
            fig_roc.add_trace(go.Scatter(
                x=roc_leak['fpr'], y=roc_leak['tpr'],
                mode='lines',
                name=f'Leaky Pipeline (AUC = {roc_leak["auc"]:.3f})',
                line=dict(color='#ff3366', width=3, dash='dash')
            ))
            
            # Diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Guess',
                line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
                showlegend=False
            ))
            
            fig_roc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Outfit'),
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(x=0.55, y=0.08)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_c2:
            st.markdown('<div class="glass-card" style="height:500px;">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">🛡️ Secure CM</div>', unsafe_allow_html=True)
            
            cm_sec = np.array(res_data["cm_secure"])
            fig_cm_sec = px.imshow(
                cm_sec,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Survived', 'Deceased'],
                y=['Survived', 'Deceased'],
                text_auto=True,
                color_continuous_scale="GnBu"
            )
            fig_cm_sec.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Outfit'),
                margin=dict(t=20, b=20, l=20, r=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cm_sec, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_c3:
            st.markdown('<div class="glass-card" style="height:500px;">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">🚫 Leaky CM</div>', unsafe_allow_html=True)
            
            cm_leak = np.array(res_data["cm_leaky"])
            fig_cm_leak = px.imshow(
                cm_leak,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Survived', 'Deceased'],
                y=['Survived', 'Deceased'],
                text_auto=True,
                color_continuous_scale="Reds"
            )
            fig_cm_leak.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Outfit'),
                margin=dict(t=20, b=20, l=20, r=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cm_leak, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 4. Feature Importance Panel
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card-header">🔑 Feature Importances (Secure Pipeline)</div>', unsafe_allow_html=True)
        
        feat_df = pd.DataFrame(res_data["feature_importances"])
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Plasma",
            title="Feature Importance Ranking"
        )
        fig_feat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Outfit'),
            margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_showscale=False,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------
# TAB 4: SINGLE-PATIENT DIAGNOSTIC TOOL
# ---------------------------------------------
with tab_predict:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">🩺 Diagnostic Input Panel</div>', unsafe_allow_html=True)
    st.write("Enter details for an individual patient to calculate their mortality risk score securely and view the execution trace.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        age_in = st.slider("Patient Age (Years)", min_value=30, max_value=100, value=65, step=1)
        sex_in = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        smoking_in = st.selectbox("Smoking Habits", options=[0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Smoker")
        diabetes_in = st.selectbox("Diabetes History", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
    with col_p2:
        ejection_fraction_in = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=35, step=1, help="Percentage of blood leaving the heart at each contraction (critical in heart failure).")
        serum_creatinine_in = st.slider("Serum Creatinine (mg/dL)", min_value=0.4, max_value=10.0, value=1.2, step=0.1, help="Normal range is typically 0.6 - 1.3 mg/dL. High values indicate kidney stress.")
        serum_sodium_in = st.selectbox("Serum Sodium (mEq/L)", options=["[Missing Value]", 120.0, 125.0, 130.0, 135.0, 137.0, 140.0, 145.0], index=5, help="Select a value or choose [Missing Value] to test the pipeline's Imputer!")
        
    with col_p3:
        platelets_in = st.selectbox("Platelet Count (kiloplatelets/mL)", options=["[Missing Value]", 50000.0, 100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 450000.0], index=5, help="Select a count or choose [Missing Value] to test the pipeline's Imputer!")
        anaemia_in = st.selectbox("Anaemia History", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes (low red blood cells)")
        high_bp_in = st.selectbox("High Blood Pressure History", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes (Hypertension)")
        cpk_in = st.slider("Creatinine Phosphokinase Level (mcg/L)", min_value=20, max_value=8000, value=500, step=50, help="Enzyme levels indicative of muscle or heart injury.")
        
    time_in = st.slider("Clinical Follow-up Period (Days)", min_value=1, max_value=300, value=120, step=1, help="Days of observation after primary diagnosis.")
    
    predict_button = st.button("🔍 Run Pipeline & Predict", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        with st.spinner("Processing clinical metrics through Scikit-Learn Pipeline..."):
            # Construct patient payload
            # Handle possible missing values
            serum_sodium_val = None if serum_sodium_in == "[Missing Value]" else float(serum_sodium_in)
            platelets_val = None if platelets_in == "[Missing Value]" else float(platelets_in)
            
            patient_payload = {
                "age": float(age_in),
                "anaemia": int(anaemia_in),
                "creatinine_phosphokinase": float(cpk_in),
                "diabetes": int(diabetes_in),
                "ejection_fraction": float(ejection_fraction_in),
                "high_blood_pressure": int(high_bp_in),
                "platelets": platelets_val if platelets_val is not None else np.nan,
                "serum_creatinine": float(serum_creatinine_in),
                "serum_sodium": serum_sodium_val if serum_sodium_val is not None else np.nan,
                "sex": int(sex_in),
                "smoking": int(smoking_in),
                "time": float(time_in)
            }
            
            # JSON doesn't handle NaN/inf natively, but we want to represent it. FastAPI accepts standard serialization or nulls.
            # In Python, we can map np.nan to None (which serializes to null in JSON).
            cleaned_payload = {}
            for k, v in patient_payload.items():
                if isinstance(v, float) and np.isnan(v):
                    cleaned_payload[k] = None
                else:
                    cleaned_payload[k] = v
                    
            try:
                res = requests.post(f"{API_URL}/predict", json=cleaned_payload)
                if res.status_code == 200:
                    trace_data = res.json()
                    
                    st.markdown("### 📋 Prediction Diagnostic Summary")
                    col_r1, col_r2 = st.columns([1, 2])
                    
                    with col_r1:
                        # Risk Card
                        prob = trace_data["mortality_probability"]
                        pct = prob * 100
                        
                        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
                        st.markdown("<h4>Estimated Mortality Risk</h4>", unsafe_allow_html=True)
                        
                        if pct >= 50:
                            st.markdown(f'<div class="risk-high"><h3>🔥 HIGH RISK</h3><h1>{pct:.1f}%</h1></div>', unsafe_allow_html=True)
                        elif pct >= 20:
                            st.markdown(f'<div class="risk-med"><h3>⚠️ MODERATE RISK</h3><h1>{pct:.1f}%</h1></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-low"><h3>🛡️ LOW RISK</h3><h1>{pct:.1f}%</h1></div>', unsafe_allow_html=True)
                            
                        st.markdown(f"<p style='margin-top:10px;'>Classification Outcome: <b>{'Death Event Predicted' if trace_data['prediction'] == 1 else 'Survival Predicted'}</b></p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col_r2:
                        # Pipeline Trace Card
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<div class="glass-card-header">🛠️ Secure Preprocessing Execution Trace</div>', unsafe_allow_html=True)
                        st.write("Below shows how the raw variables are transformed in real time by the pipeline estimators:")
                        
                        # Step 1: Input
                        st.markdown(f"""
                        <div class="trace-step">
                            <div class="trace-step-title">Step 1: Raw Inputs Received</div>
                            <div>Passed features: Age={age_in}, Ejection Fraction={ejection_fraction_in}%, Creatinine={serum_creatinine_in} mg/dL, Sodium={serum_sodium_in}, Platelets={platelets_in}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 2: Imputation
                        st.markdown(f"""
                        <div class="trace-step">
                            <div class="trace-step-title">Step 2: Numerical Imputation (SimpleImputer)</div>
                            <div>Filled values: <code>serum_sodium</code> ➔ <b>{trace_data['numerical_imputed']['serum_sodium']:.1f}</b>, <code>platelets</code> ➔ <b>{trace_data['numerical_imputed']['platelets']:.1f}</b></div>
                            <small style="color:#8da2fb;">(If missing, these were safely imputed using training-set statistics to avoid leakage.)</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 3: Scaling
                        st.markdown(f"""
                        <div class="trace-step">
                            <div class="trace-step-title">Step 3: Scaling & Transformation (StandardScaler)</div>
                            <div>Scaled values: Ejection Fraction ➔ <b>{trace_data['numerical_scaled']['ejection_fraction']:.3f}</b>, Creatinine ➔ <b>{trace_data['numerical_scaled']['serum_creatinine']:.3f}</b>, Sodium ➔ <b>{trace_data['numerical_scaled']['serum_sodium']:.3f}</b></div>
                            <small style="color:#8da2fb;">(Z-scores computed strictly using training fold mean and variance.)</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 4: Categorical Encoding
                        st.markdown(f"""
                        <div class="trace-step">
                            <div class="trace-step-title">Step 4: Categorical One-Hot Expansion</div>
                            <div>Categorical encoding vector shape: <b>{trace_data['categorical_encoded_shape']}</b>. Imputed categorical bounds verified.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 5: Classifier Prediction
                        st.markdown(f"""
                        <div class="trace-step">
                            <div class="trace-step-title">Step 5: Classifier Scoring</div>
                            <div>Input array length: <b>{len(trace_data['preprocessed_array'])}</b> ➔ Estimator output probability: <b>{prob:.4f}</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Prediction failed: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
