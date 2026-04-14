import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set Page Config
st.set_page_config(
    page_title="Demand AI | Customer Churn", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📡"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global App Styling */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit top header and black line */
    [data-testid="stHeader"] {
        display: none;
    }
    [data-testid="stDecoration"] {
        display: none;
    }
    
    /* Top Padding adjustment */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 2rem;
    }
    
    /* Elegant Metrics Cards layout with glassmorphism */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.85) 100%);
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(251, 113, 133, 0.15), 0 8px 10px -6px rgba(251, 113, 133, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px rgba(251, 113, 133, 0.2), 0 10px 10px rgba(0,0,0,0.05);
    }
    
    /* Highlighting Metric Value */
    div[data-testid="stMetricValue"] {
        color: #fb7185;
        font-weight: 800;
        font-size: 2rem;
    }

    hr {
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    h1 {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #3b82f6, #1d4ed8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    h2, h3, h4 {
        color: #1e293b;
        font-weight: 700;
    }
    
    p, label, span {
        color: #334155;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f1f5f9;
        box-shadow: 2px 0 10px rgba(0,0,0,0.03);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #fb7185 0%, #f43f5e 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(244, 63, 94, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(244, 63, 94, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 Intelligent Customer Churn Prediction")
st.markdown("Analyze customer risk and predict churn probabilities automatically using Machine Learning.")

# --- LOAD DATA AND TRAIN MODEL ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('telecom_customer_churn_prediction_dataset.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def train_model(df):
    df_processed = df.copy()
    
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
    
    label_encoders = {}
    object_cols = df_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, label_encoders, acc, X.columns

df = load_data()

if df is None:
    st.error("Dataset not found. Please ensure 'telecom_customer_churn_prediction_dataset.csv' is in the directory.")
    st.stop()

model, scaler, label_encoders, model_accuracy, feature_cols = train_model(df)

# --- SIDEBAR INTERACTIVE PREDICTION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=60)
st.sidebar.title("Customer Profile Simulator")
st.sidebar.markdown("Input characteristics to observe real-time churn risk.")

input_data = {}

# Some key features categorized
st.sidebar.subheader("Demographics")
input_data['gender'] = st.sidebar.selectbox("Gender", df['gender'].unique())
input_data['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1])
input_data['Partner'] = st.sidebar.selectbox("Partner", df['Partner'].unique())
input_data['Dependents'] = st.sidebar.selectbox("Dependents", df['Dependents'].unique())

st.sidebar.subheader("Service Usage")
input_data['tenure'] = st.sidebar.slider("Tenure (Months)", int(df['tenure'].min()), int(df['tenure'].max()), int(df['tenure'].median()))
input_data['PhoneService'] = st.sidebar.selectbox("Phone Service", df['PhoneService'].unique())
input_data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", df['MultipleLines'].unique())
input_data['InternetService'] = st.sidebar.selectbox("Internet Service", df['InternetService'].unique())
input_data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", df['OnlineSecurity'].unique())
input_data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", df['OnlineBackup'].unique())
input_data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", df['DeviceProtection'].unique())
input_data['TechSupport'] = st.sidebar.selectbox("Tech Support", df['TechSupport'].unique())
input_data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", df['StreamingTV'].unique())
input_data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", df['StreamingMovies'].unique())

st.sidebar.subheader("Account Information")
input_data['Contract'] = st.sidebar.selectbox("Contract Type", df['Contract'].unique())
input_data['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
input_data['PaymentMethod'] = st.sidebar.selectbox("Payment Method", df['PaymentMethod'].unique())
input_data['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges ($)", float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()), float(df['MonthlyCharges'].median()))
input_data['TotalCharges'] = st.sidebar.number_input("Total Charges ($)", float(df['TotalCharges'].min()), float(df['TotalCharges'].max()), float(df['TotalCharges'].median()))

st.sidebar.markdown("---")
st.sidebar.info(f"Model Engine: **Random Forest Classifier**\n\nValidation Accuracy: *{model_accuracy:.1%}*")

# Convert input to dataframe
input_df = pd.DataFrame([input_data])
# Reorder to match model
input_df = input_df[feature_cols]

# Encode
for col in input_df.columns:
    if col in label_encoders and col != 'Churn':
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Scale
input_scaled = scaler.transform(input_df)

# Predict
churn_prob = model.predict_proba(input_scaled)[0][1]
churn_pred = model.predict(input_scaled)[0]

risk_color = "🔴 High Risk of Churn" if churn_pred == 1 else "🟢 Loyal (Low Risk)"

# --- MAIN CONTENT ---

# KPIs
st.markdown("### 📈 Live Customer Demographics Overview")
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
overall_churn_rate = (df[df['Churn'] == 'Yes'].shape[0] / total_customers) * 100
avg_tenure = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()

col1.metric("Total Active Datapoints", f"{total_customers:,}")
col2.metric("Overall Churn Rate", f"{overall_churn_rate:.1f}%")
col3.metric("Average Tenure", f"{avg_tenure:.1f} months")
col4.metric("Avg Monthly Revenue", f"${avg_monthly:.2f}")

st.markdown("---")

st.markdown("### 🎯 Artificial Intelligence Risk Assessment")
p_col1, p_col2 = st.columns([1, 2])

with p_col1:
    st.markdown("#### Scenario Profile")
    st.metric("Probability of Churn", f"{churn_prob*100:.1f}%")
    st.metric("Predicted Status", risk_color)
    if churn_pred == 1:
        st.warning("This customer profile exhibits factors heavily correlated with dropping services. Consider targeted retention strategies.")
    else:
        st.success("This profile indicates strong retention probability. Continue excellent service!")

with p_col2:
    gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_prob*100,
        title = {'text': "Churn Probability (%)", 'font': {'color': '#374151'}},
        number = {'font': {'color': '#111827'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#374151"},
            'bar': {'color': "#f43f5e" if churn_prob > 0.5 else "#3b82f6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': "rgba(59, 130, 246, 0.1)"},
                {'range': [50, 100], 'color': "rgba(244, 63, 94, 0.1)"}
            ]
        }
    ))
    gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(gauge, width='stretch')

st.markdown("---")

# Charts
st.markdown("### 📊 Correlation & Pattern Analysis (All Customers)")

c_col1, c_col2 = st.columns(2)

with c_col1:
    contract_dist = px.histogram(df, x="Contract", color="Churn", barmode='group', 
                        color_discrete_map={"Yes": "#f43f5e", "No": "#3b82f6"},
                        title="Churn Frequency vs Contract Duration")
    contract_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#374151', xaxis_title="Contract Type", yaxis_title="Customer Count")
    contract_dist.update_xaxes(showgrid=False)
    contract_dist.update_yaxes(showgrid=True, gridcolor='#00000')
    st.plotly_chart(contract_dist, width='stretch')

with c_col2:
    charge_dist = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                  color_discrete_map={"Yes": "#f43f5e", "No": "#3b82f6"},
                  title="Monthly Charges Impact on Churn")
    charge_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#374151', xaxis_title="Churn Event", yaxis_title="Monthly Charges ($)")
    charge_dist.update_xaxes(showgrid=False)
    charge_dist.update_yaxes(showgrid=True, gridcolor='#00000')
    st.plotly_chart(charge_dist, width='stretch')

# Dataset preview
with st.expander("View Raw Dataset"):
    st.dataframe(df.head(100), width='stretch')

