# Enhanced Smart Data Manager & Analyzer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import joblib
import io
from fpdf import FPDF
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from streamlit_extras.stylable_container import stylable_container
from streamlit_lottie import st_lottie
import requests

# Set config first
st.set_page_config(page_title="Smart Data Manager ✨", layout="wide", page_icon="📊")

# Load lottie animation
@st.cache_data(show_spinner=False)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://lottie.host/46c8a6c2-d8c6-4b99-bf6b-746a924f3c0c/CpMiJjW3z2.json"

# Title with animation
st_lottie(load_lottie_url(lottie_url), height=150, key="header")
st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>Smart Data Manager & Analyzer 🚀</h1>
""", unsafe_allow_html=True)

# Upload section
with stylable_container("upload-box", css="background-color:#f5f7fa; border-radius:12px; padding:20px; margin-bottom:20px;"):
    uploaded_file = st.file_uploader("📤 Upload your dataset", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Show raw data
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Cleaning options
    with stylable_container("cleaning", css="background-color:#e9f0fc; border-radius:12px; padding:10px;"):
        st.subheader("🧹 Data Cleaning")
        if st.checkbox("Remove Duplicates"):
            df = df.drop_duplicates()
        if st.checkbox("Fill Missing Values with Mean"):
            df.fillna(df.mean(numeric_only=True), inplace=True)

    # Stats
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # Filter
    st.subheader("📌 Column Filter")
    selected_column = st.selectbox("Choose a column to filter", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        min_val = float(df[selected_column].min())
        max_val = float(df[selected_column].max())
        user_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
        df = df[(df[selected_column] >= user_range[0]) & (df[selected_column] <= user_range[1])]

    # Heatmap
    st.subheader("📊 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for heatmap.")

    # Column comparison
    st.subheader("📉 Visual Comparison")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_col = st.selectbox("Select Y-axis", df.columns)

    if pd.api.types.is_numeric_dtype(df[y_col]):
        st.plotly_chart(px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}"))

    # Pie/Bar
    st.subheader("📦 Category Charts")
    cat_col = st.selectbox("Select Categorical Column", df.select_dtypes(include='object').columns)
    if cat_col:
        value_counts = df[cat_col].value_counts()
        st.plotly_chart(px.bar(x=value_counts.index, y=value_counts.values, title="Bar Chart"))
        st.plotly_chart(px.pie(names=value_counts.index, values=value_counts.values, title="Pie Chart"))

    # Forecast
    st.subheader("📅 Forecasting")
    time_col = st.selectbox("Select Time Column", df.columns)
    target_col = st.selectbox("Target Column", numeric_df.columns)
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col)
        model = ExponentialSmoothing(df[target_col], trend='add', seasonal=None).fit()
        forecast = model.forecast(10)
        future_dates = pd.date_range(start=df[time_col].iloc[-1], periods=10, freq='D')
        st.plotly_chart(px.line(x=list(df[time_col]) + list(future_dates),
                                y=list(df[target_col]) + list(forecast),
                                labels={'x': 'Date', 'y': target_col},
                                title=f"Forecast of {target_col}"))
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

    # Regression
    st.subheader("🤖 Linear Regression")
    target = st.selectbox("Target Variable", numeric_df.columns)
    features = st.multiselect("Feature Columns", [col for col in numeric_df.columns if col != target])
    if target and features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        st.write(f"R²: {r2_score(y_test, preds):.2f}, MSE: {mean_squared_error(y_test, preds):.2f}")
        st.download_button("Download Model", data=joblib.dump(reg, 'reg_model.pkl')[0], file_name="regression_model.pkl")

    # Classification
    st.subheader("🧠 Classification")
    cat_target = st.selectbox("Select Categorical Target", df.select_dtypes(include='object').columns)
    class_feats = st.multiselect("Select Numeric Features", numeric_df.columns)
    if cat_target and class_feats:
        df = df.dropna(subset=[cat_target])
        X = df[class_feats]
        y = df[cat_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text(classification_report(y_test, y_pred))
        st.download_button("Download Classifier", data=joblib.dump(clf, 'clf.pkl')[0], file_name="classifier.pkl")

    # Prediction form
    st.subheader("🧾 Predict With Form")
    if features:
        inputs = [st.number_input(f"{feat}", value=float(df[feat].mean())) for feat in features]
        if st.button("Predict"):
            result = reg.predict([inputs])[0]
            st.success(f"Prediction: {result:.2f}")

    # PDF Export
    st.subheader("📋 Export Summary PDF")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Data Summary Report", ln=True, align="C")
        for col in df.describe().columns:
            mean = df.describe()[col]['mean']
            std = df.describe()[col]['std']
            pdf.cell(200, 10, txt=f"{col} - Mean: {mean:.2f}, Std: {std:.2f}", ln=True)
        pdf_bytes = io.BytesIO()
        pdf.output(pdf_bytes)
        st.download_button("Download PDF", data=pdf_bytes.getvalue(), file_name="summary.pdf", mime="application/pdf")

    # Export Clean Data
    st.subheader("💾 Download Cleaned Data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_data, file_name="cleaned_data.csv")

    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Excel", data=excel_data.getvalue(), file_name="cleaned_data.xlsx")
