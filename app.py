import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
import io
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import plotly.express as px
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Smart Data Manager", layout="wide", initial_sidebar_state="auto")

st.markdown(
    """
    <div style='background-color: #111827; padding: 20px; border-radius: 12px'>
        <h1 style='color: white; font-size: 2em;'>📊 Smart Data Manager & Analyzer</h1>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "json"])

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

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🧹 Data Cleaning Options")
    if st.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
    if st.checkbox("Fill Missing Values with Mean"):
        df.fillna(df.mean(numeric_only=True), inplace=True)

    st.subheader("📈 Summary Statistics")
    stats = df.describe()
    st.dataframe(stats)

    st.subheader("📌 Column Filtering")
    selected_column = st.selectbox("Choose column to filter", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        filter_min = float(df[selected_column].min())
        filter_max = float(df[selected_column].max())
        user_range = st.slider("Select range", filter_min, filter_max, (filter_min, filter_max))
        df = df[(df[selected_column] >= user_range[0]) & (df[selected_column] <= user_range[1])]

    st.subheader("📊 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns to show heatmap.")

    st.subheader("📉 Visual Comparison Between Columns")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis column", df.columns, key="x")
    with col2:
        y_col = st.selectbox("Select Y-axis column", df.columns, key="y")

    if pd.api.types.is_numeric_dtype(df[y_col]):
        fig2 = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig2)

    st.subheader("📦 Bar & Pie Charts by Category")
    cat_col = st.selectbox("Choose categorical column for charts", df.select_dtypes(include='object').columns)
    if cat_col:
        bar_data = df[cat_col].value_counts()
        fig3 = px.bar(x=bar_data.index, y=bar_data.values, title="Bar Chart")
        st.plotly_chart(fig3)

        fig4 = px.pie(names=bar_data.index, values=bar_data.values, title="Pie Chart")
        st.plotly_chart(fig4)

    st.subheader("📅 Forecasting Chart")
    time_col = st.selectbox("Select time column for forecasting", df.columns)
    target_col = st.selectbox("Select target column for forecasting", df.select_dtypes(include=['float64', 'int64']).columns, key="forecast")
    if time_col and target_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col)
            model = ExponentialSmoothing(df[target_col], trend='add', seasonal=None).fit()
            forecast = model.forecast(10)
            fig5 = px.line(x=list(df[time_col]) + list(pd.date_range(df[time_col].iloc[-1], periods=10, freq='D')),
                           y=list(df[target_col]) + list(forecast),
                           labels={'x': 'Date', 'y': target_col},
                           title=f"Forecasting {target_col}")
            st.plotly_chart(fig5)
        except Exception as e:
            st.warning(f"Unable to forecast: {e}")

    st.subheader("🤖 Machine Learning Insight: Linear Regression")
    ml_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    ml_target = st.selectbox("Select target variable", ml_cols, key="target")
    ml_features = st.multiselect("Select feature columns", [col for col in ml_cols if col != ml_target])

    if ml_target and ml_features:
        X = df[ml_features]
        y = df[ml_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**R² Score:** {r2:.2f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")

        st.download_button("Download Trained Model", data=joblib.dump(model, 'model.pkl')[0], file_name="linear_model.pkl")

    st.subheader("🧠 Classification Model")
    cat_target = st.selectbox("Select categorical target", df.select_dtypes(include=['object']).columns, key="cls_target")
    if cat_target:
        df = df.dropna(subset=[cat_target])
        cls_features = st.multiselect("Select numeric features for classification", ml_cols, key="cls_feat")
        if cls_features:
            X = df[cls_features]
            y = df[cat_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_cls = LogisticRegression(max_iter=1000)
            model_cls.fit(X_train, y_train)
            y_pred_cls = model_cls.predict(X_test)
            acc = accuracy_score(y_test, y_pred_cls)
            st.write(f"**Accuracy:** {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_cls))
            st.download_button("Download Classification Model", data=joblib.dump(model_cls, 'clf_model.pkl')[0], file_name="classification_model.pkl")

    st.subheader("🧾 Predict from Input Form")
    if ml_features:
        st.write("### Enter values for prediction:")
        inputs = [st.number_input(f"{feat}", value=float(df[feat].mean())) for feat in ml_features]
        if st.button("Predict"):
            model = LinearRegression()
            model.fit(df[ml_features], df[ml_target])
            prediction = model.predict([inputs])[0]
            st.success(f"Predicted {ml_target}: {prediction:.2f}")

    st.subheader("📋 Generate PDF Report")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Data Summary Report", ln=True, align="C")
        for col in stats.columns:
            pdf.cell(200, 10, txt=f"{col} - Mean: {stats[col]['mean']:.2f}, Std: {stats[col]['std']:.2f}", ln=True)

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        st.download_button("Download PDF Report", data=pdf_output.getvalue(), file_name="data_report.pdf", mime="application/pdf")

    st.subheader("💾 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="cleaned_data.csv", mime='text/csv')

    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    st.download_button("Download Excel", data=excel_output.getvalue(), file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("🔗 Dataset Information")
    st.write("**Number of Rows:**", df.shape[0])
    st.write("**Number of Columns:**", df.shape[1])
    st.write("**Column Names:**", list(df.columns))
