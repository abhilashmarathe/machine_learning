import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💸",
    layout="centered",
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .title {
        font-size: 40px !important;
        font-weight: 700 !important;
        text-align: center;
        color: #4CAF50;
    }
    .sub {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #555;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #aaa;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("income_model.pkl")
        return model
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

model = load_model()

st.markdown("<div class='title'>💸 Income Prediction App</div>", unsafe_allow_html=True)
st.write("Predict if a person earns **<=50K or >50K** using your trained ML model.")

st.divider()

if model is None:
    st.stop()

# ------------------------------------------------------------
# Extract Feature Names
# ------------------------------------------------------------
try:
    preprocessor = model.named_steps["prep"]
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
except:
    st.error("Model must be a Pipeline with preprocessing.")
    st.stop()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("⚙️ App Controls")
mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction"])
autofill = st.sidebar.checkbox("Auto-fill Example Values")

# Example autofill values
example_values = {
    "age": 37,
    "fnlwgt": 200000,
    "education-num": 13,
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass": "Private",
    "education": "Bachelors",
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States"
}

# ------------------------------------------------------------
# SINGLE PREDICTION FORM
# ------------------------------------------------------------
if mode == "Single Prediction":
    st.markdown("<div class='sub'>🧍 Enter User Details</div>", unsafe_allow_html=True)

    user_input = {}

    st.write("### 🔢 Numeric Inputs")
    for col in num_features:
        default = example_values[col] if autofill and col in example_values else 0.0
        user_input[col] = st.number_input(col, value=float(default))

    st.write("### 🔠 Categorical Inputs")
    for col in cat_features:
        default = example_values[col] if autofill and col in example_values else ""
        user_input[col] = st.text_input(col, value=default)

    st.markdown("---")

    if st.button("✨ Predict"):
        try:
            df_input = pd.DataFrame([user_input])
            prediction = model.predict(df_input)[0]

            st.success(f"💡 Predicted Income: **{prediction}**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ------------------------------------------------------------
# BATCH CSV PREDICTION
# ------------------------------------------------------------
else:
    st.markdown("<div class='sub'>📁 Upload CSV for Batch Prediction</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 🔍 Preview:")
            st.dataframe(df.head())

            if st.button("🚀 Run Batch Prediction"):
                preds = model.predict(df)
                df["prediction"] = preds

                st.success("Batch Prediction Completed!")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="income_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"CSV Processing Error: {e}")


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("<div class='footer'>Made with ❤️ by Abhi</div>", unsafe_allow_html=True)
