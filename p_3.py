import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model, scaler, and label encoder

scaler = StandardScaler()
label_encoder = LabelEncoder()

def load_model():
    try:
        with open('models/random_forest_model.pkl', 'rb') as model_file:
            loan_model = pickle.load(model_file)
        print(f"Model loaded successfully: {type(loan_model)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        loan_model = None
    return loan_model


# Preprocessing function
def preprocess_data(df, label_encoder, scaler):
    df = df.copy()

    # Fill missing values
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Encoding categorical variables
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'property_Area']:
        df[col] = label_encoder.fit_transform(df[col])

    # Scaling numerical features
    df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(
        df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
    )

    return df


def loan_prediction_page():
    # Load pre-trained model
    loan_model = load_model()

    if loan_model is None:
        st.error("Model could not be loaded.")
        return

    # Title of the page
    st.title("Loan Prediction")

    # Input fields without default values
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["No", "Yes"])
    dependents = st.number_input("Dependents", min_value=0, step=1)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
    applicant_income = st.number_input("Applicant Income", min_value=0.0, step=1.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=1.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1.0)
    loan_amount_term = st.selectbox(
        "Loan Amount Term", options=[360.0, 180.0, 480.0, 300.0, 84.0, 240.0, 120.0, 60.0, 36.0, 12.0]
    )
    credit_history = st.selectbox("Credit History", options=[0.0, 1.0])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

    # Prepare the input data
    data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "property_Area": property_area
    }

    df_input = pd.DataFrame([data])

    # Preprocess the input data (the target variable is excluded here)
    df_input = preprocess_data(df_input, label_encoder, scaler)

    # Make the prediction
    try:
        prediction = loan_model.predict(df_input)

        if prediction == 1:
            st.success("Congratulations! You are eligible for the loan.")
        else:
            st.error("Sorry! You are not eligible for the loan.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

    # Button to make another prediction
    if st.button('Make another prediction'):
        st.rerun()  # Refresh the page


# Call the loan prediction page
loan_prediction_page()
