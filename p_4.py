import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, auc)


# Function to load pre-trained models from .pkl files
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Preprocessing function
def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0', 'Loan_ID'])

    # Fill missing values
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'property_Area', 'Loan_Status']:
        df[col] = label_encoder.fit_transform(df[col])

    # Feature-target split
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Scaling numerical features
    scaler = StandardScaler()
    X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(
        X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

    return X, y


# Function to evaluate model and show detailed results in Streamlit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Function to evaluate model and show detailed results in Streamlit
# Function to evaluate model and show detailed results in Streamlit
def evaluate_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test, comparison_df=None):
    """
    Evaluates the model, displays metrics, and visualizes results in a structured and distinguishable format.
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
    }

    # Append metrics to comparison dataframe
    if comparison_df is None:
        comparison_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    comparison_df = pd.concat([comparison_df, pd.DataFrame([{"Model": model_name, **metrics}])], ignore_index=True)

    # Add a header to distinguish sections
    st.markdown("---")
    st.markdown(f"## {model_name} Evaluation")
    st.markdown("### Metrics Summary")

    # Display metrics in a Streamlit table
    st.dataframe(
        comparison_df.style.format(precision=4).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f0f0'), ('font-size', '14px')]},
            {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '12px')]}
        ])
    )

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(plt)

    # Classification Report
    st.markdown("### Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    st.dataframe(
        report_df.style.format(precision=4).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f7f7f7'), ('font-size', '14px')]},
            {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '12px')]}
        ])
    )

    # Plot ROC Curve if possible
    if hasattr(model, "predict_proba"):
        st.markdown("### ROC Curve")
        from sklearn.metrics import roc_curve
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label="ROC Curve", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        st.pyplot(plt)

    # Add a footer to separate sections
    st.markdown("---")

    return comparison_df

def run_loan_prediction_evaluation():
    # Load your pre-trained models (replace paths with the correct paths to your .pkl files)
    log_reg_model = load_model('models/lr.pkl')
    dt_model = load_model('models/dt.pkl')
    rf_model = load_model('models/rf.pkl')
    gb_model = load_model('models/gb.pkl')
    ab_model = load_model('models/ab.pkl')
    svc_model = load_model('models/svm.pkl')
    knn_model = load_model('models/knn.pkl')
    nb_model = load_model('models/nvc.pkl')
    mlp_model = load_model('models/mlp.pkl')

    # Load the data
    df = pd.read_csv('data/dataset.csv')  # replace this with your data file path
    X, y = preprocess_data(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Streamlit interface
    st.title('Loan Prediction Models Evaluation')

    comparison_df = None

    # Evaluate each model and store performance comparison
    comparison_df = evaluate_model(log_reg_model, 'Logistic Regression', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(dt_model, 'Decision Tree', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(rf_model, 'Random Forest', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(gb_model, 'Gradient Boosting', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(ab_model, 'AdaBoost', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(svc_model, 'SVM', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(knn_model, 'K-Nearest Neighbors', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(nb_model, 'Naive Bayes', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)
    comparison_df = evaluate_model(mlp_model, 'MLP Classifier', X_train, X_val, X_test, y_train, y_val, y_test, comparison_df)

    # Display comparison table at the end
    st.write("### Model Performance Comparison")
    st.dataframe(comparison_df.style.format({
        "Train Accuracy": "{:.4f}",
        "Validation Accuracy": "{:.4f}",
        "Test Accuracy": "{:.4f}",
        "ROC AUC": "{:.4f}"
    }), width=1000)







