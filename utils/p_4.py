import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to load pre-trained models from .pkl files
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Preprocessing function
def preprocess_data(df):
    df = df.copy()

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
def evaluate_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    st.write(f"## Evaluating {model_name} Model")

    # Training metrics
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    st.write(f"**Training Accuracy:** {train_acc:.4f} ({len(X_train)} samples)")
    st.write("### Training Classification Report:\n")
    st.text(classification_report(y_train, y_pred_train))

    # Validation metrics
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    st.write(f"**Validation Accuracy:** {val_acc:.4f} ({len(X_val)} samples)")
    st.write("### Validation Classification Report:\n")
    st.text(classification_report(y_val, y_pred_val))

    # Testing metrics
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    st.write(f"**Testing Accuracy:** {test_acc:.4f} ({len(X_test)} samples)")
    st.write("### Testing Classification Report:\n")
    st.text(classification_report(y_test, y_pred_test))

    # Confusion matrices for Training, Validation, and Test data
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_val = confusion_matrix(y_val, y_pred_val)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Plotting confusion matrices in Streamlit
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
    ax[0].set_title(f'Confusion Matrix (Train Data) - {model_name}')

    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
    ax[1].set_title(f'Confusion Matrix (Validation Data) - {model_name}')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[2])
    ax[2].set_title(f'Confusion Matrix (Test Data) - {model_name}')

    st.pyplot(fig)

    # ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC AUC:** {roc_auc:.4f}")

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_title(f'ROC Curve - {model_name}')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')

    st.pyplot(fig_roc)


# Main function to handle model loading, evaluation, and Streamlit display
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
    df = pd.read_csv('loan_data.csv')  # replace this with your data file path
    X, y = preprocess_data(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Streamlit interface
    st.title('Loan Prediction Models Evaluation')

    # Evaluate each model
    evaluate_model(log_reg_model, 'Logistic Regression', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(dt_model, 'Decision Tree', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(rf_model, 'Random Forest', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(gb_model, 'Gradient Boosting', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(ab_model, 'AdaBoost', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(svc_model, 'SVM', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(knn_model, 'K-Nearest Neighbors', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(nb_model, 'Naive Bayes', X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(mlp_model, 'MLP Classifier', X_train, X_val, X_test, y_train, y_val, y_test)

# Run the main function when the script is executed directly



