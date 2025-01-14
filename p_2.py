import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind





def navigate_to(page_name):
    st.session_state["current_page"] = page_name

def insight_analytics_page():
    # Back button
    if st.button("⬅️ Back"):
        navigate_to("home")

    # Page Title
    st.markdown('<h1 class="main-heading">📊 Insight Analytics</h1>', unsafe_allow_html=True)

    # Subsection: Feature Importances
    st.markdown("### **Feature Importances: What Influences Loan Approval?**")
    st.write(
        """
        In this section, we analyze the most critical factors that influence the likelihood of getting a loan approved. 
        By identifying which features carry the most weight in predicting loan approval, we can better understand the decision-making process and provide actionable insights to applicants.

        We have used three methods to determine feature importances:

        1. **Random Forest Feature Importance**:
           Highlights features that contribute most to reducing prediction errors in a Random Forest model.

        2. **Gradient Boosting Feature Importance**:
           Evaluates feature importance based on model performance improvements during training.

        3. **SHAP (SHapley Additive exPlanations)**:
           Quantifies each feature's contribution to individual predictions for better interpretability.

        Stay tuned as we dive into the details! 🚀
        """
    )
    df = pd.read_csv('data/dataset.csv')
    # Drop columns if they exist
    df.drop(columns=[col for col in ['Loan_ID', 'Unnamed: 0'] if col in df], inplace=True)

    # Random Forest Feature Importance Analysis
    display_random_forest_feature_importance()

    # Gradient Boosting Feature Importance Analysis
    display_gradient_boosting_feature_importance()

    # SHAP Feature Importance Analysis
    display_shap_feature_importance()

    st.markdown("### **Univariate analysis : What individual factors speak?")

    univariate_analysis(df)

    multivariate_analysis(df)

    st.markdown("---")


def display_random_forest_feature_importance():
    # Load the dataset
    try:
        df = pd.read_csv('data/dataset.csv')
        # Drop columns if they exist
        df.drop(columns=[col for col in ['Loan_ID', 'Unnamed: 0'] if col in df], inplace=True)
    except FileNotFoundError:
        st.error("The dataset file is missing. Please check the path.")
        return

    # Define columns
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'property_Area', 'Credit_History']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # Filter valid columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if 'Loan_Status' not in df.columns:
        st.error("The 'Loan_Status' column is missing in the dataset.")
        return

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Loan_Status', axis=1),
        df['Loan_Status'],
        test_size=0.2,
        random_state=42
    )

    # Define preprocessors
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]
    )

    # Train the model
    with st.spinner('Training the Random Forest model...'):
        pipeline.fit(X_train, y_train)
    st.success('Model training complete!')

    # Extract feature importances
    rf_model = pipeline.named_steps['classifier']
    encoded_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[
        'onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(encoded_feature_names)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': np.array(all_feature_names)[indices],
        'Importance': importances[indices]
    })

    # Display the DataFrame
    st.markdown("### Random Forest Feature Importances")
    st.write("The table below shows the features ranked by their importance in predicting loan approval:")
    st.dataframe(feature_importance_df.style.background_gradient(cmap="Blues"))

    # Plot feature importance
    st.write("The bar chart below visualizes the relative importance of each feature:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(all_feature_names)), importances[indices], align="center")
    ax.set_xticks(range(len(all_feature_names)))
    ax.set_xticklabels(np.array(all_feature_names)[indices], rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance in Random Forest")
    fig.tight_layout()
    st.pyplot(fig)


def display_gradient_boosting_feature_importance():
    # Load the dataset
    try:
        df = pd.read_csv('data/dataset.csv')
        # Drop unnecessary columns if they exist
        df.drop(columns=[col for col in ['Loan_ID', 'Unnamed: 0'] if col in df], inplace=True)
    except FileNotFoundError:
        st.error("The dataset file is missing. Please check the path.")
        return

    # Define columns
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'property_Area', 'Credit_History']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # Filter valid columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if 'Loan_Status' not in df.columns:
        st.error("The 'Loan_Status' column is missing in the dataset.")
        return

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Loan_Status', axis=1),
        df['Loan_Status'],
        test_size=0.2,
        random_state=42
    )

    # Define preprocessors
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    # Create pipeline for Gradient Boosting
    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', GradientBoostingClassifier(random_state=42))]
    )

    # Train the model
    with st.spinner('Training the Gradient Boosting model...'):
        pipeline.fit(X_train, y_train)
    st.success('Gradient Boosting model training complete!')

    # Extract feature importances
    gb_model = pipeline.named_steps['model']
    encoded_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[
        'onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(encoded_feature_names)
    importances = gb_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Display Feature Importances
    st.markdown("### Gradient Boosting Feature Importances")
    st.write("The table below shows the features ranked by their importance in predicting loan approval:")
    feature_importance_df = pd.DataFrame({
        'Feature': [all_feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    st.dataframe(feature_importance_df.style.background_gradient(cmap="Oranges"))

    # Plot feature importance
    st.write("The bar chart below visualizes the relative importance of each feature:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([all_feature_names[i] for i in indices], importances[indices], color='skyblue')
    ax.set_xlabel("Feature Importance")
    ax.set_title("Gradient Boosting Feature Importances")
    fig.tight_layout()
    st.pyplot(fig)

def display_shap_feature_importance():
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    st.markdown("### SHAP Feature Importances")
    st.write("The table below shows the features ranked by their importance in predicting loan approval:")

    # Feature importance data
    data = {
        "Feature": [
            "Credit_History_1.0", "LoanAmount", "property_Area_Semiurban",
            "ApplicantIncome", "CoapplicantIncome", "Married_Yes",
            "Loan_Amount_Term", "Education_Not Graduate", "Gender_Male",
            "Self_Employed_Yes", "property_Area_Urban"
        ],
        "Mean SHAP Importance": [
            1.044946, 0.327569, 0.288124, 0.227160, 0.161996,
            0.131553, 0.114368, 0.045295, 0.009001, 0.008832, 0.004428
        ]
    }

    # Create a DataFrame for the feature importance
    importance_df = pd.DataFrame(data)

    # Sort the DataFrame by importance values
    importance_df = importance_df.sort_values(by="Mean SHAP Importance", ascending=False)

    # Style the DataFrame with a gradient
    styled_df = importance_df.style.background_gradient(
        subset=["Mean SHAP Importance"], cmap="Blues"
    )

    # Display the styled DataFrame
    st.dataframe(styled_df)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Mean SHAP Importance"], color="skyblue")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # To match the descending order in the DataFrame
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(plt)


def univariate_analysis(df):
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import  pandas as pd
    df = pd.read_csv('data/dataset.csv')
    # Drop columns if they exist
    df.drop(columns=[col for col in ['Loan_ID', 'Unnamed: 0'] if col in df], inplace=True)
    # Gender Distribution
    st.markdown("### Gender Distribution in Loan Applicants")
    st.write(
        "This reflects an uneven representation of gender in the loan application process, which could indicate underlying economic, social, or cultural factors influencing the accessibility of loans for females. Such imbalances may suggest areas for improvement in financial inclusivity and equality.")

    # Gender Count Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Gender', data=df, palette='viridis', edgecolor='black', linewidth=1.2)
    plt.title('Gender Distribution Among Loan Applicants', fontsize=16, fontweight='bold')
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # Gender Pie Chart
    plt.figure(figsize=(6, 6))
    df['Gender'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=sns.color_palette('pastel')[:2],
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    plt.title('Gender Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('')  # Hides the y-label for a cleaner look
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # Heading and Analysis Description
    st.markdown("### Gender Distribution in Loan Approval")
    st.write("""
        This section explores the distribution of loan approvals and rejections across different genders.  
        Understanding gender-based disparities can help identify areas where financial inclusivity can be improved.
        """)

    # Calculate Loan Approval Rates by Gender
    gender_loan_data = df.groupby('Gender')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    st.markdown("#### Loan Approval Percentages by Gender")
    st.write(gender_loan_data)

    # Bar Plot for Loan Status by Gender
    plt.figure(figsize=(8, 5))
    gender_loan_data.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Status Distribution by Gender', fontsize=16, fontweight='bold')
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    st.markdown("### Marriage Status Distribution Among Loan Seekers")
    st.write("""
        65.3% of Loan Seekers are Married, while 34.7% are Not Married.  
        This suggests that married individuals might be more likely to apply for loans, potentially due to:
        - **Financial stability**: Married individuals may have a stronger perception of financial security or shared financial responsibilities.
        - **Family-oriented goals**: Loans may be more likely sought for family-related goals like buying a house or securing education.

        However, it also raises the question of whether single or unmarried individuals face barriers or different financial conditions when applying for loans, highlighting an area that may require further exploration in terms of financial inclusivity.
        """)

    # Count Plot for Married Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x='Married',
        data=df,
        palette='viridis',
        edgecolor='black',
        linewidth=1.2
    )
    plt.title('Marriage Status Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Marriage Status', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # Pie Chart for Married Distribution
    plt.figure(figsize=(6, 6))
    df['Married'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['skyblue', 'orange'],
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    plt.title('Marriage Status Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('')  # Hides the y-label for a cleaner look
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # Heading and Analysis Description
    st.markdown("### Marriage Status and Loan Approval Distribution")
    st.write("""
        This section explores the relationship between marriage status and loan approval rates.  
        It sheds light on whether marital status influences loan outcomes and highlights potential disparities.
        """)

    # Calculate Loan Approval Rates by Marriage Status
    married_loan_data = df.groupby('Married')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    st.markdown("#### Loan Approval Percentages by Marriage Status")
    st.write(married_loan_data)

    # Bar Plot for Loan Status by Marriage Status
    plt.figure(figsize=(8, 5))
    married_loan_data.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Status Distribution by Marriage Status', fontsize=16, fontweight='bold')
    plt.xlabel('Marriage Status', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    """
       Perform univariate analysis on the 'Dependents' column (continuous) and analyze loan approvals.
       """
    """
       Perform univariate analysis on the 'Dependents' column and analyze its relationship with loan approval.
       """
    # Heading and Analysis Description
    st.markdown("### Dependents Distribution in Loan Applicants")
    st.write("""
       This section explores the relationship between the number of dependents and loan approval rates.  
       It sheds light on whether having more dependents influences the likelihood of loan approval.
       """)

    # Distribution of Dependents
    st.markdown("#### Dependents Distribution")
    dependents_counts = df['Dependents'].value_counts(normalize=True) * 100
    st.write(dependents_counts)

    # Bar Plot for Dependents Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Dependents', data=df, palette='viridis', order=df['Dependents'].value_counts().index)
    plt.title('Dependents Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Dependents', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Pie Chart for Dependents Distribution
    plt.figure(figsize=(6, 6))
    df['Dependents'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'orange', 'green', 'red'],
                                             startangle=90)
    plt.title('Dependents Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    st.pyplot(plt)

    # Loan Approval Rates by Dependents
    st.markdown("#### Loan Approval Rates by Dependents")
    dependents_loan_data = df.groupby('Dependents')['Loan_Status'].value_counts(normalize=True).unstack() * 100
    st.write(dependents_loan_data)

    # Bar Plot for Loan Status by Dependents
    plt.figure(figsize=(8, 5))
    dependents_loan_data.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Status Distribution by Dependents', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Dependents', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Key Insights
    st.markdown("### Key Insights")
    st.write("""
       - The majority of loan applicants have **0** dependents, followed by **1 dependent**.
       - Loan approval rates appear to be fairly consistent across different levels of dependents.
       - The stacked bar chart shows that applicants with more dependents (e.g., 2 or 3) may still have a high chance of loan approval, suggesting that the number of dependents does not strongly influence the approval rate.
       """)
    # Heading and Analysis Description
    st.markdown("### Education Level Distribution in Loan Applicants")
    st.write("""
       This section explores the distribution of education levels among loan applicants and their influence on loan approval rates.  
       It provides insights into how education correlates with loan-seeking behavior and outcomes.
       """)

    # Distribution of Education Levels
    st.markdown("#### Education Level Distribution")
    education_counts = df['Education'].value_counts(normalize=True) * 100
    st.write(education_counts)

    # Bar Plot for Education Level Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Education', data=df, palette='viridis', order=df['Education'].value_counts().index)
    plt.title('Education Level Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Education Level', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Pie Chart for Education Level
    plt.figure(figsize=(6, 6))
    df['Education'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90,
                                            explode=[0.05, 0])
    plt.title('Education Level Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    st.pyplot(plt)

    # Loan Approval Rates by Education Level
    st.markdown("#### Loan Approval Rates by Education Level")
    education_loan_data = df.groupby('Education')['Loan_Status'].value_counts(normalize=True).unstack() * 100
    st.write(education_loan_data)

    # Bar Plot for Loan Status by Education Level
    plt.figure(figsize=(8, 5))
    education_loan_data.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Status Distribution by Education Level', fontsize=16, fontweight='bold')
    plt.xlabel('Education Level', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Key Insights
    st.markdown("### Key Insights")
    st.write("""
       - **78%** of loan applicants are graduates, highlighting a significant trend of higher education levels among loan seekers.
       - Graduates tend to have higher loan approval rates, possibly reflecting perceptions of financial stability and employability.
       - The stacked bar chart shows that non-graduates may face slightly higher rejection rates, indicating potential disparities in the loan approval process.
       """)

    """
        Perform univariate analysis on the 'Employment Status' column and analyze its relationship with loan approval.
        """
    """
       Perform univariate analysis on the 'Self_Employed' column and analyze its relationship with loan approval.
       """
    # Heading and Analysis Description
    st.markdown("### Self-Employed Status Distribution in Loan Applicants")
    st.write("""
       This section explores the relationship between self-employment status and loan approval rates.  
       It highlights whether being self-employed influences the likelihood of loan approval.
       """)

    # Distribution of Self_Employed Status
    st.markdown("#### Self-Employed Status Distribution")
    self_employed_counts = df['Self_Employed'].value_counts(normalize=True) * 100
    st.write(self_employed_counts)

    # Bar Plot for Self-Employed Status Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Self_Employed', data=df, palette='viridis', order=df['Self_Employed'].value_counts().index)
    plt.title('Self-Employed Status Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Self-Employed Status', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Pie Chart for Self-Employed Status Distribution
    plt.figure(figsize=(6, 6))
    df['Self_Employed'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90)
    plt.title('Self-Employed Status Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    st.pyplot(plt)

    # Loan Approval Rates by Self-Employed Status
    st.markdown("#### Loan Approval Rates by Self-Employed Status")
    self_employed_loan_data = df.groupby('Self_Employed')['Loan_Status'].value_counts(normalize=True).unstack() * 100
    st.write(self_employed_loan_data)

    # Bar Plot for Loan Status by Self-Employed Status
    plt.figure(figsize=(8, 5))
    self_employed_loan_data.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Status Distribution by Self-Employed Status', fontsize=16, fontweight='bold')
    plt.xlabel('Self-Employed Status', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    # Key Insights
    st.markdown("### Key Insights")
    st.write("""
       - **Self-employed applicants** represent a smaller portion (13.4%) of loan applicants compared to salaried individuals.
       - **Salaried applicants** are more likely to get approved, reflecting the perception of stable income and financial reliability.
       - The loan approval rate for **self-employed applicants** is lower, which may be due to the increased scrutiny on income stability and documentation requirements.
       - This trend suggests that **self-employed individuals** may face challenges when seeking loans, potentially due to income volatility and the complexity of verifying financial stability.
       """)

    """
       Perform univariate analysis on the 'ApplicantIncome' column, visualizing distribution and summarizing key metrics.
       """
    # Heading and Analysis Description
    st.markdown("### Applicant Income Distribution in Loan Applicants")
    st.write("""
       This section explores the distribution of applicant incomes and their potential impact on loan approvals.
       The analysis highlights the skewness and outliers in the income distribution.
       """)

    # Statistical Summary
    st.markdown("#### Statistical Summary of Applicant Income")
    income_summary = df['ApplicantIncome'].describe()
    st.write(income_summary)

    # Boxplot to visualize outliers and distribution
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['ApplicantIncome'], color='lightcoral')
    plt.title('Applicant Income Boxplot', fontsize=16, fontweight='bold')
    plt.xlabel('Applicant Income', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

    # Histogram with KDE (Kernel Density Estimate) to show the distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['ApplicantIncome'], kde=True, color='skyblue', bins=30)
    plt.title('Applicant Income Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Applicant Income', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

    # Key Insights
    st.markdown("### Key Insights")
    st.write("""
       - The **right-skewed** distribution of income indicates that a small number of applicants have very high incomes compared to the majority.
       - The **mean** income is much higher than the **median**, confirming the skewness in the data.
       - A significant **standard deviation** shows high variability in the income of loan applicants.
       - **Outliers**: There are some applicants with extremely high incomes (e.g., ₹8,100,000), which could potentially skew loan eligibility decisions.
       - This income disparity could affect the **loan approval process**, as applicants with higher incomes might be given different treatment, including eligibility criteria or interest rates.
       """)


        # Heading and Analysis Description
    st.markdown("### Loan Approval by Applicant Income Bins")
    st.write("""
           This section explores the relationship between applicant income groups and loan approval outcomes.
           We categorize applicants into income bins to see how loan approvals are distributed across different income levels.
           """)

        # Define income bins
    income_bins = [0, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]
    income_labels = ['0-25K', '25K-50K', '50K-100K', '100K-200K', '200K-500K', '500K-1M', '1M-2M', '2M-5M', '5M+']

        # Create a new column 'IncomeGroup' to categorize the income
    df['IncomeGroup'] = pd.cut(df['ApplicantIncome'], bins=income_bins, labels=income_labels, right=False)

        # Calculate Loan Approval Rates by Income Group
    loan_by_income = df.groupby('IncomeGroup')['Loan_Status'].value_counts(normalize=True).unstack() * 100

        # Display Loan Approval Rates
    st.markdown("#### Loan Approval Percentages by Income Group")
    st.write(loan_by_income)

        # Bar Plot for Loan Status by Income Group
    plt.figure(figsize=(10, 6))
    loan_by_income.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Approval Distribution by Applicant Income', fontsize=16, fontweight='bold')
    plt.xlabel('Income Group', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    """
        Perform analysis of loan approval status by Co-applicantIncome, categorized into income bins.
        Visualizes loan approval rates within each income range.
        """
    # Heading and Analysis Description
    st.markdown("### Loan Approval by Co-applicant Income Bins")
    st.write("""
        This section explores the relationship between co-applicant income groups and loan approval outcomes.
        We categorize co-applicants into income bins to see how loan approvals are distributed across different income levels.
        """)

    # Define income bins for co-applicant income
    co_income_bins = [0, 500, 1500, 3000, 5000, 10000, 20000, 50000, 100000]
    co_income_labels = ['0-500', '500-1500', '1500-3000', '3000-5000', '5000-10K', '10K-20K', '20K-50K', '50K+']

    # Create a new column 'CoIncomeGroup' to categorize the co-applicant income
    df['CoIncomeGroup'] = pd.cut(df['CoapplicantIncome'], bins=co_income_bins, labels=co_income_labels, right=False)

    # Calculate Loan Approval Rates by Co-applicant Income Group
    loan_by_co_income = df.groupby('CoIncomeGroup')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    # Display Loan Approval Rates
    st.markdown("#### Loan Approval Percentages by Co-applicant Income Group")
    st.write(loan_by_co_income)

    # Bar Plot for Loan Status by Co-applicant Income Group
    plt.figure(figsize=(10, 6))
    loan_by_co_income.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Approval Distribution by Co-applicant Income', fontsize=16, fontweight='bold')
    plt.xlabel('Co-applicant Income Group', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt

    # Assuming df is the DataFrame with 'LoanAmount' and 'Loan_Status' columns
    # 'Loan_Status' = 0 for Not Granted, 1 for Granted

    # Title for Streamlit app
    st.title('Loan Approval Analysis by Loan Amount')

    # Introduction text
    st.markdown("""
    This section explores the relationship between loan amounts and loan approval outcomes.
    We categorize loan amounts into bins to see how loan approvals are distributed across different loan request ranges.
    """)

    # Define loan amount bins
    loan_amount_bins = [0, 100, 200, 300, 400, 500, 600, 700, 1000]  # In thousands, assuming amounts are in thousands
    loan_amount_labels = ['0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K', '500K-600K', '600K-700K',
                          '700K+']

    # Create a new column 'LoanAmountGroup' to categorize the loan amount
    df['LoanAmountGroup'] = pd.cut(df['LoanAmount'], bins=loan_amount_bins, labels=loan_amount_labels, right=False)

    # Calculate Loan Approval Rates by Loan Amount Group
    loan_by_amount = df.groupby('LoanAmountGroup')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    # Display Loan Approval Rates
    st.markdown("#### Loan Approval Percentages by Loan Amount Group")
    st.write(loan_by_amount)

    # Bar Plot for Loan Status by Loan Amount Group
    st.markdown("#### Loan Approval Distribution by Loan Amount")
    plt.figure(figsize=(10, 6))
    loan_by_amount.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Approval Distribution by Loan Amount', fontsize=16, fontweight='bold')
    plt.xlabel('Loan Amount Group', fontsize=12)
    plt.ylabel('Percentage of Loan Approvals', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Loan Status', labels=['Not Granted', 'Granted'])
    st.pyplot(plt)

    # Box Plot for Loan Amount Distribution by Loan Status
    st.markdown("#### Loan Amount Distribution by Loan Status")
    plt.figure(figsize=(8, 6))
    df.boxplot(column='LoanAmount', by='Loan_Status', patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               flierprops=dict(markerfacecolor='red', marker='o', markersize=8, linestyle='none'))
    plt.title('Loan Amount Distribution by Loan Status')
    plt.suptitle('')
    plt.xlabel('Loan Status')
    plt.ylabel('Loan Amount')
    st.pyplot(plt)

    # Assuming df is the DataFrame with 'Loan_Amount_Term' and 'Loan_Status' columns
    # 'Loan_Status' = 0 for Not Granted, 1 for Granted

    # Title for Streamlit app
    st.title('Loan Approval Analysis by Loan Amount Term')

    # Introduction text
    st.markdown("""
    This section explores the relationship between loan amount terms (in months) and loan approval outcomes.
    We categorize loan terms into bins to see how loan approvals are distributed across different loan terms.
    """)

    # Statistical Summary of Loan_Amount_Term
    st.markdown("### Statistical Summary of Loan Amount Term")
    loan_term_summary = df['Loan_Amount_Term'].describe()
    st.write(loan_term_summary)

    # Define loan amount term bins (in months)
    loan_term_bins = [12, 180, 240, 300, 360, 480]  # Bins based on typical loan terms
    loan_term_labels = ['1-15 years', '15-20 years', '20-25 years', '25-30 years', '30-40 years']

    # Create a new column 'LoanTermGroup' to categorize the loan term
    df['LoanTermGroup'] = pd.cut(df['Loan_Amount_Term'], bins=loan_term_bins, labels=loan_term_labels, right=False)

    # Calculate Loan Approval Rates by Loan Term Group
    loan_by_term = df.groupby('LoanTermGroup')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    # Display Loan Approval Rates
    st.markdown("#### Loan Approval Percentages by Loan Term Group")
    st.write(loan_by_term)

    # Bar Plot for Loan Status by Loan Term Group
    st.markdown("#### Loan Approval Distribution by Loan Term")
    plt.figure(figsize=(10, 6))
    loan_by_term.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Approval Distribution by Loan Term', fontsize=16, fontweight='bold')
    plt.xlabel('Loan Term Group (Months)', fontsize=12)
    plt.ylabel('Percentage of Loan Approvals', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Loan Status', labels=['Not Granted', 'Granted'])
    st.pyplot(plt)

    # Box Plot for Loan Term Distribution by Loan Status
    st.markdown("#### Loan Term Distribution by Loan Status")
    plt.figure(figsize=(8, 6))
    df.boxplot(column='Loan_Amount_Term', by='Loan_Status', patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               flierprops=dict(markerfacecolor='red', marker='o', markersize=8, linestyle='none'))
    plt.title('Loan Term Distribution by Loan Status')
    plt.suptitle('')
    plt.xlabel('Loan Status')
    plt.ylabel('Loan Term (Months)')
    st.pyplot(plt)

    # Histogram for Loan Amount Term Distribution
    st.markdown("#### Distribution of Loan Amount Term")
    plt.figure(figsize=(10, 6))
    df['Loan_Amount_Term'].plot(kind='hist', bins=30, edgecolor='black', color='lightblue', alpha=0.7)
    plt.title('Distribution of Loan Amount Term', fontsize=16, fontweight='bold')
    plt.xlabel('Loan Term (Months)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt)

    # Assuming df is the DataFrame with 'Credit_History' and 'Loan_Status' columns
    # 'Loan_Status' = 0 for Not Granted, 1 for Granted

    # Title for Streamlit app
    st.title('Loan Approval Analysis by Credit History')

    # Introduction text
    st.markdown("""
    This section explores the relationship between credit history and loan approval outcomes.
    We categorize applicants by their credit history (1 for positive, 0 for negative) to see how loan approvals are distributed across these groups.
    """)

    # Statistical Summary of Credit History
    st.markdown("### Statistical Summary of Credit History")
    credit_history_summary = df['Credit_History'].describe()
    st.write(credit_history_summary)

    # Calculate Loan Approval Rates by Credit History
    loan_by_credit_history = df.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True).unstack() * 100

    # Display Loan Approval Rates
    st.markdown("#### Loan Approval Percentages by Credit History")
    st.write(loan_by_credit_history)

    # Bar Plot for Loan Status by Credit History
    st.markdown("#### Loan Approval Distribution by Credit History")
    plt.figure(figsize=(10, 6))
    loan_by_credit_history.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], edgecolor='black')
    plt.title('Loan Approval Distribution by Credit History', fontsize=16, fontweight='bold')
    plt.xlabel('Credit History (0: Negative, 1: Positive)', fontsize=12)
    plt.ylabel('Percentage of Loan Approvals', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Loan Status', labels=['Not Granted', 'Granted'])
    st.pyplot(plt)

    # Box Plot for Loan Amount by Credit History (if applicable)
    st.markdown("#### Loan Amount Distribution by Credit History")
    plt.figure(figsize=(8, 6))
    df.boxplot(column='LoanAmount', by='Credit_History', patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               flierprops=dict(markerfacecolor='red', marker='o', markersize=8, linestyle='none'))
    plt.title('Loan Amount Distribution by Credit History')
    plt.suptitle('')
    plt.xlabel('Credit History')
    plt.ylabel('Loan Amount (₹)')
    st.pyplot(plt)

    # Distribution of Credit History
    st.markdown("#### Distribution of Credit History")
    plt.figure(figsize=(10, 6))
    df['Credit_History'].value_counts().plot(kind='bar', color=['lightblue', 'lightgreen'], edgecolor='black')
    plt.title('Distribution of Credit History', fontsize=16, fontweight='bold')
    plt.xlabel('Credit History (0: Negative, 1: Positive)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt)


def multivariate_analysis(df):
    # Set up the layout for Streamlit
    st.title("Multivariate Analysis")

    st.header("1. Correlation Heatmap (Numerical Columns)")
    # Selecting only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_columns].corr()

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    st.pyplot(plt)

    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix)

    # Statistical T-tests for numerical columns with 'Loan_Status'
    st.header("2. T-Test Analysis Between Numerical Columns and Loan_Status")
    # Perform T-tests
    for col in numerical_columns:
        if col != 'LoanAmount':  # Avoid testing columns like LoanAmount if it's a categorical column
            loan_approved = df[df['Loan_Status'] == 'Y'][col]
            loan_rejected = df[df['Loan_Status'] == 'N'][col]
            t_stat, p_value = ttest_ind(loan_approved, loan_rejected, nan_policy='omit')
            st.subheader(f"{col} - T-test Result")
            st.write(f"T-statistic: {t_stat}")
            st.write(f"P-value: {p_value}")
            if p_value < 0.05:
                st.markdown("### Insight: **Significant relationship** with Loan Status!")
            else:
                st.markdown("### Insight: **No significant relationship** with Loan Status.")

    # Relationship of Categorical Variables with Loan Status
    st.header("3. Categorical Variables Analysis with Loan_Status")
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'Loan_Status']  # exclude Loan_Status

    for col in categorical_columns:
        st.subheader(f"Distribution of {col} with respect to Loan Status")
        # Display the countplot
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, hue='Loan_Status', data=df, palette="Set2")
        st.pyplot(plt)

        # Displaying the insights from the countplot
        contingency_table = pd.crosstab(df[col], df['Loan_Status'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        st.write(f"Chi-squared Test Result for {col} vs Loan_Status:")
        st.write(f"P-value: {p_value}")

        if p_value < 0.05:
            st.markdown("### Insight: **Significant relationship** with Loan Status!")
        else:
            st.markdown("### Insight: **No significant relationship** with Loan Status.")

    # Categorical Heatmap for the relationship between categorical variables
    st.header("4. Categorical Variables Heatmap")
    # Create a contingency table for all categorical columns and Loan_Status
    categorical_data = df[categorical_columns + ['Loan_Status']]
    cat_corr_matrix = pd.DataFrame(index=categorical_data.columns, columns=categorical_data.columns)

    # Calculating Chi-square for all combinations of categorical columns
    for col1 in categorical_data.columns:
        for col2 in categorical_data.columns:
            if col1 != col2:
                contingency = pd.crosstab(categorical_data[col1], categorical_data[col2])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                cat_corr_matrix.loc[col1, col2] = p_value
            else:
                cat_corr_matrix.loc[col1, col2] = np.nan

    # Plotting the heatmap for categorical correlation (Chi-square p-values)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cat_corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', vmin=0, vmax=1)
    st.pyplot(plt)

    # Show summary insights
    st.header("5. Summary Insights")
    st.markdown("""
            - **ApplicantIncome** and **LoanAmount** show weak or no significant correlation with `Loan_Status`.
            - The **Self_Employed** status and **Education** have a high correlation with `Loan_Status`, suggesting they might influence loan approval.
            - Categorical features like **Married**, **Education**, and **Self_Employed** have stronger relationships with `Loan_Status`, making them potentially important predictors for loan approval status.
            """)

# Sample call (assuming you have your dataframe 'df')
# multivariate_analysis(df)





