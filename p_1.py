import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def navigate_to(page_name):
    st.session_state["current_page"] = page_name


def how_model_works_page():
    # Navigation Button
    if st.button("⬅ Back to Home", key="back_button"):
        # Add navigation logic here, e.g., changing a page variable
        st.write("Navigating back to home...")
        navigate_to("home")

    # Title Section - Main Heading
    st.markdown('<h1 class="main-heading">📊 How the Model Works</h1>', unsafe_allow_html=True)
    st.write(
        """
        In this section, we explore the dataset and features that drive the loan eligibility prediction model. 
        Each feature plays an essential role in determining whether a loan application gets approved or denied. 
        Let’s dive into the features and their importance.
        """
    )
    st.markdown("---")

    # About the Training Set Subsection
    st.markdown("<h3 style='color:#2874A6;'>📊 About the Training Set</h3>", unsafe_allow_html=True)

    # Null Values Section
    st.markdown("### Null Values in the Dataset")
    null_values = {
        "Loan_ID": 0,
        "Gender": 15,
        "Married": 3,
        "Dependents": 15,
        "Education": 1,
        "Self_Employed": 32,
        "ApplicantIncome": 2,
        "CoapplicantIncome": 1,
        "LoanAmount": 22,
        "Loan_Amount_Term": 14,
        "Credit_History": 50,
        "property_Area": 0,
        "Loan_Status": 0
    }

    null_df = pd.DataFrame(list(null_values.items()), columns=["Feature", "Null Values"])
    st.table(null_df)

    st.markdown("---")

    # Statistics Section
    st.markdown("### Summary Statistics of the Training Set")
    stats = {
        "Feature": ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"],
        "Count": [612, 613, 592, 600, 564],
        "Mean": [5405.54, 1620.89, 146.41, 342.00, 0.842],
        "Std": [6118.91, 2928.62, 85.59, 65.12, 0.365],
        "Min": [150.00, 0.00, 9.00, 12.00, 0.00],
        "25%": [2875.75, 0.00, 100.00, 360.00, 1.00],
        "50%": [3806.00, 1167.00, 128.00, 360.00, 1.00],
        "75%": [5803.75, 2302.00, 168.00, 360.00, 1.00],
        "Max": [81000.00, 41667.00, 700.00, 480.00, 1.00]
    }

    stats_df = pd.DataFrame(stats)
    st.table(stats_df)

    st.markdown("---")

    # Feature Description Section
    st.markdown("<h3 style='color:#2874A6;'>📋 Dataset Features</h3>", unsafe_allow_html=True)

    # Feature Descriptions
    features = {
        "Gender": """
            <p><strong>Description:</strong> 
            The <strong>gender</strong> of the applicant (Male/Female). While this feature might not directly influence eligibility, 
            societal trends and historical data may show patterns in loan approval related to gender differences, such as employment rates or income levels.</p>

            <p><strong>Why it's Important:</strong> 
            Helps the model account for potential socio-economic trends in the data.</p>
        """,
        "Married": """
            <p><strong>Description:</strong> 
            Whether the applicant is <strong>married</strong> or not. Married applicants may have more stability due to family support, 
            or financial responsibilities that can affect loan repayment.</p>

            <p><strong>Why it's Important:</strong> 
            Marital status can impact financial behavior, providing insight into an applicant's long-term financial stability.</p>
        """,
        "Dependents": """
            <p><strong>Description:</strong> 
            The number of <strong>dependents</strong> the applicant supports. This feature can indicate the applicant’s financial burden, 
            where more dependents could reduce disposable income, impacting loan repayment capacity.</p>

            <p><strong>Why it's Important:</strong> 
            It gives an idea of how much financial responsibility an applicant has, which can influence loan eligibility.</p>
        """,
        "Education": """
            <p><strong>Description:</strong> 
            The educational background of the applicant (Graduate/Not Graduate). A higher level of education often correlates with 
            better job prospects and income, which can influence loan approval decisions.</p>

            <p><strong>Why it's Important:</strong> 
            Education level impacts earning potential, and graduates are typically more financially stable.</p>
        """,
        "Self_Employed": """
            <p><strong>Description:</strong> 
            Whether the applicant is <strong>self-employed</strong> or works for an employer. Self-employed individuals may have fluctuating income, 
            which could present risks for loan repayment.</p>

            <p><strong>Why it's Important:</strong> 
            Income stability is key in assessing loan eligibility, and self-employment can introduce variability in this area.</p>
        """,
        "ApplicantIncome": """
            <p><strong>Description:</strong> 
            The <strong>income</strong> of the applicant. A higher income is often an indicator of a better ability to repay loans, thus improving eligibility.</p>

            <p><strong>Why it's Important:</strong> 
            Income is one of the most crucial factors in determining whether an applicant can repay a loan.</p>
        """,
        "CoapplicantIncome": """
            <p><strong>Description:</strong> 
            The income of the <strong>coapplicant</strong> (if any). When combined with the applicant's income, it enhances the overall financial profile, 
            providing a stronger basis for loan eligibility.</p>

            <p><strong>Why it's Important:</strong> 
            Coapplicant income helps assess the overall financial strength of the applicant, increasing the likelihood of loan approval.</p>
        """,
        "LoanAmount": """
            <p><strong>Description:</strong> 
            The total <strong>loan amount</strong> applied for. The loan amount should align with the applicant’s ability to repay. Excessive loan amounts 
            may reduce approval chances due to repayment concerns.</p>

            <p><strong>Why it's Important:</strong> 
            A loan amount that is too high relative to income and other factors may signal risk to the lender.</p>
        """,
        "Loan_Amount_Term": """
            <p><strong>Description:</strong> 
            The <strong>loan term</strong> in months. Shorter terms may have higher monthly payments, while longer terms can lower payments but may increase total interest.</p>

            <p><strong>Why it's Important:</strong> 
            The term affects the applicant’s ability to repay. Too long of a loan term might be perceived as a financial strain.</p>
        """,
        "Credit_History": """
            <p><strong>Description:</strong> 
            Whether the applicant has a <strong>positive credit history</strong>. A good credit history signifies reliability and reduces the risk to the lender, thus increasing eligibility.</p>

            <p><strong>Why it's Important:</strong> 
            A positive credit history is one of the most significant indicators of whether the applicant can reliably repay a loan.</p>
        """,
        "Property_Area": """
            <p><strong>Description:</strong> 
            The area where the property is located (<strong>Urban/Semiurban/Rural</strong>). Properties in urban areas tend to have higher values, which could impact the loan amount.</p>

            <p><strong>Why it's Important:</strong> 
            The type of property area can affect the overall value of the loan and the applicant’s stability in repaying it.</p>
        """,
        "Loan_Status": """
            <p><strong>Description:</strong> 
            The <strong>target variable</strong> indicating whether the loan was approved or not. The status (approved/rejected) is the outcome our model predicts.</p>

            <p><strong>Why it's Important:</strong> 
            This is the final decision that our model aims to predict based on all the other features.</p>
        """
    }

    # Display each feature with improved presentation and spacing
    for feature, description in features.items():
        st.markdown(f"""
            <div style="background-color:#F2F3F4; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="color:#2874A6; font-size: 1.2rem; font-weight: bold;">{feature}</h4>
                {description}
            </div>
        """, unsafe_allow_html=True)

    # Data Cleaning & Imputing Missing Values Section
    st.markdown("<h3 style='color:#2874A6;'>🧹 Data Cleaning & Handling Missing Values</h3>", unsafe_allow_html=True)

    # Section Description
    st.write("""
        In this section, we discuss how we handled missing values in the dataset. Rather than directly applying mean, median, or mode imputation, 
        we took a strategic approach by observing the relationships between the features and considering class imbalances. 
        This allowed us to make more informed decisions about how to fill the missing values, ensuring that the imputation 
        aligned with the underlying patterns in the data.
    """)

    st.markdown("### Missing Values Imputation Approach")

    st.write("""
        We identified the features with missing values and applied different imputation techniques based on the type of data and the 
        relationships between features. Instead of just using statistical measures like mean or median, we considered:
        - **Class Imbalances**: Ensuring that the imputation didn't introduce or worsen class imbalances.
        - **Feature Relationships**: Imputing values in a way that respected the correlations between features.
    """)

    # Handling Missing Values in Gender
    st.markdown("### **Gender**")
    st.write("""
        The **Gender** column had missing values. As the dataset was relatively small and there was class imbalance, 
        we noticed the following proportions of gender in the data:
        - **Male**: 81.3%
        - **Female**: 18.7%

        To handle the missing values, we decided to maintain this class ratio by strategically imputing the missing values. 
        We randomly imputed 81% of the missing values as **Male** and the remaining 19% as **Female**, 
        preserving the original distribution in the data. No other significant relationships were found to impute the missing values based on other features.
    """)

    # Handling Missing Values in Married
    st.markdown("### **Married**")
    st.write("""
        The **Married** column had missing values, and the class distribution was as follows:
        - **Married (True)**: 3 records
        - **Not Married (False)**: 611 records

        **Step**: Imputing Missing Values in the Married Column  
        In this step, we address the missing values in the **Married** column, which is a categorical variable indicating whether the applicant is married ("Yes" or "No").

        **Imputation Method**: We used **Mode Imputation**, where missing values are replaced by the most frequent category (mode). This approach is particularly suitable when the number of missing values is small, as is the case with the **Married** column in our dataset. Since the number of missing values is minimal, imputing with the mode ensures that the missing data does not have a significant impact on the analysis.

        **Chosen Approach**: The most frequent value in the **Married** column is **Not Married (False)**. Therefore, we chose to impute all missing values with the **Not Married** label to maintain consistency with the existing distribution.
    """)

    st.markdown("#### **Dependents Column**")
    st.write(
        """
        In the Dependents column, we observed that most self-employed individuals and females had no dependents 
        (i.e., value = 0). We used this observation to strategically impute the missing values. 
        Most missing values were filled with '0', as this was the most common value.
        Additionally, the class imbalance (599 with 0 dependents, 15 with dependents) was considered while filling missing values.
        """
    )

    # Education Column Imputation
    st.markdown("#### **Education Column**")
    st.write(
        """
        The Education column had only one missing value. As there was only one missing value, we used mode imputation to replace it. 
        The mode value was 'Graduate', so the missing value was imputed with 'Graduate' to maintain the consistency of the dataset.
        """
    )
    st.markdown("#### **Self_Employed Column**")
    st.write(
        """
        The Self_Employed column had missing values, but no significant pattern was observed. 
        To maintain consistency, we imputed the missing values with the mode, which was 'No'. 
        This approach ensures that the dataset is complete while reflecting the general trend of applicants being non-self-employed.
        """
    )

    st.markdown("#### **ApplicantIncome Column**")
    st.write(
        """
        The ApplicantIncome column had a few missing values. We observed that different groups of dependents had significantly different income distributions. 
        Therefore, rather than imputing the missing values with a global mean, we decided to impute the missing values based on the group the applicant belongs to.
        Specifically, we filled the missing values by calculating the mean income within each dependent category, ensuring the imputation reflects the income pattern for each group.
        """
    )

    # CoapplicantIncome Column Imputation
    st.markdown("#### **CoapplicantIncome Column**")
    st.write(
        """
        The CoapplicantIncome column had only one missing value. We observed that applicants who were married typically had a higher coapplicant income. 
        Therefore, we imputed the missing value based on the applicant's marital status, specifically filling the missing value with the mean coapplicant income for applicants in the same marital status group.
        """
    )

    # LoanAmount Column Imputation
    st.markdown("#### **LoanAmount Column**")
    st.write(
        """
        The LoanAmount column had 22 missing values. Given the class imbalance in the dataset, where the majority of the applicants were married, 
        we observed that the loan amounts for married individuals tended to be higher. To impute the missing values, we grouped the data by the 
        'Married' column and replaced the missing LoanAmount values with the mean loan amount of each group.
        """
    )

    # Loan_Amount_Term Column Imputation
    st.markdown("#### **Loan_Amount_Term Column**")
    st.write(
        """
        The Loan_Amount_Term column had 14 missing values. As there were no significant relationships observed with other features and given the 
        nature of the column (which represents the term of the loan), we imputed the missing values using the mode (most frequent value) of the 
        existing values. This approach ensures that the missing data is filled with a value that most accurately represents the majority of the dataset.
        """
    )

    # Credit_History Column Imputation
    st.markdown("#### **Credit_History Column**")
    st.write(
        """
        The Credit_History column had 50 missing values. Given the absence of significant relationships with other features and the nature of 
        this column (indicating whether an applicant has a good credit history), we imputed the missing values using the mode (most frequent value). 
        This ensures that the missing data is filled with a value that most accurately represents the majority of the dataset, providing a more 
        reliable imputation while maintaining consistency across the data.
        """
    )

    st.markdown("---")

    # Feature Importance Visualization
    st.markdown("<h3 style='color:#2874A6;'>📊 Feature Importance</h3>", unsafe_allow_html=True)
    st.write(
        "Feature importance helps us understand which features contribute most to the loan eligibility prediction.")

    # Add more details as needed...
