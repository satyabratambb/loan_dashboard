import streamlit as st
import p_1
import p_2
import p_3
import p_4


# Set up the Streamlit app configuration
st.set_page_config(
    page_title="Loan Eligibility Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-heading {
        font-size: 2.5rem;
        color: #2E86C1;
        font-weight: bold;
        text-align: center;
    }
    .subheading {
        font-size: 1.5rem;
        color: #2874A6;
        margin-bottom: 15px;
    }
    .module-box {
        background-color: #F2F3F4;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .module-icon {
        font-size: 3rem;
        margin-bottom: 10px;
        color: #2E86C1;
    }
    .footer {
        font-size: 0.9rem;
        color: #5D6D7E;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# Navigation logic
def navigate_to(page_name):
    st.session_state["current_page"] = page_name

# Define each page's content
def home_page():
    # Title Section
    st.markdown('<h1 class="main-heading">💳 Welcome to the Loan Eligibility Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.2rem;'>
            Analyze loan data, explore insights, predict eligibility, and learn about the model performance.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Module Overview Section
    st.markdown('<h2 class="subheading">📌 Explore the Modules</h2>', unsafe_allow_html=True)

    # Layout for modules
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Module 1: How the Model Works
        st.markdown('<div class="module-icon">📊</div>', unsafe_allow_html=True)
        st.markdown("### **How the Model Works**")
        st.write(
            "Discover the dataset used to build our prediction model and the process involved in creating it. "
            "Learn about data preprocessing, feature engineering, and model training."
        )
        if st.button("Learn More", key="module_1"):
            navigate_to("how_model_works")

        # Module 2: Insights & Analytics
        st.markdown('<div class="module-icon">📈</div>', unsafe_allow_html=True)
        st.markdown("### **Insights & Analytics**")
        st.write(
            "Explore interesting trends and patterns in the loan dataset. "
            "Uncover insights like approval rates, key applicant demographics, and more."
        )
        if st.button("View Insights", key="module_2"):
            navigate_to("insights_analytics")

    with col2:
        # Module 3: Loan Eligibility Prediction
        st.markdown('<div class="module-icon">🔍</div>', unsafe_allow_html=True)
        st.markdown("### **Loan Eligibility Prediction**")
        st.write(
            "Enter your details to check if you're eligible for a loan. "
            "Our machine learning model provides quick and accurate results."
        )
        if st.button("Start Prediction", key="module_3"):
            navigate_to("loan_prediction")

        # Module 4: Model Summary
        st.markdown('<div class="module-icon">📂</div>', unsafe_allow_html=True)
        st.markdown("### **Model Summary**")
        st.write(
            "Get a detailed breakdown of the models we used, their accuracy, precision, and other performance metrics. "
            "See how we optimized the models for best results."
        )
        if st.button("View Model Summary", key="module_4"):
            navigate_to("model_summary")

    st.markdown("---")

    # Footer
    st.markdown(
        """
        <div class="footer">
           
            <p>🔒 Your privacy is our priority. We ensure your data remains confidential.</p>
        </div>
        """,
        unsafe_allow_html=True
    )






# Main Page Rendering Logic
if st.session_state["current_page"] == "home":
    home_page()
elif st.session_state["current_page"] == "how_model_works":
    if hasattr(p_1, "how_model_works_page"):
        p_1.how_model_works_page()
    else:
        st.error("Page not found: how_model_works_page")
elif st.session_state["current_page"] == "insights_analytics":
    if hasattr(p_2, "insight_analytics_page"):
        p_2.insight_analytics_page()
    else:
        st.error("Page not found: insight_analytics_page")
elif st.session_state["current_page"] == "loan_prediction":
    if hasattr(p_3, "loan_prediction_page"):
        p_3.loan_prediction_page()
    else:
        st.error("Page not found: insight_analytics_page")

elif st.session_state["current_page"] == "model_summary":
    if hasattr(p_4, "run_loan_prediction_evaluation"):
        p_4.run_loan_prediction_evaluation()
    else:
        st.error("Page not found: insight_analytics_page")


