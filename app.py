import streamlit as st

from predict import predict_email_details


EXAMPLES = {
    "Prize Scam": "Congratulations! You won a $1,000 gift card. Click here now to claim your reward before midnight.",
    "Class Update": "Hi team, please review the project report tonight so we can practice our presentation tomorrow.",
    "Account Alert": "Urgent: your email account will be suspended. Verify your password immediately using this link.",
}


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .hero-box {
        background: linear-gradient(135deg, #f3f6ff, #eef2ff);
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #dbe3ff;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #111827;
    }

    .hero-subtitle {
        color: #4b5563;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .card {
        background-color: white;
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .metric-box {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-number {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6b7280;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HERO SECTION ----------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📧 Spam Email Detector</div>
        <div class="hero-subtitle">
            Detect suspicious emails using Natural Language Processing,
            machine learning, and spam-related text analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("Project Information")

    st.write(
        """
        This application uses:
        - NLTK preprocessing
        - CountVectorizer
        - Feature engineering
        - Multinomial Naive Bayes
        """
    )

    st.divider()

    st.subheader("How To Use")

    st.write(
        """
        1. Paste an email message
        2. Click **Check Email**
        3. Review the prediction results
        """
    )

# ---------------- MAIN LAYOUT ----------------
input_col, output_col = st.columns([1.2, 1])

# ---------------- INPUT COLUMN ----------------
with input_col:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Email Input")

    selected_example = st.selectbox(
        "Choose an example",
        ["Custom"] + list(EXAMPLES.keys()),
    )

    default_text = (
        ""
        if selected_example == "Custom"
        else EXAMPLES[selected_example]
    )

    email_input = st.text_area(
        "Paste email text",
        value=default_text,
        height=280,
        placeholder="Paste the email subject and body here...",
    )

    st.caption(f"Character count: {len(email_input)}")

    check_email = st.button(
        "Check Email",
        type="primary",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- OUTPUT COLUMN ----------------
with output_col:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Detection Results")

    if check_email:

        if not email_input.strip():

            st.warning("Please enter email text before checking.")

        else:

            with st.spinner("Analyzing email..."):

                try:
                    details = predict_email_details(email_input)

                except FileNotFoundError as error:
                    st.error(str(error))

                else:

                    # Prediction message
                    if details["label"] == "Spam":

                        st.error(
                            f"⚠️ Spam detected ({details['confidence']}% confidence)"
                        )

                    else:

                        st.success(
                            f"✅ Email appears safe ({details['confidence']}% confidence)"
                        )

                    # Progress bar
                    st.progress(details["spam_probability"] / 100)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Metrics
                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-number">
                                    {details['spam_probability']}%
                                </div>
                                <div class="metric-label">
                                    Spam Probability
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with metric_col2:

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-number">
                                    {details['ham_probability']}%
                                </div>
                                <div class="metric-label">
                                    Safe Probability
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Additional details
                    st.info(
                        f"""
Possible misspellings: {details['misspelling_count']}

Misspelling ratio: {details['misspelling_ratio']}%
"""
                    )

                    if details["possible_misspellings"]:

                        st.write("### Flagged Words")

                        st.write(
                            ", ".join(details["possible_misspellings"])
                        )

                    else:

                        st.write("### Flagged Words")
                        st.write("No suspicious misspellings detected.")

    else:

        st.info(
            "Paste an email message and click 'Check Email' to begin analysis."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Built with Streamlit, Python, Scikit-learn, and NLTK
    </div>
    """,
    unsafe_allow_html=True,
)
