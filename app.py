import streamlit as st

from predict import predict_email_details


EXAMPLES = {
    "Prize Scam": "Congratulations! You won a $1,000 gift card. Click here now to claim your reward before midnight.",
    "Class Update": "Hi team, please review the project report tonight so we can practice our presentation tomorrow.",
    "Account Alert": "Urgent: your email account will be suspended. Verify your password immediately using this link.",
}


st.set_page_config(page_title="Spam Email Detector", page_icon=":mailbox:", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
        }
        .result-box {
            border: 1px solid #d6d9de;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
        }
        .small-note {
            color: #5f6673;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Spam Email Detector")
st.caption(
    "A Natural Language Processing app using NLTK text cleaning, CountVectorizer, "
    "spelling features, and a Multinomial Naive Bayes classifier."
)

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Check an email")
    selected_example = st.selectbox(
        "Load an example",
        ["Custom"] + list(EXAMPLES.keys()),
    )
    default_text = "" if selected_example == "Custom" else EXAMPLES[selected_example]
    email_input = st.text_area(
        "Email text",
        value=default_text,
        height=240,
        placeholder="Paste the email subject and body here...",
    )

    check_email = st.button("Check Email", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if check_email:
        if not email_input.strip():
            st.warning("Please enter email text before checking.")
        else:
            try:
                details = predict_email_details(email_input)
            except FileNotFoundError as error:
                st.error(str(error))
            else:
                if details["label"] == "Spam":
                    st.error(f"Spam detected ({details['confidence']}% confidence)")
                else:
                    st.success(f"Not spam ({details['confidence']}% confidence)")

                st.progress(details["spam_probability"] / 100)
                st.write(f"Spam probability: {details['spam_probability']}%")
                st.write(f"Not-spam probability: {details['ham_probability']}%")
                st.write(
                    f"Possible misspellings: {details['misspelling_count']} "
                    f"({details['misspelling_ratio']}% of checked words)"
                )
                if details["possible_misspellings"]:
                    st.write(
                        "Flagged words: "
                        + ", ".join(details["possible_misspellings"])
                    )
    else:
        st.markdown(
            "<div class='result-box small-note'>Paste an email and run the detector to see the model's prediction.</div>",
            unsafe_allow_html=True,
        )
