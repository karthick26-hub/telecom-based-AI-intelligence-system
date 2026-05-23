import streamlit as st
import torch

from transformers import BertTokenizer, BertForSequenceClassification

from rag_system import retrieve_documents
from database import save_feedback, fetch_feedback



# -----------------------------------
# Authentication Check
# -----------------------------------

if (
    "logged_in" not in st.session_state
    or st.session_state.logged_in is False
):

    st.error("Please login first")

    st.stop()

if st.session_state.role != "User":

    st.error("Access Denied")

    st.stop()
# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Telecom AI Brand Intelligence System",
    page_icon="📡",
    layout="wide"
)


# -----------------------------------
# Sidebar Filters
# -----------------------------------

st.sidebar.title("Dashboard Filters")

selected_service = st.sidebar.selectbox(
    "Select Service Category",
    [
        "All",
        "Mobile Network",
        "Broadband",
        "Cloud Services",
        "IoT Services",
        "Billing",
        "Customer Support"
    ]
)

selected_sentiment = st.sidebar.selectbox(
    "Select Sentiment",
    [
        "All",
        "Positive",
        "Neutral",
        "Negative"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
This AI system uses:
- BERT Transformer
- RAG Architecture
- FAISS Vector Search
- Telecom Policy Retrieval
"""
)


# -----------------------------------
# Load Model
# -----------------------------------

model = BertForSequenceClassification.from_pretrained(
    "./sentiment_model"
)

tokenizer = BertTokenizer.from_pretrained(
    "./sentiment_model"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)


# -----------------------------------
# Label Mapping
# -----------------------------------

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# -----------------------------------
# Sentiment Prediction Function
# -----------------------------------

def predict_sentiment(text):

    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

    return label_map[prediction]


# -----------------------------------
# Detect Telecom Service
# -----------------------------------

def detect_service(text):

    text = text.lower()

    if "signal" in text or "network" in text:
        return "Mobile Network"

    elif (
        "wifi" in text
        or "internet" in text
        or "broadband" in text
    ):
        return "Broadband"

    elif "cloud" in text or "server" in text:
        return "Cloud Services"

    elif "iot" in text or "device" in text:
        return "IoT Services"

    elif (
        "bill" in text
        or "payment" in text
        or "recharge" in text
    ):
        return "Billing"

    else:
        return "Customer Support"


# -----------------------------------
# AI Response Generator
# -----------------------------------

def generate_response(
    sentiment,
    category,
    retrieved_text
):

    if sentiment == "Negative":

        return f"""
We are sorry for the inconvenience regarding {category}.

Based on our telecom policies:

{retrieved_text}

Our support team recommends checking the issue and contacting customer support if the problem continues.
"""

    elif sentiment == "Positive":

        return """
Thank you for your valuable positive feedback.

We are happy that our telecom services met your expectations.
"""

    else:

        return """
Thank you for your feedback.

We appreciate your response and will continue improving our telecom services.
"""


# -----------------------------------
# Main Title
# -----------------------------------

st.title(
    "📡 Telecom AI Brand Intelligence System"
)

st.markdown(
    """
AI-powered telecom complaint analysis using:
- BERT Transformer
- Sentiment Analysis
- Service Category Detection
- RAG Document Retrieval
- AI Response Generation
"""
)


# -----------------------------------
# Dashboard Metrics
# -----------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Complaints Processed",
        value="10,000+"
    )

with col2:
    st.metric(
        label="AI Model",
        value="BERT"
    )

with col3:
    st.metric(
        label="RAG Status",
        value="Active"
    )

with col4:
    st.metric(
        label="System Status",
        value="Online"
    )


st.markdown("---")


# -----------------------------------
# Complaint Input
# -----------------------------------

user_input = st.text_area(
    "Enter Telecom Complaint",
    height=180,
    placeholder="Example: My internet speed is extremely slow and customer support is not responding."
)


# -----------------------------------
# Analyze Button
# -----------------------------------

if st.button("Analyze Complaint"):

    if user_input.strip() != "":

        # Predict Sentiment
        result = predict_sentiment(
            user_input
        )

        # Detect Service Category
        category = detect_service(
            user_input
        )

        # Retrieve Telecom Documents
        retrieved_text = retrieve_documents(
            user_input
        )

        # Generate AI Response
        response = generate_response(
            result,
            category,
            retrieved_text
        )
        # Save Feedback to Database
        save_feedback(
            user_input,
            result,
            category,
            response
        )
        # -----------------------------------
        # Results Section
        # -----------------------------------

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Sentiment Output
        with col1:

            st.subheader("Predicted Sentiment")

            if result == "Positive":
                st.success(result)

            elif result == "Negative":
                st.error(result)

            else:
                st.warning(result)

        # Category Output
        with col2:

            st.subheader("Detected Service Category")

            st.info(category)

        # Retrieved Policies
        st.subheader("Retrieved Telecom Policies")

        with st.expander(
            "View Retrieved Telecom Policies"
        ):

            st.write(retrieved_text)

        # AI Response
        st.subheader("AI Generated Response")

        st.info(response)
        
    else:

        st.warning(
            "Please enter complaint text"
        )


# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Telecom AI Brand Intelligence System | Powered by BERT + RAG + Streamlit"
)
if st.sidebar.button("Logout"):

    st.session_state.logged_in = False

    st.session_state.role = None

    st.success("Logged out successfully")