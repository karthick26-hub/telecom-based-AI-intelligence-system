import streamlit as st
import torch
import pandas as pd
import plotly.express as px

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from rag_system import retrieve_documents

from database import (
    save_feedback,
    fetch_feedback
)


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Telecom AI Brand Intelligence System",
    page_icon="📡",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown(
    """
    <style>

    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #4F46E5,
            #7C3AED
        );

        color: white;

        border-radius: 10px;

        border: none;

        padding: 0.6rem 1.2rem;

        font-weight: bold;

        width: 100%;
    }

    /* Text Area */
    textarea {
        background-color: #161B22 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
    }

    /* Expanders */
    details {
        background-color: #161B22;
        border-radius: 10px;
        padding: 10px;
    }

    /* Titles */
    h1, h2, h3 {
        color: #E6EDF3;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='
        background: linear-gradient(
            90deg,
            #4F46E5,
            #7C3AED
        );
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 20px;
    '>

    <h1 style='color:white;'>
    📡 Telecom AI Brand Intelligence System
    </h1>

    <p style='color:white; font-size:18px;'>
    AI-powered Telecom Analytics using
    BERT + RAG + Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)
# -----------------------------------
# SESSION STATE
# -----------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None


# -----------------------------------
# LOAD MODEL
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
# LABEL MAP
# -----------------------------------

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# -----------------------------------
# SENTIMENT PREDICTION
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
# SERVICE CATEGORY DETECTION
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
# AI RESPONSE GENERATION
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
# LOGIN SCREEN
# -----------------------------------

if not st.session_state.logged_in:

    st.title(
        "🔐 Telecom AI Brand Intelligence System"
    )

    st.markdown(
        """
AI-powered telecom complaint analysis platform
using:
- BERT Transformer
- RAG Architecture
- Semantic Search
- AI Response Generation
"""
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    role = st.selectbox(
        "Select Role",
        [
            "User",
            "Admin"
        ]
    )

    if st.button("Login"):

        # ADMIN LOGIN
        if (
            username == "admin"
            and password == "admin123"
            and role == "Admin"
        ):

            st.session_state.logged_in = True

            st.session_state.role = "Admin"

            st.rerun()

        # USER LOGIN
        elif (
            username == "user"
            and password == "user123"
            and role == "User"
        ):

            st.session_state.logged_in = True

            st.session_state.role = "User"

            st.rerun()

        else:

            st.error(
                "Invalid Credentials"
            )

    st.markdown("---")

    st.subheader("Demo Credentials")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
Admin Login

Username: admin
Password: admin123
"""
        )

    with col2:

        st.info(
            """
User Login

Username: user
Password: user123
"""
        )


# -----------------------------------
# MAIN APP
# -----------------------------------

else:

    # SIDEBAR
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go To",
        [
            "Dashboard"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.success(
        f"Logged in as {st.session_state.role}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False

        st.session_state.role = None

        st.rerun()


    # -----------------------------------
    # USER DASHBOARD
    # -----------------------------------

    if st.session_state.role == "User":

        st.title(
            "📡 Telecom Complaint Portal"
        )

        st.markdown(
            """
Submit telecom complaints and receive:
- Sentiment Analysis
- Service Category Detection
- Telecom Policy Retrieval
- AI-generated Responses
"""
        )

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "AI Model",
                "BERT"
            )

        with col2:
            st.metric(
                "RAG Status",
                "Active"
            )

        with col3:
            st.metric(
                "System Status",
                "Online"
            )

        st.markdown("---")

        # Complaint Input
        user_input = st.text_area(
            "Enter Telecom Complaint",
            height=180
        )

        # Analyze Button
        if st.button(
            "Analyze Complaint"
        ):

            if user_input.strip() != "":

                # Sentiment
                result = predict_sentiment(
                    user_input
                )

                # Category
                category = detect_service(
                    user_input
                )

                # RAG Retrieval
                retrieved_text = retrieve_documents(
                    user_input
                )

                # AI Response
                response = generate_response(
                    result,
                    category,
                    retrieved_text
                )

                # Save Database
                save_feedback(
                    user_input,
                    result,
                    category,
                    response
                )

                st.markdown("---")

                col1, col2 = st.columns(2)

                # Sentiment
                with col1:

                    st.subheader(
                        "Predicted Sentiment"
                    )

                    if result == "Positive":
                        st.success(result)

                    elif result == "Negative":
                        st.error(result)

                    else:
                        st.warning(result)

                # Category
                with col2:

                    st.subheader(
                        "Detected Service Category"
                    )

                    st.info(category)

                # Policies
                st.subheader(
                    "Retrieved Telecom Policies"
                )

                with st.expander(
                    "View Retrieved Policies"
                ):

                    st.write(
                        retrieved_text
                    )

                # AI Response
                st.subheader(
                    "AI Generated Response"
                )

                st.info(response)

            else:

                st.warning(
                    "Please enter complaint text"
                )


    # -----------------------------------
    # ADMIN DASHBOARD
    # -----------------------------------

    elif st.session_state.role == "Admin":

        st.title(
            "📊 Telecom AI Admin Dashboard"
        )

        st.markdown(
            """
Monitor telecom customer complaints,
AI analytics,
and sentiment trends.
"""
        )

        # Fetch Data
        data = fetch_feedback()

        columns = [
            "ID",
            "Complaint",
            "Sentiment",
            "Category",
            "AI_Response",
            "Timestamp"
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        # Metrics
        total_complaints = len(df)

        positive_count = len(
            df[df["Sentiment"] == "Positive"]
        )

        neutral_count = len(
            df[df["Sentiment"] == "Neutral"]
        )

        negative_count = len(
            df[df["Sentiment"] == "Negative"]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Complaints",
                total_complaints
            )

        with col2:
            st.metric(
                "Positive",
                positive_count
            )

        with col3:
            st.metric(
                "Neutral",
                neutral_count
            )

        with col4:
            st.metric(
                "Negative",
                negative_count
            )

        st.markdown("---")

        # Pie Chart
        st.subheader(
            "Sentiment Distribution"
        )

        fig1 = px.pie(
            df,
            names="Sentiment",
            title="Customer Sentiment Analysis"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # Bar Chart
        st.subheader(
            "Service Category Distribution"
        )

        fig2 = px.bar(
            df,
            x="Category",
            color="Sentiment",
            title="Telecom Service Complaints"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # Complaint Table
        st.subheader(
            "Complaint History"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # Latest Complaints
        st.subheader(
            "Latest Complaints"
        )

        latest_df = df.sort_values(
            by="Timestamp",
            ascending=False
        ).head(5)

        for index, row in latest_df.iterrows():

            with st.expander(
                f"{row['Category']} | {row['Sentiment']}"
            ):

                st.write(
                    f"Complaint: {row['Complaint']}"
                )

                st.write(
                    f"AI Response: {row['AI_Response']}"
                )

                st.write(
                    f"Timestamp: {row['Timestamp']}"
                )
# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown(
    """
    <hr>

    <center>

    🚀 Developed using
    BERT Transformers,
    RAG Architecture,
    FAISS Vector Search,
    and Streamlit

    </center>
    """,
    unsafe_allow_html=True
)