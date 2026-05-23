import streamlit as st
import pandas as pd
import plotly.express as px

from database import fetch_feedback

# -----------------------------------
# Authentication Check
# -----------------------------------

if (
    "logged_in" not in st.session_state
    or st.session_state.logged_in is False
):

    st.error("Please login first")

    st.stop()

if st.session_state.role != "Admin":

    st.error("Access Denied")

    st.stop()

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="📊",
    layout="wide"
)


# -----------------------------------
# Dashboard Title
# -----------------------------------

st.title("📊 Telecom AI Admin Dashboard")

st.markdown(
    """
Monitor telecom customer complaints,
sentiment analytics,
and service category trends.
"""
)


# -----------------------------------
# Fetch Database Data
# -----------------------------------

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


# -----------------------------------
# Metrics
# -----------------------------------

total_complaints = len(df)

positive_count = len(
    df[df["Sentiment"] == "Positive"]
)

negative_count = len(
    df[df["Sentiment"] == "Negative"]
)

neutral_count = len(
    df[df["Sentiment"] == "Neutral"]
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


# -----------------------------------
# Sentiment Distribution Chart
# -----------------------------------

st.subheader("Sentiment Distribution")

fig1 = px.pie(
    df,
    names="Sentiment",
    title="Customer Sentiment Analysis"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# -----------------------------------
# Service Category Chart
# -----------------------------------

st.subheader("Service Category Distribution")

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


# -----------------------------------
# Complaint History Table
# -----------------------------------

st.subheader("Complaint History")

st.dataframe(
    df,
    use_container_width=True
)


# -----------------------------------
# Latest Complaints
# -----------------------------------

st.subheader("Latest Complaints")

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
if st.sidebar.button("Logout"):

    st.session_state.logged_in = False

    st.session_state.role = None

    st.success("Logged out successfully")