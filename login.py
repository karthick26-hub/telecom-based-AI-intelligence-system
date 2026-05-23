import streamlit as st
import subprocess


# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Telecom AI Login",
    page_icon="🔐",
    layout="centered"
)


# -----------------------------------
# Session State
# -----------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None


# -----------------------------------
# Title
# -----------------------------------

st.title("🔐 Telecom AI Brand Intelligence System")

st.markdown(
    """
Login to access:
- User Complaint Portal
- Admin Analytics Dashboard
"""
)


# -----------------------------------
# Login Form
# -----------------------------------

username = st.text_input("Username")

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


# -----------------------------------
# Login Button
# -----------------------------------

if st.button("Login"):

    # Simple Demo Credentials

    if (
        username == "admin"
        and password == "admin123"
        and role == "Admin"
    ):

        st.session_state.logged_in = True
        st.session_state.role = "Admin"

        st.success("Admin Login Successful")

        st.info(
            "Run admin_dashboard.py to access dashboard"
        )

    elif (
        username == "user"
        and password == "user123"
        and role == "User"
    ):

        st.session_state.logged_in = True
        st.session_state.role = "User"

        st.success("User Login Successful")

        st.info(
            "Run app.py to access complaint portal"
        )

    else:

        st.error("Invalid Credentials")


# -----------------------------------
# Demo Credentials
# -----------------------------------

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