import sqlite3


# Create connection
conn = sqlite3.connect(
    "telecom_feedback.db",
    check_same_thread=False
)

cursor = conn.cursor()


# Create table
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS feedback (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        complaint TEXT,

        sentiment TEXT,

        category TEXT,

        ai_response TEXT,

        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
)

conn.commit()


# Function to save feedback
def save_feedback(
    complaint,
    sentiment,
    category,
    ai_response
):

    cursor.execute(
        """
        INSERT INTO feedback
        (
            complaint,
            sentiment,
            category,
            ai_response
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            complaint,
            sentiment,
            category,
            ai_response
        )
    )

    conn.commit()


# Function to fetch all feedback
def fetch_feedback():

    cursor.execute(
        """
        SELECT * FROM feedback
        ORDER BY timestamp DESC
        """
    )

    return cursor.fetchall()