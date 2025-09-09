import sqlite3
import os

DATABASE = "audio_recordings.db"


def initialize_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS recordings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        filepath TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()


def save_recording_info(filename):
    """Insert a recording and store the full path correctly."""
    initialize_database()  # Ensure table exists

    filename = os.path.basename(filename)
    full_path = os.path.join(os.getcwd(), "recordings", filename)  # Add recordings/
    full_path = os.path.abspath(full_path)  # Get absolute path

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO recordings (filename, filepath) VALUES (?, ?)",
                   (filename, full_path))

    conn.commit()
    conn.close()

    #print(f"✅ Stored recording: {filename} at {full_path}")

def fetch():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM recordings;")
    rows = cursor.fetchall()

    conn.close()

    #for row in rows:  # Print stored data
        #print(f"🔍 DB Row: {row}")

    return rows

