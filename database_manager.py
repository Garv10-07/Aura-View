# file: database_manager.py

import sqlite3
import datetime

class DatabaseManager:
    def __init__(self, db_file="crowd_data.db"):
        """Database se connect karta hai aur table banata hai agar woh pehle se nahin hai."""
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.create_table()

    def create_table(self):
        """'crowd_counts' naam ki table banata hai."""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_count INTEGER NOT NULL,
            timestamp DATETIME NOT NULL
        )
        """)
        self.conn.commit()

    def save_crowd_data(self, count):
        """Table mein naya data (count aur timestamp) save karta hai."""
        timestamp = datetime.datetime.now()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO crowd_counts (person_count, timestamp) VALUES (?, ?)",
            (count, timestamp)
        )
        self.conn.commit()
        print(f"💾 Data Saved: {count} people at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")