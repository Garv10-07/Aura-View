import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="crowd_data.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()

        # Crowd Data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS crowd_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            count INTEGER NOT NULL
        )
        """)

        # Settings table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

        # Users table (login)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    # ------------------------
    # Crowd data functions
    # ------------------------
    def save_crowd_data(self, count: int):
        conn = self._connect()
        cur = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("INSERT INTO crowd_data(timestamp, count) VALUES(?,?)", (ts, count))
        conn.commit()
        conn.close()
        print(f"💾 Data Saved: {count} people at {ts}")

    # ------------------------
    # Settings
    # ------------------------
    def set_setting(self, key: str, value: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO settings(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        conn.commit()
        conn.close()

    def get_setting(self, key: str, default=None):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        return default

    # ------------------------
    # Users (Login)
    # ------------------------
    def create_user(self, username: str, password_hash: str):
        conn = self._connect()
        cur = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
            (username, password_hash, created_at),
        )
        conn.commit()
        conn.close()

    def get_user(self, username: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        return row
