import sqlite3

conn = sqlite3.connect("legal_contracts.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE contract_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        persona TEXT,
        user_context TEXT,
        contract_text TEXT,
        summary TEXT,
        risk_score REAL,
        risk_analysis TEXT,
        clauses_json TEXT,
        visualization_data TEXT
    )
""")

cursor.execute("""
    CREATE TABLE user_sessions (
        session_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

conn.commit()
conn.close()

print("Database initialized!")
