"""
Script to create/recreate the course_progress table
Run this once to set up the table properly
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'app.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("Creating/updating course_progress table...")

# Create course_progress table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS course_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        topic_id TEXT,
        status TEXT DEFAULT 'not_started',
        progress_percentage REAL DEFAULT 0,
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(course_id) REFERENCES courses(id),
        UNIQUE(user_id, course_id, topic_id)
    )
''')

conn.commit()

# Verify the table
c.execute('PRAGMA table_info(course_progress)')
cols = c.fetchall()
print(f"course_progress table columns: {[col[1] for col in cols]}")

# Check existing records
c.execute('SELECT COUNT(*) FROM course_progress')
count = c.fetchone()[0]
print(f"Existing progress records: {count}")

conn.close()
print("\nDone! Table is ready.")
