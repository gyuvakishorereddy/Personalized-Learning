import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'app.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Count users
c.execute('SELECT COUNT(*) FROM users')
total = c.fetchone()[0]
print(f'Total users in database: {total}')
print()

# List all users
c.execute('SELECT id, username, email, created_at FROM users ORDER BY id')
users = c.fetchall()

if users:
    print('User List:')
    print('-' * 60)
    for u in users:
        print(f'  ID: {u[0]} | Username: {u[1]} | Email: {u[2]}')
else:
    print('No users found in database.')

# Count courses
c.execute('SELECT COUNT(*) FROM courses')
courses_count = c.fetchone()[0]
print(f'\nTotal courses in database: {courses_count}')

# Check course_progress table
print('\n' + '=' * 60)
print('COURSE PROGRESS:')
print('=' * 60)
try:
    c.execute('SELECT COUNT(*) FROM course_progress')
    progress_count = c.fetchone()[0]
    print(f'Total progress records: {progress_count}')
    
    c.execute('''
        SELECT cp.user_id, u.username, cp.course_id, cp.status, cp.progress_percentage, cp.started_at
        FROM course_progress cp
        JOIN users u ON cp.user_id = u.id
        ORDER BY cp.started_at DESC
    ''')
    progress_records = c.fetchall()
    
    if progress_records:
        print('\nProgress Records:')
        for p in progress_records:
            print(f'  User: {p[1]} | Course ID: {p[2]} | Status: {p[3]} | Progress: {p[4]}%')
    else:
        print('No progress records yet.')
except Exception as e:
    print(f'Error checking course_progress: {e}')

conn.close()
