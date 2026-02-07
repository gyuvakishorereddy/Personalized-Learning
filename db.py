import os
import sqlite3
import uuid
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()
    
    # Initialize database tables (do NOT drop existing tables to preserve user data)
    cursor.executescript(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            bio TEXT,
            learning_goals TEXT,
            avatar_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_guest INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS programming_languages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            icon_class TEXT
        );

        CREATE TABLE IF NOT EXISTS assessment_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            options TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            difficulty_level TEXT NOT NULL,
            category TEXT,
            explanation TEXT,
            FOREIGN KEY(language_id) REFERENCES programming_languages(id)
        );

        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            language TEXT,
            difficulty TEXT,
            level TEXT,
            prerequisites TEXT
        );

        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            topic_id TEXT,
            status TEXT NOT NULL,
            start_time TEXT,
            completion_time TEXT,
            time_spent INTEGER DEFAULT 0,
            concentration_score REAL DEFAULT 0,
            progress_percentage REAL DEFAULT 0,
            module_index INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id)
        );

        CREATE TABLE IF NOT EXISTS sub_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            topic_id TEXT NOT NULL,
            sub_exercise_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            content TEXT,
            exercise_type TEXT NOT NULL, -- 'theory', 'example', 'practice', 'quiz', 'project'
            difficulty TEXT DEFAULT 'beginner',
            estimated_time INTEGER DEFAULT 10, -- minutes
            prerequisites TEXT, -- JSON array of required sub-exercise indices
            learning_objectives TEXT, -- JSON array of learning goals
            instructions TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            UNIQUE(course_id, topic_id, sub_exercise_index)
        );

        CREATE TABLE IF NOT EXISTS sub_exercise_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            topic_id TEXT NOT NULL,
            sub_exercise_id INTEGER NOT NULL,
            status TEXT NOT NULL, -- 'not_started', 'in_progress', 'completed', 'skipped'
            start_time TEXT,
            completion_time TEXT,
            time_spent INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            concentration_score REAL DEFAULT 0,
            emotion_data TEXT, -- JSON array of emotion tracking data
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(sub_exercise_id) REFERENCES sub_exercises(id),
            UNIQUE(user_id, course_id, topic_id, sub_exercise_id)
        );

        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            emotion TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            course_id INTEGER,
            topic_id TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id)
        );

        CREATE TABLE IF NOT EXISTS learning_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            topic_id TEXT NOT NULL,
            sub_exercise_id INTEGER,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(sub_exercise_id) REFERENCES sub_exercises(id)
        );

        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            quiz_title TEXT,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            language TEXT,
            difficulty TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            UNIQUE(user_id, course_id)
        );

        CREATE TABLE IF NOT EXISTS module_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            module_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            score REAL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, module_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

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
        );
        '''
    )
    connection.commit()

    # Initialize database only

    connection.close()


import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str, is_guest: bool = False, email: str = None) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    hashed_password = hash_password(password)
    cursor.execute(
        '''INSERT INTO users 
           (username, password, email, is_guest, created_at, updated_at) 
           VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))''',
        (username, hashed_password, email, 1 if is_guest else 0),
    )
    connection.commit()
    user_id = cursor.lastrowid
    connection.close()
    return user_id


def get_user_by_credentials(username: str, password: str):
    connection = get_connection()
    cursor = connection.cursor()
    hashed_password = hash_password(password)
    cursor.execute(
        'SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password)
    )
    row = cursor.fetchone()
    connection.close()
    return row


def get_user_by_email_credentials(email: str, password: str):
    """Authenticate user by email and password"""
    connection = get_connection()
    cursor = connection.cursor()
    hashed_password = hash_password(password)
    cursor.execute(
        'SELECT * FROM users WHERE email = ? AND password = ? AND is_guest = 0', (email, hashed_password)
    )
    row = cursor.fetchone()
    connection.close()
    return row


def get_user_by_id(user_id: int):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    connection.close()
    return row


def list_courses():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM courses ORDER BY language, difficulty, id ASC')
    courses = cursor.fetchall()
    connection.close()
    return courses


def get_courses_by_language_and_difficulty(language: str, difficulty: str):
    """Get courses filtered by language and difficulty"""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'SELECT * FROM courses WHERE language = ? AND difficulty = ? ORDER BY id ASC',
        (language, difficulty)
    )
    courses = cursor.fetchall()
    connection.close()
    return courses


def add_course(title: str, description: str = '', language: str = 'general', difficulty: str = 'beginner', level: str = 'Beginner') -> int:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'INSERT INTO courses (title, description, language, difficulty, level) VALUES (?, ?, ?, ?, ?)',
        (title, description, language, difficulty, level)
    )
    connection.commit()
    course_id = cursor.lastrowid
    connection.close()
    return course_id


def save_progress(user_id: int, course_id: int, status: str):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'INSERT INTO progress (user_id, course_id, status, start_time) VALUES (?, ?, ?, ?)',
        (user_id, course_id, status, datetime.utcnow().isoformat())
    )
    connection.commit()
    connection.close()


def save_emotion(user_id: int, emotion: str, timestamp: str | None = None, course_id: int = None, topic_id: str = None):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'INSERT INTO emotions (user_id, emotion, timestamp, course_id, topic_id) VALUES (?, ?, ?, ?, ?)',
        (user_id, emotion, timestamp or datetime.utcnow().isoformat(), course_id, topic_id),
    )
    connection.commit()
    connection.close()


def save_learning_emotion(user_id: int, emotion: str, module_type: str, course_id: int = None, topic_id: str = None, quiz_id: str = None):
    """Save emotion with learning context for analytics"""
    connection = get_connection()
    cursor = connection.cursor()
    timestamp = datetime.utcnow().isoformat()
    
    # Save to emotions table with context
    cursor.execute(
        'INSERT INTO emotions (user_id, emotion, timestamp, course_id, topic_id) VALUES (?, ?, ?, ?, ?)',
        (user_id, emotion, timestamp, course_id, topic_id),
    )
    
    # Could also save to a more detailed learning_analytics table if needed
    # For now, we'll use the existing emotions table structure
    
    connection.commit()
    connection.close()


def get_emotion_analytics(user_id: int, course_id: int = None, topic_id: str = None, hours: int = 24):
    """Get emotion analytics for learning comprehension assessment"""
    connection = get_connection()
    cursor = connection.cursor()
    
    # Base query
    query = '''
        SELECT emotion, timestamp, course_id, topic_id 
        FROM emotions 
        WHERE user_id = ? 
        AND datetime(timestamp) >= datetime('now', '-{} hours')
    '''.format(hours)
    
    params = [user_id]
    
    # Add filters if provided
    if course_id:
        query += " AND course_id = ?"
        params.append(course_id)
    
    if topic_id:
        query += " AND topic_id = ?"
        params.append(topic_id)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    emotions = cursor.fetchall()
    connection.close()
    
    # Process emotions for analytics
    if not emotions:
        return {
            'understanding_level': 'neutral',
            'confidence_score': 0.5,
            'emotion_distribution': {},
            'learning_state': 'unknown',
            'recommendations': []
        }
    
    emotion_counts = {}
    for emotion_row in emotions:
        emotion = emotion_row['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total_emotions = len(emotions)
    
    # Calculate understanding metrics
    positive_emotions = ['happy', 'excited', 'confident', 'focused']
    negative_emotions = ['confused', 'frustrated', 'bored', 'anxious']
    
    positive_count = sum(emotion_counts.get(emotion, 0) for emotion in positive_emotions)
    negative_count = sum(emotion_counts.get(emotion, 0) for emotion in negative_emotions)
    
    confidence_score = (positive_count - negative_count) / total_emotions + 0.5
    confidence_score = max(0, min(1, confidence_score))  # Clamp to 0-1
    
    # Determine understanding level
    if confidence_score >= 0.7:
        understanding_level = 'good'
        learning_state = 'engaged'
    elif confidence_score >= 0.4:
        understanding_level = 'moderate'
        learning_state = 'learning'
    else:
        understanding_level = 'struggling'
        learning_state = 'needs_support'
    
    # Generate recommendations
    recommendations = []
    if understanding_level == 'struggling':
        recommendations.extend([
            'Consider reviewing previous topics',
            'Take a short break',
            'Try a different learning approach'
        ])
    elif understanding_level == 'moderate':
        recommendations.extend([
            'Practice with additional exercises',
            'Review key concepts'
        ])
    else:
        recommendations.extend([
            'Continue to next topic',
            'Try advanced exercises'
        ])
    
    # Calculate emotion distribution percentages
    emotion_distribution = {
        emotion: (count / total_emotions) * 100 
        for emotion, count in emotion_counts.items()
    }
    
    return {
        'understanding_level': understanding_level,
        'confidence_score': confidence_score,
        'emotion_distribution': emotion_distribution,
        'learning_state': learning_state,
        'recommendations': recommendations,
        'total_emotions_tracked': total_emotions,
        'recent_emotion': emotions[0]['emotion'] if emotions else 'neutral'
    }


def insert_quiz_result(user_id: int, quiz_title: str, score: int, total: int, language: str = None, difficulty: str = None) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'INSERT INTO quiz_results (user_id, quiz_title, score, total, language, difficulty, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, quiz_title, score, total, language, difficulty, datetime.utcnow().isoformat()),
    )
    connection.commit()
    quiz_result_id = cursor.lastrowid
    connection.close()
    return quiz_result_id


def get_quiz_result(quiz_result_id: int):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM quiz_results WHERE id = ?', (quiz_result_id,))
    row = cursor.fetchone()
    connection.close()
    return row


def get_recent_emotions(user_id: int, limit: int = 50):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        'SELECT emotion, timestamp FROM emotions WHERE user_id = ? ORDER BY id DESC LIMIT ?',
        (user_id, limit),
    )
    rows = cursor.fetchall()
    connection.close()
    return rows


def get_programming_languages():
    """Get all available programming languages"""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM programming_languages ORDER BY name ASC')
    languages = cursor.fetchall()
    connection.close()
    return languages

def update_user_profile(user_id: int, data: dict) -> bool:
    """Update user profile information"""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        update_fields = []
        params = []
        
        # Build dynamic update query based on provided fields
        for field in ['username', 'email', 'bio', 'learning_goals', 'avatar_url', 'password']:
            if field in data and data[field]:
                update_fields.append(f"{field} = ?")
                params.append(data[field])
        
        if update_fields:
            update_fields.append("updated_at = datetime('now')")
            query = f'''
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE id = ?
            '''
            params.append(user_id)
            
            cursor.execute(query, params)
            connection.commit()
            connection.close()
            return True
            
        return False
    except Exception:
        return False

def get_student_data(user_id: int) -> dict:
    # Aggregate minimal data for ML model input
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute('SELECT COUNT(*) AS cnt FROM progress WHERE user_id = ?', (user_id,))
    courses_started = cursor.fetchone()['cnt']

    cursor.execute('SELECT AVG(score) AS avg_score FROM quiz_results WHERE user_id = ?', (user_id,))
    avg_score_row = cursor.fetchone()
    avg_score = avg_score_row['avg_score'] if avg_score_row and avg_score_row['avg_score'] is not None else 0

    cursor.execute('SELECT emotion FROM emotions WHERE user_id = ? ORDER BY id DESC LIMIT 1', (user_id,))
    last_emotion_row = cursor.fetchone()
    last_emotion = last_emotion_row['emotion'] if last_emotion_row else 'neutral'

    # Get recent quiz performance
    cursor.execute('SELECT language, difficulty, score, total FROM quiz_results WHERE user_id = ? ORDER BY id DESC LIMIT 5', (user_id,))
    recent_quizzes = cursor.fetchall()

    connection.close()
    return {
        'courses_started': courses_started,
        'avg_score': avg_score,
        'last_emotion': last_emotion,
        'recent_quizzes': recent_quizzes
    }


# Sub-Exercise Management Functions

def create_sub_exercise(course_id: int, topic_id: str, sub_exercise_index: int,
                       title: str, description: str, content: str, exercise_type: str,
                       difficulty: str = 'beginner', estimated_time: int = 10,
                       prerequisites: list = None, learning_objectives: list = None,
                       instructions: str = '') -> int:
    """Create a new sub-exercise"""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute('''
        INSERT INTO sub_exercises
        (course_id, topic_id, sub_exercise_index, title, description, content,
         exercise_type, difficulty, estimated_time, prerequisites, learning_objectives, instructions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        course_id, topic_id, sub_exercise_index, title, description, content,
        exercise_type, difficulty, estimated_time,
        json.dumps(prerequisites or []), json.dumps(learning_objectives or []), instructions
    ))

    connection.commit()
    sub_exercise_id = cursor.lastrowid
    connection.close()
    return sub_exercise_id


def get_sub_exercises(course_id: int, topic_id: str) -> list:
    """Get all sub-exercises for a topic"""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute('''
        SELECT * FROM sub_exercises
        WHERE course_id = ? AND topic_id = ?
        ORDER BY sub_exercise_index ASC
    ''', (course_id, topic_id))

    sub_exercises = cursor.fetchall()
    connection.close()

    # Parse JSON fields
    result = []
    for sub_ex in sub_exercises:
        sub_ex_dict = dict(sub_ex)
        try:
            sub_ex_dict['prerequisites'] = json.loads(sub_ex['prerequisites'] or '[]')
            sub_ex_dict['learning_objectives'] = json.loads(sub_ex['learning_objectives'] or '[]')
        except json.JSONDecodeError:
            sub_ex_dict['prerequisites'] = []
            sub_ex_dict['learning_objectives'] = []
        result.append(sub_ex_dict)

    return result


def get_sub_exercise_progress(user_id: int, course_id: int, topic_id: str) -> dict:
    """Get user's progress on all sub-exercises for a topic"""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute('''
        SELECT se.*, sep.status, sep.start_time, sep.completion_time,
               sep.time_spent, sep.attempts, sep.score, sep.concentration_score
        FROM sub_exercises se
        LEFT JOIN sub_exercise_progress sep ON se.id = sep.sub_exercise_id
            AND sep.user_id = ? AND sep.course_id = ? AND sep.topic_id = ?
        WHERE se.course_id = ? AND se.topic_id = ?
        ORDER BY se.sub_exercise_index ASC
    ''', (user_id, course_id, topic_id, course_id, topic_id))

    results = cursor.fetchall()
    connection.close()

    progress = {}
    for row in results:
        sub_ex_id = row['id']
        progress[sub_ex_id] = {
            'sub_exercise': dict(row),
            'status': row['status'] or 'not_started',
            'start_time': row['start_time'],
            'completion_time': row['completion_time'],
            'time_spent': row['time_spent'] or 0,
            'attempts': row['attempts'] or 0,
            'score': row['score'] or 0,
            'concentration_score': row['concentration_score'] or 0
        }

    return progress


def update_sub_exercise_progress(user_id: int, course_id: int, topic_id: str,
                               sub_exercise_id: int, status: str,
                               time_spent: int = 0, score: float = 0,
                               concentration_score: float = 0, emotion_data: list = None) -> bool:
    """Update user's progress on a sub-exercise"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Check if progress record exists
        cursor.execute('''
            SELECT id FROM sub_exercise_progress
            WHERE user_id = ? AND course_id = ? AND topic_id = ? AND sub_exercise_id = ?
        ''', (user_id, course_id, topic_id, sub_exercise_id))

        existing = cursor.fetchone()
        current_time = datetime.utcnow().isoformat()

        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE sub_exercise_progress
                SET status = ?, time_spent = time_spent + ?, score = ?,
                    concentration_score = ?, emotion_data = ?,
                    completion_time = CASE WHEN ? = 'completed' THEN ? ELSE completion_time END,
                    attempts = attempts + 1, updated_at = ?
                WHERE user_id = ? AND course_id = ? AND topic_id = ? AND sub_exercise_id = ?
            ''', (
                status, time_spent, score, concentration_score,
                json.dumps(emotion_data or []), status, current_time, current_time,
                user_id, course_id, topic_id, sub_exercise_id
            ))
        else:
            # Create new record
            cursor.execute('''
                INSERT INTO sub_exercise_progress
                (user_id, course_id, topic_id, sub_exercise_id, status, start_time,
                 completion_time, time_spent, attempts, score, concentration_score, emotion_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ''', (
                user_id, course_id, topic_id, sub_exercise_id, status, current_time,
                current_time if status == 'completed' else None,
                time_spent, score, concentration_score, json.dumps(emotion_data or [])
            ))

        connection.commit()
        connection.close()
        return True

    except Exception as e:
        print(f"Error updating sub-exercise progress: {e}")
        return False

def get_user_by_email(email):
    """Get user by email address"""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    connection.close()
    return user


def save_course_progress(user_id: int, course_id: int, topic_id: str = None, 
                         status: str = 'in_progress', progress_percentage: float = 0) -> bool:
    """Save or update user's course progress"""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute('''
            INSERT INTO course_progress (user_id, course_id, topic_id, status, progress_percentage, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id, course_id, topic_id) DO UPDATE SET
                status = excluded.status,
                progress_percentage = excluded.progress_percentage,
                updated_at = datetime('now')
        ''', (user_id, course_id, topic_id, status, progress_percentage))
        
        connection.commit()
        connection.close()
        return True
    except Exception as e:
        print(f"Error saving course progress: {e}")
        return False


def get_user_course_progress(user_id: int, course_id: int = None) -> list:
    """Get user's progress for a course or all courses"""
    connection = get_connection()
    cursor = connection.cursor()
    
    if course_id:
        cursor.execute('''
            SELECT course_id, topic_id, status, progress_percentage, started_at, updated_at
            FROM course_progress 
            WHERE user_id = ? AND course_id = ?
            ORDER BY updated_at DESC
        ''', (user_id, course_id))
    else:
        cursor.execute('''
            SELECT course_id, topic_id, status, progress_percentage, started_at, updated_at
            FROM course_progress 
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
    
    results = cursor.fetchall()
    connection.close()
    return [dict(row) for row in results]


def get_user_enrolled_courses(user_id: int) -> list:
    """Get list of courses the user has started (enrolled in)"""
    connection = get_connection()
    cursor = connection.cursor()
    
    cursor.execute('''
        SELECT DISTINCT course_id, MIN(started_at) as enrolled_at,
               MAX(progress_percentage) as best_progress,
               MAX(updated_at) as last_activity
        FROM course_progress 
        WHERE user_id = ?
        GROUP BY course_id
        ORDER BY last_activity DESC
    ''', (user_id,))
    
    results = cursor.fetchall()
    connection.close()
    return [dict(row) for row in results]

