from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from ml_models import predict_emotion  # Use your real model function

# ...existing code...

import json
import os
import sqlite3
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# Import your existing modules
from db import init_db, create_user, get_user_by_credentials, get_user_by_email_credentials, get_user_by_email, list_courses, get_courses_by_language_and_difficulty, get_connection
from utils import save_progress, get_student_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize database
init_db()

# ...existing code...

# Add the emotion detection route after app initialization
@app.route('/api/emotion_detect', methods=['POST'])
def emotion_detect():
    data = request.get_json()
    img_data = data.get('image')
    if not img_data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'face_detected': False, 'is_concentrated': False, 'emotion': 'No Face'})
        # Use your real model for prediction
        emotion, concentration, face_found = predict_emotion(frame)
        return jsonify({
            'face_detected': bool(face_found),
            'is_concentrated': bool(concentration),
            'emotion': emotion
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
import cv2
import numpy as np
from flask import request, jsonify
import base64
import random

# Try to import enhanced detection libraries (optional)
try:
    import dlib
    dlib_available = True
except ImportError:
    dlib_available = False
    print("dlib not available, using basic detection")

try:
    import mediapipe as mp
    # MediaPipe Face Detection (more accurate)
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize MediaPipe
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    mediapipe_available = True
except ImportError:
    mediapipe_available = False
    print("MediaPipe not available, using OpenCV only")

# Enhanced face detection using multiple methods
# Primary: OpenCV Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Enhanced emotion mapping with concentration indicators
emotion_concentration_map = {
    'happy': True,
    'neutral': True,
    'focused': True,
    'concentrated': True,
    'alert': True,
    'surprised': False,
    'sad': False,
    'angry': False,
    'fearful': False,
    'disgusted': False,
    'distracted': False,
    'tired': False
}
# ...existing code...

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort, send_from_directory, flash
from datetime import datetime
import json
import os
import sqlite3
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# Import your existing modules
from db import init_db, create_user, get_user_by_credentials, get_user_by_email_credentials, get_user_by_email, list_courses, get_courses_by_language_and_difficulty, get_connection
from utils import save_progress, get_student_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize database
init_db()

# Endpoint for face and emotion recognition
@app.route('/api/emotion', methods=['POST'])
def api_emotion():
    try:
        print("=== Emotion API called ===")  # Debug log
        data = request.json
        img_data = data.get('image')
        if not img_data:
            print("ERROR: No image provided")
            return jsonify({'error': 'No image provided'}), 400
        
        print(f"Received image data: {len(img_data)} characters")
        
        # Decode base64 image
        img_bytes = base64.b64decode(img_data.split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("ERROR: Could not decode image")
            return jsonify({'error': 'Could not decode image'}), 400
        
        print(f"Decoded image shape: {frame.shape}")
        
        # Enhanced face detection with multiple methods
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = False
        face_count = 0
        concentration_score = 0
        emotion = 'unknown'
        face_coordinates = []  # Store face positions for frontend drawing
        
        # Method 1: OpenCV Haar Cascades - detect all faces
        faces_haar = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,   # Good sensitivity
            minNeighbors=5,    # Balance between accuracy and detection
            minSize=(60, 60),  # Reasonable minimum size
            maxSize=(250, 250), # Reasonable maximum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Store all face coordinates for frontend drawing
        for (x, y, w, h) in faces_haar:
            face_coordinates.append({
                'x': int(x),
                'y': int(y), 
                'width': int(w),
                'height': int(h),
                'confidence': 0.8  # Default confidence for Haar cascades
            })
        
        # Method 2: MediaPipe Face Detection (if available) - supplement Haar detection
        faces_mediapipe = []
        if mediapipe_available:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                # Process all detections with good confidence
                for detection in results.detections:
                    confidence = detection.score[0]
                    if confidence > 0.5:  # Only include confident detections
                        # Convert MediaPipe coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Check if this face is already detected by Haar (avoid duplicates)
                        is_duplicate = False
                        for haar_face in faces_haar:
                            hx, hy, hw, hh = haar_face
                            # Check overlap - if centers are close, consider it duplicate
                            haar_center_x = hx + hw // 2
                            haar_center_y = hy + hh // 2
                            mp_center_x = x + width // 2
                            mp_center_y = y + height // 2
                            
                            distance = ((haar_center_x - mp_center_x) ** 2 + (haar_center_y - mp_center_y) ** 2) ** 0.5
                            overlap_threshold = min(hw, hh, width, height) * 0.5
                            
                            if distance < overlap_threshold:
                                is_duplicate = True
                                break
                        
                        # Only add if not duplicate
                        if not is_duplicate:
                            face_coordinates.append({
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'confidence': round(confidence, 2)
                            })
                            faces_mediapipe.append(detection)
        
        # Combine detection results
        total_faces = len(faces_haar) + len(faces_mediapipe)
        
        if len(faces_haar) > 0 or len(faces_mediapipe) > 0:
            faces_detected = True
            face_count = len(face_coordinates)  # Count all detected faces
            
            # Analyze all detected faces for emotion and concentration
            concentration_scores = []
            emotions = []
            
            # Process Haar detected faces
            for i, (x, y, w, h) in enumerate(faces_haar):
                face_roi = gray[y:y+h, x:x+w]
                
                # Check for eyes (indicates alertness)
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                eye_detected = len(eyes) >= 2
                
                # Check for smile (indicates engagement)
                smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
                smile_detected = len(smiles) > 0
                
                # Calculate face quality metrics
                face_size = w * h
                face_position_score = 1.0 if (frame.shape[1] * 0.2 < x + w/2 < frame.shape[1] * 0.8) else 0.5
                face_size_score = min(1.0, face_size / (frame.shape[0] * frame.shape[1] * 0.1))
                
                # Enhanced emotion prediction based on facial features
                if eye_detected and smile_detected:
                    face_emotions = ['happy', 'focused', 'engaged', 'alert']
                elif eye_detected:
                    face_emotions = ['neutral', 'concentrated', 'focused', 'alert']
                elif smile_detected:
                    face_emotions = ['happy', 'relaxed']
                else:
                    face_emotions = ['neutral', 'tired', 'distracted']
                
                face_emotion = random.choice(face_emotions)
                emotions.append(face_emotion)
                
                # Calculate concentration score (0-1) for this face
                face_concentration = (
                    (0.4 if eye_detected else 0) +
                    (0.2 if smile_detected else 0) +
                    (0.2 * face_position_score) +
                    (0.2 * face_size_score)
                )
                concentration_scores.append(face_concentration)
            
            # Process MediaPipe detected faces
            for detection in faces_mediapipe:
                confidence = detection.score[0]
                
                if confidence > 0.7:
                    face_emotions = ['focused', 'concentrated', 'alert', 'neutral']
                    face_concentration = min(1.0, confidence + 0.2)
                else:
                    face_emotions = ['neutral', 'distracted']
                    face_concentration = confidence * 0.8
                
                face_emotion = random.choice(face_emotions)
                emotions.append(face_emotion)
                concentration_scores.append(face_concentration)
            
            # Calculate overall emotion and concentration
            if emotions:
                # Use the most positive emotion if multiple faces
                emotion_priority = {'happy': 5, 'engaged': 4, 'focused': 4, 'alert': 3, 'concentrated': 3, 'neutral': 2, 'relaxed': 2, 'tired': 1, 'distracted': 0}
                emotion = max(emotions, key=lambda e: emotion_priority.get(e, 0))
            else:
                emotion = 'neutral'
            
            # Average concentration score across all faces
            if concentration_scores:
                concentration_score = sum(concentration_scores) / len(concentration_scores)
            else:
                concentration_score = 0.5
            
            # Determine if user is concentrated based on emotion and score
            is_concentrated = emotion_concentration_map.get(emotion, True) and concentration_score > 0.4
            
            logger.info(f"Enhanced detection: {face_count} faces, emotion: {emotion}, "
                       f"concentration: {is_concentrated}, score: {concentration_score:.2f}")
            
            return jsonify({
                'emotion': emotion,
                'concentration': is_concentrated,
                'face_detected': True,
                'face_count': face_count,
                'concentration_score': round(concentration_score, 2),
                'detection_method': 'haar+mediapipe' if faces_mediapipe else 'haar',
                'confidence': round(concentration_score * 100, 1),
                'face_coordinates': face_coordinates  # Add face positions for drawing boxes
            })
        else:
            logger.info("No faces detected with enhanced methods")
            return jsonify({
                'emotion': 'unknown', 
                'concentration': False, 
                'face_detected': False, 
                'face_count': 0,
                'concentration_score': 0,
                'detection_method': 'none',
                'confidence': 0,
                'face_coordinates': []  # Empty array when no faces
            })
    
    except Exception as e:
        logger.error(f"Error in enhanced emotion detection: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# Course filtering functions
def get_courses_with_complete_quiz():
    """Filter courses to only include those with complete quiz JSON files (60 questions: 20B, 20M, 20A)"""
    quiz_dir = 'data/quiz_questions'
    courses_with_complete_quiz = set()
    
    if not os.path.exists(quiz_dir):
        logger.warning(f"Quiz directory {quiz_dir} not found!")
        return courses_with_complete_quiz
    
    try:
        for quiz_file in os.listdir(quiz_dir):
            if quiz_file.endswith('.json'):
                quiz_path = os.path.join(quiz_dir, quiz_file)
                try:
                    with open(quiz_path, 'r') as f:
                        quiz_data = json.load(f)
                        
                    course_id = quiz_data.get('course_id')
                    questions = quiz_data.get('questions', [])
                    
                    if course_id and questions:
                        # Count questions by level
                        basic_count = len([q for q in questions if q.get('level') == 'basic'])
                        medium_count = len([q for q in questions if q.get('level') == 'medium'])
                        advanced_count = len([q for q in questions if q.get('level') == 'advanced'])
                        
                        # Check if quiz has sufficient questions and proper level distribution
                        # Accept courses with at least 20 questions and all three levels represented
                        total_questions = len(questions)
                        has_all_levels = basic_count > 0 and medium_count > 0 and advanced_count > 0
                        
                        if (total_questions >= 20 and has_all_levels):
                            courses_with_complete_quiz.add(course_id)
                            logger.info(f"Course {course_id} has complete quiz: {quiz_file} ({total_questions} questions: {basic_count} basic, {medium_count} medium, {advanced_count} advanced)")
                        else:
                            logger.warning(f"Course {course_id} quiz incomplete: {quiz_file} ({total_questions} questions: {basic_count} basic, {medium_count} medium, {advanced_count} advanced)")
                
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading quiz file {quiz_file}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error scanning quiz directory: {e}")
    
    logger.info(f"Found {len(courses_with_complete_quiz)} courses with complete quizzes: {list(courses_with_complete_quiz)}")
    return courses_with_complete_quiz

def filter_courses_with_quiz(courses):
    """Filter a list of courses to only include those with complete quiz files"""
    complete_quiz_course_ids = get_courses_with_complete_quiz()
    filtered_courses = []
    
    for course in courses:
        course_id = course.get('id') if isinstance(course, dict) else course['id']
        if course_id in complete_quiz_course_ids:
            filtered_courses.append(course)
        else:
            logger.debug(f"Filtering out course {course_id} - no complete quiz file")
    
    logger.info(f"Filtered courses: {len(filtered_courses)} out of {len(courses)} total courses")
    return filtered_courses

@app.route('/')
def index():
    if 'user_id' not in session:
        # Show landing page for non-authenticated users
        return render_template('index.html')
    
    # Redirect authenticated users to dashboard
    return redirect(url_for('dashboard'))


@app.route('/camera-test')
def camera_test():
    with open('camera_test.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user_id = session['user_id']
    username = session.get('username', 'User')
    
    # Load courses from quiz JSON files (13 courses)
    import glob, json
    courses = []
    
    for file in glob.glob('data/quiz_questions/*.json'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                course = {
                    'id': data.get('course_id', len(courses) + 1),
                    'title': data.get('course_title', 'Unknown Course'),
                    'description': f"Learn {data.get('course_title')} concepts and best practices.",
                    'total_questions': len(data.get('questions', [])),
                    'total_topics': 5,  # Default number of topics
                    'total_exercises': 20  # Default number of exercises
                }
                courses.append(course)
        except Exception as e:
            print(f"Error loading course from {file}: {e}")
    
    print(f"DEBUG: Loaded {len(courses)} courses from JSON files")
    
    # Get user's ongoing courses (courses with progress)
    ongoing_courses = []
    available_courses = []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    for course in courses:
        # Check if user has progress in this course
        try:
            cursor.execute('''
                SELECT COUNT(*) as progress_count, AVG(progress_percentage) as avg_progress
                FROM course_progress 
                WHERE user_id = ? AND course_id = ?
            ''', (user_id, course['id']))
            
            progress_result = cursor.fetchone()
            has_progress = progress_result and progress_result['progress_count'] > 0
        except:
            # Table doesn't exist or other error, assume no progress
            has_progress = False
            progress_result = None
        
        if has_progress:
            # User has started this course
            course['progress'] = {
                'percentage': progress_result['avg_progress'] or 0,
                'completed_topics': progress_result['progress_count'],
                'total_topics': course['total_topics'],
                'status': 'in_progress' if progress_result['avg_progress'] < 100 else 'completed'
            }
            ongoing_courses.append(course)
        else:
            # Course available to start
            available_courses.append(course)

    conn.close()
    
    print(f"DEBUG: Found {len(available_courses)} available courses and {len(ongoing_courses)} ongoing courses")
    
    return render_template('dashboard.html', 
                         courses=courses, 
                         available_courses=available_courses,
                         ongoing_courses=ongoing_courses,
                         user_id=user_id,
                         username=username)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('profile.html')

@app.route('/career-recommendations')
def career_recommendations():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('career_recommendations.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = get_user_by_email_credentials(email, password)
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email'] = user[3] if len(user) > 3 else email
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/course/<int:course_id>/modules')
def course_modules(course_id):
    user_id = session.get('user_id')
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get course details
        cursor.execute('SELECT * FROM courses WHERE id = ?', (course_id,))
        course = cursor.fetchone()
        
        if not course:
            course = {'id': course_id, 'title': f'Course {course_id}', 'description': 'Sample course'}
        else:
            course = dict(course)
        
        # Get all modules for the course
        try:
            cursor.execute('''
                SELECT id, title, description, module_order 
                FROM modules 
                WHERE course_id = ?
                ORDER BY module_order
            ''', (course_id,))
            modules = cursor.fetchall()
        except Exception as module_error:
            print(f"Error fetching modules: {module_error}")
            modules = []
        
        # Get submodules for each module
        all_modules = []
        previous_module_completed = True  # First module is always unlocked
        
        for index, module in enumerate(modules):
            module_dict = dict(module)
            cursor.execute('''
                SELECT id, title, description, content, submodule_order
                FROM submodules
                WHERE module_id = ?
                ORDER BY submodule_order
            ''', (module_dict['id'],))
            submodules_raw = [dict(sm) for sm in cursor.fetchall()]
            
            # Check each submodule's completion status and add locking
            submodules_with_status = []
            previous_submodule_completed = True  # First submodule is always unlocked
            
            for sm_index, sm in enumerate(submodules_raw):
                # Check if this submodule is completed
                sm_completed = False
                if user_id:
                    try:
                        cursor.execute('''
                            SELECT status FROM submodule_progress 
                            WHERE user_id = ? AND submodule_id = ?
                        ''', (user_id, sm['id']))
                        sm_progress = cursor.fetchone()
                        if sm_progress and sm_progress['status'] == 'completed':
                            sm_completed = True
                            sm['status'] = 'completed'
                        else:
                            sm['status'] = 'not_started'
                    except:
                        sm['status'] = 'not_started'
                else:
                    sm['status'] = 'not_started'
                
                # Sequential locking for submodules:
                # First submodule is always unlocked, rest are locked until previous is completed
                if sm_index == 0:
                    sm['locked'] = False
                else:
                    sm['locked'] = not previous_submodule_completed
                
                previous_submodule_completed = sm_completed
                submodules_with_status.append(sm)
            
            module_dict['submodules'] = submodules_with_status
            
            # Calculate module status based on submodule completions
            total_submodules = len(submodules_with_status)
            completed_submodules = sum(1 for sm in submodules_with_status if sm.get('status') == 'completed')
            
            if total_submodules > 0:
                if completed_submodules == total_submodules:
                    module_status = 'completed'
                elif completed_submodules > 0:
                    module_status = 'in_progress'
                else:
                    module_status = 'not_started'
            else:
                module_status = 'not_started'
            
            module_dict['status'] = module_status
            module_dict['score'] = int((completed_submodules / total_submodules * 100) if total_submodules > 0 else 0)
            module_dict['completed_submodules'] = completed_submodules
            module_dict['total_submodules'] = total_submodules
            
            # Sequential locking: Module is locked unless:
            # 1. It's the first module (index == 0), OR
            # 2. The previous module was completed
            if index == 0:
                module_dict['locked'] = False
            else:
                module_dict['locked'] = not previous_module_completed
            
            # Update for next iteration: check if this module is completed
            previous_module_completed = (module_status == 'completed')
            
            all_modules.append(module_dict)
        
        conn.close()
        
        return render_template('course_modules.html', 
                             course=course,
                             sub_exercises=all_modules,
                             course_id=course_id)
    except Exception as e:
        print(f"Error in course_modules: {e}")
        return render_template('course_modules.html',
                             course={'id': course_id, 'title': f'Course {course_id}', 'description': 'Sample course'},
                             sub_exercises=[],
                             course_id=course_id)

@app.route('/register', methods=['GET', 'POST'])
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!')
            return render_template('signup.html')
            
        if password != confirm_password:
            flash('Passwords do not match!')
            return render_template('signup.html')
            
        if len(password) < 6:
            flash('Password must be at least 6 characters long!')
            return render_template('signup.html')
            
        # Check if user already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            flash('Email already registered! Please use a different email.')
            return render_template('signup.html')
            
        # Create new user
        try:
            user_id = create_user(username, password, is_guest=False, email=email)
            if user_id:
                flash('Account created successfully! Please login.')
                return redirect(url_for('login'))
            else:
                flash('Error creating account. Please try again.')
                return render_template('signup.html')
        except Exception as e:
            flash('Error creating account. Please try again.')
            return render_template('signup.html')
    
    return render_template('signup.html')

@app.route('/course/<int:course_id>')
def course_detail(course_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get course details
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM courses WHERE id = ?', (course_id,))
    course = cursor.fetchone()
    
    if not course:
        abort(404)
    
    # Get topics for this course
    cursor.execute('SELECT * FROM topics WHERE course_id = ? ORDER BY topic_order', (course_id,))
    topics = cursor.fetchall()
    conn.close()
    
    return render_template('course_detail.html', course=course, topics=topics)

@app.route('/start_course/<int:course_id>')
def start_course_redirect(course_id):
    user_id = session.get('user_id')
    
    # Save progress when user starts a course
    if user_id:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Check if user already has progress for this course
            cursor.execute('''
                SELECT id FROM course_progress 
                WHERE user_id = ? AND course_id = ?
            ''', (user_id, course_id))
            
            existing = cursor.fetchone()
            
            if not existing:
                # Insert new progress record - course started
                cursor.execute('''
                    INSERT INTO course_progress (user_id, course_id, topic_id, status, progress_percentage, started_at, updated_at)
                    VALUES (?, ?, ?, 'in_progress', 0, datetime('now'), datetime('now'))
                ''', (user_id, course_id, 'start'))
                conn.commit()
                print(f"DEBUG: Started course {course_id} for user {user_id}")
            else:
                # Update last accessed time
                cursor.execute('''
                    UPDATE course_progress 
                    SET updated_at = datetime('now')
                    WHERE user_id = ? AND course_id = ?
                ''', (user_id, course_id))
                conn.commit()
                print(f"DEBUG: Updated course {course_id} access for user {user_id}")
            
            conn.close()
        except Exception as e:
            print(f"Error saving course start progress: {e}")
    
    return redirect(url_for('course_modules', course_id=course_id))

def get_next_topic_id(course_id, current_topic_id):
    """Get the next topic ID for navigation"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get current topic's index
        cursor.execute('''
            SELECT sub_exercise_index FROM sub_exercises 
            WHERE course_id = ? AND topic_id = ? 
            ORDER BY sub_exercise_index LIMIT 1
        ''', (course_id, current_topic_id))
        
        current_result = cursor.fetchone()
        if not current_result:
            return None
            
        current_index = current_result[0]
        
        # Get next topic
        cursor.execute('''
            SELECT DISTINCT topic_id FROM sub_exercises 
            WHERE course_id = ? AND sub_exercise_index > ? 
            ORDER BY sub_exercise_index LIMIT 1
        ''', (course_id, current_index))
        
        next_result = cursor.fetchone()
        conn.close()
        
        return next_result[0] if next_result else None
        
    except Exception as e:
        logger.error(f"Error getting next topic: {e}")
        return None

@app.route('/course/<int:course_id>/submodule/<int:submodule_id>')
def view_submodule(course_id, submodule_id):
    """View individual submodule content"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get course details
        cursor.execute('SELECT * FROM courses WHERE id = ?', (course_id,))
        course = cursor.fetchone()
        
        if not course:
            flash('Course not found', 'error')
            return redirect(url_for('dashboard'))
        
        course = dict(course)
        
        # Get submodule details
        cursor.execute('SELECT * FROM submodules WHERE id = ?', (submodule_id,))
        submodule = cursor.fetchone()
        
        if not submodule:
            flash('Submodule not found', 'error')
            return redirect(url_for('course_modules', course_id=course_id))
        
        submodule = dict(submodule)
        
        # Get module details
        cursor.execute('SELECT * FROM modules WHERE id = ?', (submodule['module_id'],))
        module = cursor.fetchone()
        
        if module:
            module = dict(module)
        else:
            module = {'title': 'Unknown Module'}
        
        # Get navigation (previous and next submodules)
        cursor.execute('''
            SELECT id, title FROM submodules 
            WHERE module_id = ? AND submodule_order < ?
            ORDER BY submodule_order DESC LIMIT 1
        ''', (submodule['module_id'], submodule['submodule_order']))
        prev_submodule = cursor.fetchone()
        
        cursor.execute('''
            SELECT id, title FROM submodules 
            WHERE module_id = ? AND submodule_order > ?
            ORDER BY submodule_order ASC LIMIT 1
        ''', (submodule['module_id'], submodule['submodule_order']))
        next_submodule = cursor.fetchone()
        
        conn.close()
        
        return render_template('submodule_content.html',
                             course=course,
                             module=module,
                             submodule=submodule,
                             prev_submodule=dict(prev_submodule) if prev_submodule else None,
                             next_submodule=dict(next_submodule) if next_submodule else None)
        
    except Exception as e:
        print(f"Error in view_submodule: {str(e)}")
        flash('Error loading submodule', 'error')
        return redirect(url_for('course_modules', course_id=course_id))

@app.route('/course/<int:course_id>/topic/<topic_id>')
def view_topic(course_id, topic_id):
    try:
        return redirect(url_for('course_modules', course_id=course_id))
    except Exception as e:
        print(f"Error redirecting: {e}")
        return redirect(url_for('index'))
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get course details
        cursor.execute('SELECT * FROM courses WHERE id = ?', (course_id,))
        course = cursor.fetchone()
        
        if not course:
            # Create mock course if not found
            course = {'id': course_id, 'title': f'Course {course_id}', 'description': 'Sample course'}
        else:
            course = dict(course)
        
        # Get topic details (if topics table exists)
        try:
            cursor.execute('SELECT * FROM topics WHERE course_id = ? AND id = ?', (course_id, topic_id))
            topic = cursor.fetchone()
            if topic:
                topic = dict(topic)
            else:
                topic = {'id': topic_id, 'title': f'Topic {topic_id}', 'content': f'<p>Welcome to topic {topic_id}!</p>'}
        except:
            topic = {'id': topic_id, 'title': f'Topic {topic_id}', 'content': f'<p>Welcome to topic {topic_id}!</p>'}
        
        # Get modules (sub-exercises) directly for the course
        try:
            cursor.execute('''
                SELECT * FROM sub_exercises 
                WHERE course_id = ? 
                AND title LIKE 'Module%'
                ORDER BY sub_exercise_index
            ''', (course_id,))
            sub_exercises = cursor.fetchall()
            
            # Convert to dictionaries and get progress
            exercises_with_progress = []
            for exercise in sub_exercises:
                exercise_dict = dict(exercise)
                
                # Get progress for this exercise if user is logged in
                if user_id:
                    cursor.execute('''
                        SELECT status, score, completion_time 
                        FROM sub_exercise_progress 
                        WHERE user_id = ? AND sub_exercise_id = ?
                    ''', (user_id, exercise['id']))
                    progress = cursor.fetchone()
                    
                    if progress:
                        exercise_dict['status'] = progress['status']
                        exercise_dict['score'] = progress['score']
                        exercise_dict['completed'] = progress['status'] == 'completed'
                    else:
                        exercise_dict['status'] = 'not_started'
                        exercise_dict['score'] = 0
                        exercise_dict['completed'] = False
                else:
                    exercise_dict['status'] = 'not_started'
                    exercise_dict['score'] = 0
                    exercise_dict['completed'] = False
                
                exercises_with_progress.append(exercise_dict)
            
            sub_exercises = exercises_with_progress
            
        except Exception as e:
            print(f"Error fetching sub-exercises: {e}")
            # Create mock exercises if table doesn't exist
            sub_exercises = create_mock_exercises(course_id, topic_id)
        
        conn.close()
        
        # Get user progress for continue course functionality
        user_progress = None
        if user_id:
            user_progress = get_user_progress(user_id, course_id, topic_id)
        
        print("Rendering template with real data")
        return render_template('course_module.html', 
                             course=course, 
                             topic=topic, 
                             sub_exercises=sub_exercises,
                             course_id=course_id,
                             topic_id=topic_id,
                             user_progress=user_progress,
                             next_topic_id=get_next_topic_id(course_id, topic_id))
        
    except Exception as e:
        print(f"Error in view_topic: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to mock data
        mock_course = {'id': course_id, 'title': f'Course {course_id}'}
        mock_topic = {'title': f'Topic {topic_id}', 'content': f'<p>Welcome to topic {topic_id}!</p>'}
        mock_exercises = create_mock_exercises(course_id, topic_id)
        
        return render_template('course_module.html', 
                             course=mock_course, 
                             topic=mock_topic, 
                             sub_exercises=mock_exercises,
                             course_id=course_id,
                             topic_id=topic_id,
                             next_topic_id=None)

def create_mock_exercises(course_id, topic_id):
    """Create mock exercises with proper structure"""
    return [
        {
            'id': 1,
            'title': 'Module 1: Introduction to Python',
            'content': '<h3>Welcome to Python Programming!</h3><p>Learn the fundamentals of Python programming language.</p>',
            'exercise_type': 'module',
            'difficulty': 'beginner',
            'estimated_time': 30,
            'completed': False,
            'status': 'not_started',
            'score': 0,
            'instructions': 'Start with Python basics'
        },
        {
            'id': 2,
            'title': 'Code Example',
            'content': f'<h3>Practical Example</h3><p>Here\'s how to apply what you learned:</p><pre><code>print("Hello from topic {topic_id}!")\n# Add your code here</code></pre>',
            'exercise_type': 'example',
            'difficulty': 'beginner',
            'estimated_time': 10,
            'completed': False,
            'status': 'not_started',
            'score': 0,
            'instructions': 'Study the example and understand how it works.'
        },
        {
            'id': 3,
            'title': 'Practice Exercise',
            'content': f'<h3>Your Turn!</h3><p>Now try to implement this yourself:</p><div class="exercise-task"><h4>Task:</h4><p>Create a simple program that demonstrates the concepts from topic {topic_id}.</p></div>',
            'exercise_type': 'practice',
            'difficulty': 'intermediate',
            'estimated_time': 20,
            'completed': False,
            'status': 'not_started',
            'score': 0,
            'instructions': 'Write your own code to solve this problem.'
        },
        {
            'id': 4,
            'title': 'Knowledge Quiz',
            'content': f'<h3>Test Your Understanding</h3><p>Complete this quiz to check your knowledge of topic {topic_id}.</p>',
            'exercise_type': 'quiz',
            'difficulty': 'intermediate',
            'estimated_time': 15,
            'completed': False,
            'status': 'not_started',
            'score': 0,
            'instructions': 'Answer all questions to test your understanding.'
        },
        {
            'id': 5,
            'title': 'Mini Project',
            'content': f'<h3>Apply Your Skills</h3><p>Build a small project using everything you\'ve learned in topic {topic_id}.</p>',
            'exercise_type': 'project',
            'difficulty': 'advanced',
            'estimated_time': 30,
            'completed': False,
            'status': 'not_started',
            'score': 0,
            'instructions': 'Create a complete project that demonstrates mastery.'
        }
    ]

def get_user_progress(user_id, course_id, topic_id):
    """Get user's progress for continue course functionality"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get overall course progress
        cursor.execute('''
            SELECT * FROM course_progress 
            WHERE user_id = ? AND course_id = ? AND topic_id = ?
        ''', (user_id, course_id, topic_id))
        progress = cursor.fetchone()
        
        # Get completed exercises count
        cursor.execute('''
            SELECT COUNT(*) as completed_count
            FROM sub_exercise_progress 
            WHERE user_id = ? AND course_id = ? AND topic_id = ? AND status = 'completed'
        ''', (user_id, course_id, topic_id))
        completed_exercises = cursor.fetchone()
        
        # Get total exercises count
        cursor.execute('''
            SELECT COUNT(*) as total_count
            FROM sub_exercises 
            WHERE course_id = ? AND topic_id = ?
        ''', (course_id, topic_id))
        total_exercises = cursor.fetchone()
        
        conn.close()
        
        if progress:
            progress_dict = dict(progress)
            progress_dict['completed_exercises'] = completed_exercises['completed_count'] if completed_exercises else 0
            progress_dict['total_exercises'] = total_exercises['total_count'] if total_exercises else 5
            progress_dict['progress_percentage'] = (progress_dict['completed_exercises'] / progress_dict['total_exercises']) * 100 if progress_dict['total_exercises'] > 0 else 0
            return progress_dict
        
        return {
            'status': 'not_started',
            'progress_percentage': 0,
            'completed_exercises': 0,
            'total_exercises': total_exercises['total_count'] if total_exercises else 5
        }
        
    except Exception as e:
        print(f"Error getting user progress: {e}")
        return {
            'status': 'not_started',
            'progress_percentage': 0,
            'completed_exercises': 0,
            'total_exercises': 5
        }

# Continue Course Route
@app.route('/continue-course/<int:course_id>')
def continue_course(course_id):
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find the user's last accessed topic or next incomplete topic
        cursor.execute('''
            SELECT topic_id, progress_percentage, status 
            FROM course_progress 
            WHERE user_id = ? AND course_id = ?
            ORDER BY updated_at DESC 
            LIMIT 1
        ''', (user_id, course_id))
        
        last_progress = cursor.fetchone()
        
        if last_progress and last_progress['status'] != 'completed':
            # Continue from last topic
            topic_id = last_progress['topic_id']
        else:
            # Find first incomplete topic or start from beginning
            cursor.execute('''
                SELECT DISTINCT topic_id 
                FROM sub_exercises 
                WHERE course_id = ? 
                ORDER BY topic_id 
                LIMIT 1
            ''', (course_id,))
            
            first_topic = cursor.fetchone()
            if first_topic:
                topic_id = first_topic['topic_id']
            else:
                topic_id = f"{course_id}_0_0"  # Default topic format
        
        conn.close()
        return redirect(url_for('view_topic', course_id=course_id, topic_id=topic_id))
        
    except Exception as e:
        logger.error(f"Error in continue_course: {e}")
        # Fallback to first topic
        return redirect(url_for('view_topic', course_id=course_id, topic_id=f"{course_id}_0_0"))

# Module Status API - Update module progress
@app.route('/api/module_status', methods=['POST'])
def update_module_status():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    try:
        data = request.get_json()
        module_id = data.get('module_id')
        status = data.get('status', 'started')  # 'started', 'in_progress', 'completed'
        course_id = data.get('course_id')
        
        if not module_id:
            return jsonify({'success': False, 'error': 'Module ID required'}), 400
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Update module_progress table
        cursor.execute('''
            INSERT INTO module_progress (user_id, module_id, status, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(user_id, module_id) DO UPDATE SET
                status = excluded.status,
                updated_at = datetime('now')
        ''', (user_id, module_id, status))
        
        # If course_id is provided, also update course progress
        if course_id:
            # Count completed modules for this course
            cursor.execute('''
                SELECT COUNT(*) as completed
                FROM module_progress mp
                JOIN modules m ON mp.module_id = m.id
                WHERE mp.user_id = ? AND m.course_id = ? AND mp.status = 'completed'
            ''', (user_id, course_id))
            completed = cursor.fetchone()['completed']
            
            # Count total modules for this course
            cursor.execute('''
                SELECT COUNT(*) as total FROM modules WHERE course_id = ?
            ''', (course_id,))
            total = cursor.fetchone()['total']
            
            # Calculate progress percentage
            progress = (completed / total * 100) if total > 0 else 0
            
            # Update course_progress
            cursor.execute('''
                INSERT INTO course_progress (user_id, course_id, topic_id, status, progress_percentage, updated_at)
                VALUES (?, ?, 'overall', ?, ?, datetime('now'))
                ON CONFLICT(user_id, course_id, topic_id) DO UPDATE SET
                    status = excluded.status,
                    progress_percentage = excluded.progress_percentage,
                    updated_at = datetime('now')
            ''', (user_id, course_id, 
                  'completed' if progress >= 100 else 'in_progress',
                  progress))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'module_id': module_id,
            'status': status,
            'message': f'Module status updated to {status}'
        })
        
    except Exception as e:
        logger.error(f"Error updating module status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Complete Module API - Mark module as completed and update course progress
@app.route('/api/module/complete', methods=['POST'])
def complete_module():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    try:
        data = request.get_json()
        course_id = data.get('course_id')
        module_id = data.get('module_id')
        module_index = data.get('module_index', 0)
        
        if not course_id or not module_id:
            return jsonify({'success': False, 'error': 'Course ID and Module ID required'}), 400
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Mark module as completed
        cursor.execute('''
            INSERT INTO module_progress (user_id, module_id, status, score, updated_at)
            VALUES (?, ?, 'completed', 100, datetime('now'))
            ON CONFLICT(user_id, module_id) DO UPDATE SET
                status = 'completed',
                score = 100,
                updated_at = datetime('now')
        ''', (user_id, module_id))
        
        # Count total modules started/completed by this user for this course
        # Using a simpler approach: count based on module_index
        cursor.execute('''
            SELECT COUNT(DISTINCT module_id) as completed_modules
            FROM module_progress
            WHERE user_id = ? AND status = 'completed'
        ''', (user_id,))
        completed_count = cursor.fetchone()['completed_modules']
        
        # Assume 5 modules per course (as shown in dashboard)
        total_modules = 5
        progress_percentage = min(100, (completed_count / total_modules) * 100)
        
        # Update course progress with more specific tracking
        cursor.execute('''
            UPDATE course_progress 
            SET progress_percentage = ?,
                status = CASE WHEN ? >= 100 THEN 'completed' ELSE 'in_progress' END,
                updated_at = datetime('now')
            WHERE user_id = ? AND course_id = ?
        ''', (progress_percentage, progress_percentage, user_id, course_id))
        
        # If no rows updated (first module completion), insert
        if cursor.rowcount == 0:
            cursor.execute('''
                INSERT INTO course_progress (user_id, course_id, topic_id, status, progress_percentage, updated_at)
                VALUES (?, ?, 'module', 'in_progress', ?, datetime('now'))
            ''', (user_id, course_id, progress_percentage))
        
        conn.commit()
        conn.close()
        
        print(f"DEBUG: Module {module_id} completed. Progress: {progress_percentage}% ({completed_count}/{total_modules})")
        
        return jsonify({
            'success': True,
            'module_id': module_id,
            'status': 'completed',
            'progress_percentage': round(progress_percentage, 1),
            'completed_modules': completed_count,
            'total_modules': total_modules
        })
        
    except Exception as e:
        logger.error(f"Error completing module: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Submodule Complete API - Mark submodule as completed
@app.route('/api/submodule/complete', methods=['POST'])
def complete_submodule():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    try:
        data = request.get_json()
        course_id = data.get('course_id')
        module_id = data.get('module_id')
        submodule_id = data.get('submodule_id')
        
        if not submodule_id:
            return jsonify({'success': False, 'error': 'Submodule ID required'}), 400
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Ensure the submodule_progress table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submodule_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                course_id INTEGER,
                module_id INTEGER,
                submodule_id INTEGER NOT NULL,
                status TEXT DEFAULT 'completed',
                completed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, submodule_id)
            )
        ''')
        
        # Mark submodule as completed
        cursor.execute('''
            INSERT INTO submodule_progress (user_id, course_id, module_id, submodule_id, status, completed_at)
            VALUES (?, ?, ?, ?, 'completed', datetime('now'))
            ON CONFLICT(user_id, submodule_id) DO UPDATE SET
                status = 'completed',
                completed_at = datetime('now')
        ''', (user_id, course_id or 0, module_id or 0, submodule_id))
        
        # Count completed submodules for this module
        cursor.execute('''
            SELECT COUNT(*) as completed
            FROM submodule_progress
            WHERE user_id = ? AND module_id = ? AND status = 'completed'
        ''', (user_id, module_id or 0))
        completed_submodules = cursor.fetchone()['completed']
        
        # Get total submodules for this module
        total_submodules = 3  # Default 3 submodules per module
        try:
            cursor.execute('''
                SELECT COUNT(*) as total FROM submodules WHERE module_id = ?
            ''', (module_id,))
            result = cursor.fetchone()
            if result and result['total'] > 0:
                total_submodules = result['total']
        except:
            pass
        
        # Calculate module progress percentage
        module_progress = min(100, (completed_submodules / total_submodules) * 100)
        
        # Update overall course progress
        if course_id:
            # Calculate overall progress: each completed submodule adds to total
            progress_increment = 20.0 / total_submodules  # 20% per module, divided by submodules
            cursor.execute('''
                UPDATE course_progress 
                SET progress_percentage = MIN(100, progress_percentage + ?),
                    updated_at = datetime('now')
                WHERE user_id = ? AND course_id = ?
            ''', (progress_increment, user_id, course_id))
        
        conn.commit()
        conn.close()
        
        print(f"DEBUG: Submodule {submodule_id} completed! Module progress: {module_progress}%")
        
        return jsonify({
            'success': True,
            'submodule_id': submodule_id,
            'status': 'completed',
            'module_progress': round(module_progress, 1),
            'completed_submodules': completed_submodules,
            'total_submodules': total_submodules
        })
        
    except Exception as e:
        logger.error(f"Error completing submodule: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Exercise Progress Route
@app.route('/api/exercise/complete', methods=['POST'])
def complete_exercise():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    try:
        data = request.get_json()
        course_id = data.get('course_id')
        topic_id = data.get('topic_id')
        exercise_id = data.get('exercise_id')
        score = data.get('score', 100)
        time_spent = data.get('time_spent', 0)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Update or insert exercise progress
        cursor.execute('''
            INSERT OR REPLACE INTO sub_exercise_progress 
            (user_id, course_id, topic_id, sub_exercise_id, status, score, 
             completion_time, time_spent, updated_at)
            VALUES (?, ?, ?, ?, 'completed', ?, datetime('now'), ?, datetime('now'))
        ''', (user_id, course_id, topic_id, exercise_id, score, time_spent))
        
        # Update overall course progress
        cursor.execute('''
            SELECT COUNT(*) as completed
            FROM sub_exercise_progress 
            WHERE user_id = ? AND course_id = ? AND topic_id = ? AND status = 'completed'
        ''', (user_id, course_id, topic_id))
        
        completed_count = cursor.fetchone()['completed']
        
        cursor.execute('''
            SELECT COUNT(*) as total
            FROM sub_exercises 
            WHERE course_id = ? AND topic_id = ?
        ''', (course_id, topic_id))
        
        total_count = cursor.fetchone()['total']
        progress_percentage = (completed_count / total_count * 100) if total_count > 0 else 100
        
        # Update course progress
        cursor.execute('''
            INSERT OR REPLACE INTO course_progress 
            (user_id, course_id, topic_id, status, progress_percentage, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (user_id, course_id, topic_id, 
              'completed' if progress_percentage >= 100 else 'in_progress', 
              progress_percentage))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'progress_percentage': progress_percentage,
            'completed_exercises': completed_count,
            'total_exercises': total_count
        })
        
    except Exception as e:
        logger.error(f"Error completing exercise: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Get Course Structure Route (Modules → Exercises → Tasks)
@app.route('/api/course/<int:course_id>/structure')
def get_course_structure(course_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get course details
        cursor.execute('SELECT * FROM courses WHERE id = ?', (course_id,))
        course = cursor.fetchone()
        
        if not course:
            return jsonify({'success': False, 'error': 'Course not found'}), 404
        
        # Get all topics (modules) for this course
        cursor.execute('''
            SELECT DISTINCT topic_id, 
                   COUNT(*) as exercise_count,
                   MIN(sub_exercise_index) as first_exercise
            FROM sub_exercises 
            WHERE course_id = ? 
            GROUP BY topic_id 
            ORDER BY topic_id
        ''', (course_id,))
        
        topics = cursor.fetchall()
        
        course_structure = {
            'course': dict(course),
            'modules': []
        };
        
        for topic in topics:
            # Get exercises for this topic
            cursor.execute('''
                SELECT id, title, description, exercise_type, difficulty, 
                       estimated_time, sub_exercise_index
                FROM sub_exercises 
                WHERE course_id = ? AND topic_id = ?
                ORDER BY sub_exercise_index
            ''', (course_id, topic['topic_id']))
            
            exercises = cursor.fetchall()
            
            module_data = {
                'topic_id': topic['topic_id'],
                'title': f"Module {topic['topic_id']}",
                'exercise_count': topic['exercise_count'],
                'exercises': [dict(exercise) for exercise in exercises]
            }
            
            course_structure['modules'].append(module_data)
        
        conn.close()
        return jsonify({'success': True, 'structure': course_structure})
        
    except Exception as e:
        logger.error(f"Error getting course structure: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Test route
@app.route('/test-simple')
def test_simple():
    return "<h1>Test Route Works!</h1>"

@app.route('/test-template')
def test_template():
    return render_template('topic_with_tracking_clean.html', 
                         course={'id': 5, 'title': 'Test Course'}, 
                         topic={'title': 'Test Topic', 'content': '<p>Test content</p>'}, 
                         sub_exercises=[
                             {'id': 1, 'title': 'Test Exercise', 'content': '<p>Test</p>', 'type': 'text', 'completed': False}
                         ])

@app.route('/results', methods=['GET', 'POST'])
def results():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Handle POST request from quiz page with detailed results
    if request.method == 'POST':
        try:
            quiz_results_json = request.form.get('quiz_results')
            if quiz_results_json:
                quiz_data = json.loads(quiz_results_json)
                return render_template('results.html', 
                                     quiz_data=quiz_data,
                                     detailed_view=True)
        except Exception as e:
            logger.error(f"Error processing quiz results: {e}")
    
    # Handle GET request - show historical results
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get user's quiz results
        cursor.execute('''
            SELECT quiz_title, score, total, language, difficulty, submitted_at
            FROM quiz_results 
            WHERE user_id = ? 
            ORDER BY submitted_at DESC 
            LIMIT 20
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return render_template('results.html', results=[dict(r) for r in results])
        
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return render_template('results.html', results=[])

@app.route('/quiz_page', methods=['GET', 'POST'])
def quiz_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    import glob, json, os
    courses = []
    
    # List all quiz JSON files
    quiz_dir = os.path.join('data', 'quiz_questions', '*.json')
    for file in glob.glob(quiz_dir):
        try:
            print(f"Loading quiz file: {file}")
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: Empty file {file}")
                    continue
                    
                data = json.loads(content)
                filename = os.path.basename(file).replace('.json', '')
                
                questions = data.get('questions', [])
                levels = sorted(set(q.get('level', 'unknown') for q in questions if 'level' in q))
                
                courses.append({
                    'title': data.get('course_title', filename),
                    'filename': filename,
                    'levels': levels,
                    'question_count': len(questions)
                })
                print(f"Successfully loaded: {data.get('course_title', filename)} with {len(questions)} questions")
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {file}: {e}")
            continue
        except Exception as e:
            print(f"Error loading course from {file}: {e}")
            continue
    
    selected_course = request.args.get('course')
    selected_level = request.args.get('level')
    questions = []
    
    if selected_course and selected_level:
        file_path = os.path.join('data', 'quiz_questions', f'{selected_course}.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_questions = data.get('questions', [])
                level_questions = [q for q in all_questions if q.get('level') == selected_level]
                questions = level_questions[:20]  # Limit to 20 questions
                print(f"Loaded {len(questions)} questions for {selected_course} - {selected_level}")
        except Exception as e:
            print(f"Error loading questions from {file_path}: {e}")
            questions = []
    
    print(f"Total courses loaded: {len(courses)}")
    return render_template('quiz_page.html', 
                         courses=courses, 
                         selected_course=selected_course, 
                         selected_level=selected_level, 
                         questions=questions)

@app.route('/enhanced_quiz')
def enhanced_quiz():
    return render_template('enhanced_quiz.html')

@app.route('/emotion_tracker')
def emotion_tracker():
    return render_template('emotion_tracker.html')

@app.route('/quiz/questions/<course_id>/<level>')
def get_quiz_questions(course_id, level):
    import os, json
    try:
        # Find the JSON file for this course
        filename = None
        quiz_dir = 'data/quiz_questions'
        
        for file in os.listdir(quiz_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(quiz_dir, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if str(data.get('course_id')) == str(course_id):
                            filename = file
                            break
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}")
                    continue
        
        if not filename:
            return jsonify({'success': False, 'error': 'Course not found', 'questions': []}), 404
        
        # Load and filter questions
        with open(os.path.join(quiz_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = [q for q in data.get('questions', []) if q.get('level') == level]
            
            # Ensure we get exactly 20 questions
            if len(questions) >= 20:
                questions = questions[:20]  # Take first 20 questions
            elif len(questions) == 0:
                return jsonify({'success': False, 'error': f'No questions found for {level} level', 'questions': []}), 404
            else:
                return jsonify({'success': False, 'error': f'Only {len(questions)} questions available for {level} level (need 20)', 'questions': []}), 404
        
        return jsonify({'success': True, 'questions': questions, 'course_title': data.get('course_title', 'Unknown Course')})
        
    except Exception as e:
        logger.error(f"Error getting quiz questions: {e}")
        return jsonify({'success': False, 'error': str(e), 'questions': []}), 500

# API route for quiz available courses
@app.route('/api/quiz/available-courses')
def api_quiz_available_courses():
    try:
        import os, json
        available_courses = []
        
        # Read all JSON quiz files directly
        quiz_dir = 'data/quiz_questions'
        if os.path.exists(quiz_dir):
            for filename in os.listdir(quiz_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(quiz_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            course_id = data.get('course_id')
                            course_title = data.get('course_title', 'Unknown Course')
                            language = data.get('language', 'general')
                            
                            # Check if the course has questions for all levels
                            questions = data.get('questions', [])
                            basic_count = len([q for q in questions if q.get('level') == 'basic'])
                            intermediate_count = len([q for q in questions if q.get('level') == 'intermediate'])
                            advanced_count = len([q for q in questions if q.get('level') == 'advanced'])
                            
                            # Only include courses that have at least 20 questions per level
                            if basic_count >= 20 and intermediate_count >= 20 and advanced_count >= 20:
                                available_courses.append({
                                    'id': course_id,
                                    'title': course_title,
                                    'language': language,
                                    'filename': filename,
                                    'question_counts': {
                                        'basic': basic_count,
                                        'intermediate': intermediate_count,
                                        'advanced': advanced_count
                                    }
                                })
                    except Exception as e:
                        logger.error(f"Error reading quiz file {filename}: {e}")
                        continue
        
        return jsonify({
            'success': True,
            'courses': available_courses
        })
    except Exception as e:
        logger.error(f"Error in API quiz available courses: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'courses': []
        }), 500

# API route for course questions
@app.route('/api/quiz/course-questions')
def api_quiz_course_questions():
    try:
        course_id = request.args.get('course_id')
        difficulty = request.args.get('difficulty', 'basic')
        topic_id = request.args.get('topic_id', None)
        
        if not course_id:
            return jsonify({
                'success': False,
                'error': 'Course ID is required',
                'questions': []
            }), 400
        
        # Try to load quiz questions from JSON file
        quiz_dir = 'data/quiz_questions'
        questions = []
        
        if os.path.exists(quiz_dir):
            for file in os.listdir(quiz_dir):
                if file.endswith('.json'):
                    quiz_path = os.path.join(quiz_dir, file)
                    try:
                        with open(quiz_path, 'r', encoding='utf-8') as f:
                            quiz_data = json.load(f)
                            
                        file_course_id = str(quiz_data.get('course_id', ''))
                        
                        if file_course_id == str(course_id):
                            quiz_questions = quiz_data.get('questions', [])
                            
                            # Filter by difficulty if specified and not 'all'
                            if difficulty != 'all':
                                quiz_questions = [q for q in quiz_questions if q.get('level') == difficulty]
                            
                            # Filter by topic if specified
                            if topic_id:
                                quiz_questions = [q for q in quiz_questions if q.get('topic_id') == topic_id]
                            
                            questions.extend(quiz_questions)
                            
                    except Exception as e:
                        logger.error(f"Error reading quiz file {file}: {e}")
                        continue
        
        # If we found questions, return them (limit to 15 for manageable quiz)
        if questions:
            # Shuffle questions for variety
            import random
            random.shuffle(questions)
            selected_questions = questions[:15]
            
            # Ensure questions have proper structure
            formatted_questions = []
            for i, q in enumerate(selected_questions):
                formatted_q = {
                    'id': q.get('id', i + 1),
                    'question': q.get('question', q.get('text', 'Sample question')),
                    'options': q.get('options', q.get('choices', ['Option A', 'Option B', 'Option C', 'Option D'])),
                    'correct_answer': q.get('correct_answer', q.get('correct', 0)),
                    'level': q.get('level', difficulty),
                    'explanation': q.get('explanation', 'This is the correct answer.'),
                    'topic_id': q.get('topic_id', topic_id)
                }
                formatted_questions.append(formatted_q)
            
            return jsonify({
                'success': True,
                'questions': formatted_questions,
                'source': 'json_file',
                'total_available': len(questions)
            })
        
        # Fallback: Generate topic-specific mock questions
        mock_questions = generate_mock_questions(course_id, topic_id, difficulty, 10)
        
        return jsonify({
            'success': True,
            'questions': mock_questions,
            'source': 'generated',
            'total_available': len(mock_questions)
        })
        
    except Exception as e:
        logger.error(f"Error in API quiz course questions: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'questions': []
        }), 500

def generate_mock_questions(course_id, topic_id, difficulty, count):
    """Generate mock questions based on course and topic"""
    base_questions = {
        'python': [
            {
                'question': 'What is the correct way to create a list in Python?',
                'options': ['list = []', 'list = ()', 'list = {}', 'list = <>'],
                'correct_answer': 0,
                'explanation': 'Square brackets [] are used to create lists in Python.'
            },
            {
                'question': 'Which keyword is used to define a function in Python?',
                'options': ['function', 'def', 'func', 'define'],
                'correct_answer': 1,
                'explanation': 'The "def" keyword is used to define functions in Python.'
            },
            {
                'question': 'What does the len() function return?',
                'options': ['The length of an object', 'The type of an object', 'The value of an object', 'Nothing'],
                'correct_answer': 0,
                'explanation': 'len() returns the number of items in an object.'
            }
        ],
        'javascript': [
            {
                'question': 'How do you declare a variable in JavaScript?',
                'options': ['var myVar;', 'variable myVar;', 'v myVar;', 'declare myVar;'],
                'correct_answer': 0,
                'explanation': 'Variables in JavaScript are declared using var, let, or const keywords.'
            },
            {
                'question': 'What is the correct way to write a JavaScript array?',
                'options': ['var colors = "red", "green", "blue"', 'var colors = ["red", "green", "blue"]', 'var colors = (red, green, blue)', 'var colors = {red, green, blue}'],
                'correct_answer': 1,
                'explanation': 'JavaScript arrays are written with square brackets and comma-separated values.'
            }
        ]
    }
    
    # Determine course type based on course_id or use default
    course_type = 'python'  # Default
    if course_id:
        if '2' in str(course_id):
            course_type = 'javascript'
    
    questions = base_questions.get(course_type, base_questions['python'])
    
    # Generate questions with proper IDs
    mock_questions = []
    for i in range(min(count, len(questions))):
        q = questions[i % len(questions)]
        mock_q = {
            'id': i + 1,
            'question': f"[{difficulty.title()}] {q['question']}",
            'options': q['options'],
            'correct_answer': q['correct_answer'],
            'level': difficulty,
            'explanation': q['explanation'],
            'topic_id': topic_id
        }
        mock_questions.append(mock_q)
    
    return mock_questions

# Quiz Session Management
@app.route('/api/quiz/start-session', methods=['POST'])
def start_quiz_session():
    try:
        data = request.get_json()
        course_id = data.get('course_id')
        topic_id = data.get('topic_id')
        difficulty = data.get('difficulty', 'basic')
        
        user_id = session.get('user_id')
        
        # Create a quiz session ID
        session_id = f"{course_id}_{topic_id}_{difficulty}_{datetime.now().timestamp()}"
        
        # Store quiz session in session storage
        session[f'quiz_session_{session_id}'] = {
            'course_id': course_id,
            'topic_id': topic_id,
            'difficulty': difficulty,
            'started_at': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Quiz session started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting quiz session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/quiz/session/<session_id>/questions')
def get_quiz_session_questions(session_id):
    try:
        # Get session data
        quiz_session = session.get(f'quiz_session_{session_id}')
        if not quiz_session:
            return jsonify({'success': False, 'error': 'Quiz session not found'}), 404
        
        course_id = quiz_session['course_id']
        topic_id = quiz_session['topic_id']
        difficulty = quiz_session['difficulty']
        
        # Get questions for this session
        questions_response = api_quiz_course_questions()
        
        # Parse the response
        if hasattr(questions_response, 'get_json'):
            questions_data = questions_response.get_json()
        else:
            questions_data = questions_response
            
        return jsonify(questions_data)
        
    except Exception as e:
        logger.error(f"Error getting quiz session questions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# API route for quiz submission
@app.route('/api/quiz/submit', methods=['POST'])
def api_quiz_submit():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        answers = data.get('answers', [])
        course_id = data.get('course_id')
        quiz_title = data.get('quiz_title', f'Course {course_id} Quiz')
        difficulty = data.get('difficulty', 'basic')
        language = data.get('language', 'python')
        user_id = session.get('user_id')

        # Calculate score (mock calculation)
        total_questions = len(answers)
        correct_answers = sum(1 for answer in answers if answer.get('is_correct', False))
        score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        # Save results to database if user is logged in
        if user_id:
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quiz_results 
                    (user_id, quiz_title, score, total, language, difficulty, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, quiz_title, int(score_percentage), total_questions, language, difficulty, datetime.now().isoformat()))
                conn.commit()
                conn.close()
            except Exception as db_error:
                logger.error(f"Database error saving quiz results: {db_error}")

        return jsonify({
            'success': True,
            'score': score_percentage,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'message': f'Quiz completed! You scored {score_percentage:.1f}%'
        })

    except Exception as e:
        logger.error(f"Error in API quiz submit: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Admin route to seed course structure
@app.route('/admin/seed-course-structure')
def seed_course_structure():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Sample course structure: Course → Modules (Topics) → Exercises → Tasks
        sample_courses = [
            {
                'id': 1,
                'title': 'Python Fundamentals',
                'description': 'Learn Python programming from basics to advanced concepts',
                'language': 'python',
                'difficulty': 'beginner',
                'modules': [
                    {
                        'topic_id': '1_0_0',
                        'title': 'Python Basics',
                        'exercises': [
                            {'title': 'Introduction to Python', 'type': 'theory', 'difficulty': 'beginner'},
                            {'title': 'Variables and Data Types', 'type': 'example', 'difficulty': 'beginner'},
                            {'title': 'Practice: Variables', 'type': 'practice', 'difficulty': 'beginner'},
                            {'title': 'Basic Operations Quiz', 'type': 'quiz', 'difficulty': 'beginner'},
                            {'title': 'Calculator Project', 'type': 'project', 'difficulty': 'intermediate'}
                        ]
                    },
                    {
                        'topic_id': '1_1_0',
                        'title': 'Control Structures',
                        'exercises': [
                            {'title': 'If Statements', 'type': 'theory', 'difficulty': 'beginner'},
                            {'title': 'Loops Overview', 'type': 'example', 'difficulty': 'beginner'},
                            {'title': 'Practice: Loops', 'type': 'practice', 'difficulty': 'intermediate'},
                            {'title': 'Control Flow Quiz', 'type': 'quiz', 'difficulty': 'intermediate'},
                            {'title': 'Number Guessing Game', 'type': 'project', 'difficulty': 'intermediate'}
                        ]
                    }
                ]
            },
            {
                'id': 2,
                'title': 'JavaScript Essentials',
                'description': 'Master JavaScript for web development',
                'language': 'javascript',
                'difficulty': 'beginner',
                'modules': [
                    {
                        'topic_id': '2_0_0',
                        'title': 'JavaScript Fundamentals',
                        'exercises': [
                            {'title': 'Introduction to JavaScript', 'type': 'theory', 'difficulty': 'beginner'},
                            {'title': 'Variables and Functions', 'type': 'example', 'difficulty': 'beginner'},
                            {'title': 'Practice: Functions', 'type': 'practice', 'difficulty': 'beginner'},
                            {'title': 'JavaScript Basics Quiz', 'type': 'quiz', 'difficulty': 'beginner'},
                            {'title': 'Interactive Webpage', 'type': 'project', 'difficulty': 'intermediate'}
                        ]
                    }
                ]
            }
        ]
        
        # Insert courses
        for course in sample_courses:
            cursor.execute('''
                INSERT OR REPLACE INTO courses (id, title, description, language, difficulty)
                VALUES (?, ?, ?, ?, ?)
            ''', (course['id'], course['title'], course['description'], 
                  course['language'], course['difficulty']))
            
            # Insert modules and exercises
            for module in course['modules']:
                for i, exercise in enumerate(module['exercises']):
                    cursor.execute('''
                        INSERT OR REPLACE INTO sub_exercises 
                        (course_id, topic_id, sub_exercise_index, title, exercise_type, 
                         difficulty, estimated_time, description, content, instructions)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        course['id'],
                        module['topic_id'],
                        i + 1,
                        exercise['title'],
                        exercise['type'],
                        exercise['difficulty'],
                        15,  # estimated_time
                        f"Learn about {exercise['title']}",
                        f"<h3>{exercise['title']}</h3><p>Content for {exercise['title']} goes here.</p>",
                        f"Complete the {exercise['title']} exercise."
                    ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully seeded {len(sample_courses)} courses with modules and exercises',
            'courses_created': len(sample_courses)
        })
        
    except Exception as e:
        logger.error(f"Error seeding course structure: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Route to get user's course progress dashboard
@app.route('/api/user/progress')
def get_user_progress_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all courses with user progress
        cursor.execute('''
            SELECT c.id, c.title, c.description, c.language, c.difficulty,
                   COUNT(DISTINCT se.id) as total_exercises,
                   COUNT(DISTINCT sep.id) as completed_exercises,
                   AVG(sep.score) as avg_score,
                   MAX(cp.updated_at) as last_activity
            FROM courses c
            LEFT JOIN sub_exercises se ON c.id = se.course_id
            LEFT JOIN sub_exercise_progress sep ON se.id = sep.sub_exercise_id AND sep.user_id = ?
            LEFT JOIN course_progress cp ON c.id = cp.course_id AND cp.user_id = ?
            GROUP BY c.id, c.title, c.description, c.language, c.difficulty
            ORDER BY last_activity DESC, c.id
        ''', (user_id, user_id))
        
        courses_progress = cursor.fetchall()
        
        progress_data = []
        for course in courses_progress:
            total_ex = course['total_exercises'] or 0
            completed_ex = course['completed_exercises'] or 0
            progress_percentage = (completed_ex / total_ex * 100) if total_ex > 0 else 0;
            
            progress_data.append({
                'course_id': course['id'],
                'title': course['title'],
                'description': course['description'],
                'language': course['language'],
                'difficulty': course['difficulty'],
                'total_exercises': total_ex,
                'completed_exercises': completed_ex,
                'progress_percentage': round(progress_percentage, 1),
                'avg_score': round(course['avg_score'] or 0, 1),
                'last_activity': course['last_activity'],
                'can_continue': completed_ex > 0 and progress_percentage < 100
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'progress': progress_data,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error getting user progress: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Route to get next topic/module for continue course
@app.route('/api/course/<int:course_id>/next-topic')
def get_next_topic(course_id):
    user_id = session.get('user_id')
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if user_id:
            # Find next incomplete topic
            cursor.execute('''
                SELECT DISTINCT se.topic_id,
                       COUNT(se.id) as total_exercises,
                       COUNT(sep.id) as completed_exercises
                FROM sub_exercises se
                LEFT JOIN sub_exercise_progress sep ON se.id = sep.sub_exercise_id 
                    AND sep.user_id = ? AND sep.status = 'completed'
                WHERE se.course_id = ?
                GROUP BY se.topic_id
                HAVING completed_exercises < total_exercises
                ORDER BY se.topic_id
                LIMIT 1
            ''', (user_id, course_id))
            
            next_topic = cursor.fetchone()
            
            if next_topic:
                topic_id = next_topic['topic_id']
            else:
                # All topics completed or start from beginning
                cursor.execute('''
                    SELECT DISTINCT topic_id 
                    FROM sub_exercises 
                    WHERE course_id = ? 
                    ORDER BY topic_id 
                    LIMIT 1
                ''', (course_id,))
                
                first_topic = cursor.fetchone()
                topic_id = first_topic['topic_id'] if first_topic else f"{course_id}_0_0"
        else:
            # No user, start from first topic
            cursor.execute('''
                SELECT DISTINCT topic_id 
                FROM sub_exercises 
                WHERE course_id = ? 
                ORDER BY topic_id 
                LIMIT 1
            ''', (course_id,))
            
            first_topic = cursor.fetchone()
            topic_id = first_topic['topic_id'] if first_topic else f"{course_id}_0_0"
        
        conn.close()
        
        return jsonify({
            'success': True,
            'next_topic': topic_id,
            'course_id': course_id
        })
        
    except Exception as e:
        logger.error(f"Error getting next topic: {e}")
        return jsonify({
            'success': False, 
            'next_topic': f"{course_id}_0_0",
            'course_id': course_id
        })

# Enhanced emotion detection route with real computer vision
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    print('DEBUG: /detect_emotion called')
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400
    frame_data = data['frame']
    print(f'DEBUG: Received frame data, length={len(frame_data)}')
    detection_mode = data.get('detection_mode', 'analysis')
    # Emotion/face recognition code removed
    return jsonify({'success': False, 'error': 'Emotion detection not available'})

# Submit quiz results from topic quiz
@app.route('/topic/<topic_id>/submit_quiz', methods=['POST'])
def submit_topic_quiz(topic_id):
    user_id = session.get('user_id')
    
    try:
        data = request.get_json() if request.is_json else request.form
        # Extract quiz data
        answers = data.get('answers', [])
        course_id = data.get('course_id')
        total_questions = len(answers)
        # Calculate score
        correct_answers = 0
        for answer in answers:
            if answer.get('is_correct', False):
                correct_answers += 1
        score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        # Save to database if user is logged in
        # ...existing code...
    except Exception as e:
        logger.error(f"Error submitting topic quiz: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/available_courses', methods=['GET'])
def get_available_courses():
    """Fetch all available courses with details."""
    try:
        courses = list_courses()
        available_courses = []
        for course in courses:
            available_courses.append({
                'id': course['id'] if isinstance(course, dict) else course[0],
                'title': course['title'] if isinstance(course, dict) else course[1],
                'description': course['description'] if isinstance(course, dict) else (course[2] if len(course) > 2 else '')
            })
        return jsonify({"success": True, "courses": available_courses})
    except Exception as e:
        logger.error(f"Error fetching available courses: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/user_courses', methods=['GET'])
def get_user_courses():
    """Fetch courses started by the user."""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "User not logged in"})

    user_id = session['user_id']
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT course_id FROM user_courses WHERE user_id = ?", (user_id,))
        user_courses = [row['course_id'] for row in cursor.fetchall()]
        return jsonify({"success": True, "user_courses": user_courses})
    except Exception as e:
        logger.error(f"Error fetching user courses: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/start_course', methods=['POST'])
def start_course():
    """Mark a course as started by the user."""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "User not logged in"})

    user_id = session['user_id']
    course_id = request.json.get('course_id')

    if not course_id:
        return jsonify({"success": False, "error": "Course ID is required"})

    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO user_courses (user_id, course_id) VALUES (?, ?)", (user_id, course_id)
        )
        connection.commit()
        return jsonify({"success": True, "message": "Course started successfully"})
    except Exception as e:
        logger.error(f"Error starting course: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/emotion', methods=['POST'])
def save_emotion_data():
    """Save emotion data from frontend during learning."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    user_id = session['user_id']
    data = request.json
    emotion = data.get('emotion')
    course_id = data.get('course_id')
    topic_id = data.get('topic_id')
    timestamp = datetime.now().isoformat()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO emotions (user_id, emotion, timestamp, course_id, topic_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, emotion, timestamp, course_id, topic_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Note: /api/module_status is defined at line ~1100 with enhanced functionality

@app.route('/api/emotion_analyze', methods=['POST'])
def emotion_analyze():
    """Analyze emotion from webcam frame using ML model."""
    data = request.json
    image_data = data.get('image')
    course_id = data.get('course_id')
    topic_id = data.get('topic_id')
    # Emotion/face recognition code removed
    emotion = 'unknown'
    suggestion = 'Emotion detection not available.'
    suggestion_class = 'text-muted'
    return jsonify({
        'success': True,
        'emotion': emotion,
        'suggestion': suggestion,
        'suggestion_class': suggestion_class
    })

@app.route('/quiz_review', methods=['POST'])
def quiz_review():
    import json
    answers_json = request.form.get('answers')
    review_questions = []
    if answers_json:
        try:
            review_questions = json.loads(answers_json)
        except Exception as e:
            return f"Error parsing answers: {e}", 400
    return render_template('quiz_review.html', questions=review_questions)

# Career Recommendations API
@app.route('/api/career-recommendations')
def get_career_recommendations():
    """Get all available career recommendations with their tech stack requirements"""
    try:
        # Load career recommendations from JSON file
        career_file = 'career_recommendations.json'
        if not os.path.exists(career_file):
            return jsonify({'success': False, 'error': 'Career recommendations not found'}), 404
        
        with open(career_file, 'r', encoding='utf-8') as f:
            career_data = json.load(f)
        
        # Get list of available courses from quiz files
        available_languages = set()
        quiz_dir = 'data/quiz_questions'
        
        if os.path.exists(quiz_dir):
            for filename in os.listdir(quiz_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(quiz_dir, filename), 'r', encoding='utf-8') as f:
                            quiz_data = json.load(f)
                            language = quiz_data.get('language', '').lower()
                            if language:
                                available_languages.add(language)
                    except Exception as e:
                        logger.error(f"Error reading quiz file {filename}: {e}")
                        continue
        
        # Process careers to only show tech stack items that are available
        processed_careers = []
        
        for career in career_data.get('careers', []):
            career_copy = dict(career)
            available_skills = []
            
            # Filter skills based on available languages
            for skill in career.get('required_skills', []):
                skill_info = career_data.get('skill_mapping', {}).get(skill, {})
                skill_language = skill_info.get('language', '').lower()
                
                # Only include skills if their language is available in quiz files
                if skill_language in available_languages or skill_language in ['html', 'docker', 'aws']:
                    available_skills.append({
                        'skill_id': skill,
                        'display_name': skill_info.get('display_name', skill),
                        'language': skill_language,
                        'course_id': skill_info.get('course_id')
                    })
            
            # Only include career if it has at least one available skill
            if available_skills:
                career_copy['available_skills'] = available_skills
                career_copy['total_skills'] = len(available_skills)
                career_copy['skill_count'] = f"{len(available_skills)} technologies"
                processed_careers.append(career_copy)
        
        return jsonify({
            'success': True,
            'careers': processed_careers,
            'available_languages': list(available_languages)
        })
        
    except Exception as e:
        logger.error(f"Error getting career recommendations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/career-recommendations/<career_id>')
def get_career_detail(career_id):
    """Get detailed information about a specific career path"""
    try:
        career_file = 'career_recommendations.json'
        if not os.path.exists(career_file):
            return jsonify({'success': False, 'error': 'Career data not found'}), 404
        
        with open(career_file, 'r', encoding='utf-8') as f:
            career_data = json.load(f)
        
        # Find the career
        selected_career = None
        for career in career_data.get('careers', []):
            if career.get('id') == career_id:
                selected_career = career
                break
        
        if not selected_career:
            return jsonify({'success': False, 'error': 'Career not found'}), 404
        
        # Get available languages from quiz files
        available_languages = set()
        quiz_dir = 'data/quiz_questions'
        course_info = {}
        
        if os.path.exists(quiz_dir):
            for filename in os.listdir(quiz_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(quiz_dir, filename), 'r', encoding='utf-8') as f:
                            quiz_data = json.load(f)
                            language = quiz_data.get('language', '').lower()
                            course_id = quiz_data.get('course_id')
                            course_title = quiz_data.get('course_title')
                            
                            if language:
                                available_languages.add(language)
                                course_info[course_id] = {
                                    'title': course_title,
                                    'language': language
                                }
                    except Exception as e:
                        logger.error(f"Error reading quiz file {filename}: {e}")
                        continue
        
        # Build detailed skill information
        skill_mapping = career_data.get('skill_mapping', {})
        detailed_skills = []
        
        for skill_id in selected_career.get('required_skills', []):
            skill_info = skill_mapping.get(skill_id, {})
            skill_language = skill_info.get('language', '').lower()
            course_id = skill_info.get('course_id')
            
            # Check if this skill's language is available
            is_available = skill_language in available_languages or skill_language in ['html', 'docker', 'aws']
            
            if is_available:
                course_data = course_info.get(course_id, {})
                detailed_skills.append({
                    'skill_id': skill_id,
                    'display_name': skill_info.get('display_name', skill_id),
                    'language': skill_language,
                    'course_id': course_id,
                    'course_title': course_data.get('title', 'Unknown Course'),
                    'is_available': True
                })
        
        # Add user progress if logged in
        user_progress = {}
        if 'user_id' in session:
            user_id = session['user_id']
            conn = get_connection()
            cursor = conn.cursor()
            
            for skill in detailed_skills:
                course_id = skill['course_id']
                cursor.execute('''
                    SELECT AVG(progress_percentage) as avg_progress
                    FROM course_progress 
                    WHERE user_id = ? AND course_id = ?
                ''', (user_id, course_id))
                
                progress_result = cursor.fetchone()
                user_progress[course_id] = progress_result['avg_progress'] if progress_result else 0
            
            conn.close()
        
        return jsonify({
            'success': True,
            'career': {
                'id': selected_career.get('id'),
                'title': selected_career.get('title'),
                'description': selected_career.get('description'),
                'icon': selected_career.get('icon'),
                'color': selected_career.get('color'),
                'skills': detailed_skills,
                'total_skills': len(detailed_skills),
                'user_progress': user_progress
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting career detail: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
