import os

# ============================================================
# ENVIRONMENT DETECTION
# ============================================================

IS_RENDER = os.environ.get('RENDER', False)
IS_PRODUCTION = os.environ.get('DEBUG', 'False').lower() == 'false'

# Database path for Render
if IS_RENDER:
    DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leads.db')
    print(f"🗄️  Render mode - Database at: {DATABASE}")

    # ============================================================
# app.py - LeadScout Standalone with XGBoost Support
# ============================================================

import os
import json
import sqlite3
import traceback
import logging
from datetime import datetime
from contextlib import contextmanager

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try to import xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# Import AI Assistant
from ai_assistant import AILeadAssistant, init_ai_database

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'leadscout-secret-key')
CORS(app)

# ============================================================
# DATABASE SETUP
# ============================================================

DATABASE = os.environ.get('DATABASE_PATH', 'leads.db')

# Force absolute path to ensure database is in the project folder
if not os.path.isabs(DATABASE):
    DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATABASE)
    os.environ['DATABASE_PATH'] = DATABASE

print(f"📁 Database path: {DATABASE}")

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_connection():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    """Initialize database with all required tables and indexes"""
    try:
        with db_connection() as conn:
            # Create leads table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_id INTEGER,
                    client_data TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability_yes REAL NOT NULL,
                    probability_no REAL NOT NULL,
                    confidence REAL NOT NULL,
                    priority TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'single'
                )
            ''')
            
            # Create indexes for leads
            conn.execute('CREATE INDEX IF NOT EXISTS idx_leads_display ON leads(display_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON leads(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_prediction ON leads(prediction)')
            
            # Create chat_history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for chat_history
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_lead ON chat_history(lead_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp)')
            
            logger.info("✅ Database initialized successfully")
            print(f"✅ Database tables created at: {DATABASE}")
            
            # Verify tables were created
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📊 Tables: {tables}")
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def migrate_existing_data():
    """Migrate existing data if needed"""
    try:
        with db_connection() as conn:
            # Check if display_id column exists
            cursor = conn.execute("PRAGMA table_info(leads)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'display_id' not in columns:
                conn.execute("ALTER TABLE leads ADD COLUMN display_id INTEGER")
                logger.info("✅ Added display_id column")
            
            # Assign display IDs to existing leads
            cursor = conn.execute("SELECT id FROM leads ORDER BY id ASC")
            leads = cursor.fetchall()
            for display_num, (lead_id,) in enumerate(leads, 1):
                conn.execute(
                    "UPDATE leads SET display_id = ? WHERE id = ?",
                    (display_num, lead_id)
                )
            if leads:
                logger.info(f"✅ Assigned display IDs to {len(leads)} existing leads")
    except Exception as e:
        logger.warning(f"⚠️ Migration warning: {e}")

# Initialize database immediately
print("\n" + "="*60)
print("🗄️  INITIALIZING DATABASE...")
print("="*60)

try:
    # Ensure the database directory exists
    db_dir = os.path.dirname(DATABASE)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"✅ Created database directory: {db_dir}")
    
    # Initialize database
    init_db()
    migrate_existing_data()
    print("✅ Database setup complete!")
except Exception as e:
    print(f"❌ CRITICAL: Database initialization failed: {e}")
    print("⚠️  The app may not work properly without a database.")
print("="*60 + "\n")

# ============================================================
# LOAD MODEL AND PREPROCESSOR (XGBOOST SUPPORT)
# ============================================================

model = None
preprocessor = None
model_type = None

def load_model():
    global model, preprocessor, model_type
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            logger.warning("⚠️ Models directory not found!")
            return False
            
        # Load preprocessor first
        preprocessor = joblib.load('models/preprocessor.pkl')
        logger.info("✅ Preprocessor loaded!")
        
        # Try loading model
        model_paths = [
            'models/best_model.pkl',
            'models/best_model.joblib',
            'models/xgboost_model.json',
            'models/xgboost_model.ubj',
            'models/model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"📂 Found model file: {path}")
                
                # Try loading with joblib first
                try:
                    model = joblib.load(path)
                    logger.info(f"✅ Model loaded with joblib from: {path}")
                    
                    # Check if it's XGBoost
                    if XGB_AVAILABLE and isinstance(model, (xgb.XGBClassifier, xgb.Booster)):
                        model_type = 'xgboost'
                        logger.info("✅ XGBoost model detected!")
                        
                        # If it's a Booster, wrap it
                        if isinstance(model, xgb.Booster):
                            logger.info("⚠️ Raw Booster detected - wrapping for compatibility")
                            class XGBClassifierWrapper:
                                def __init__(self, booster):
                                    self.booster = booster
                                    self._estimator_type = "classifier"
                                
                                def predict(self, X):
                                    import numpy as np
                                    import xgboost as xgb
                                    if isinstance(X, np.ndarray):
                                        dmatrix = xgb.DMatrix(X)
                                        predictions = self.booster.predict(dmatrix)
                                        return (predictions > 0.5).astype(int)
                                    return self.booster.predict(xgb.DMatrix(X))
                                
                                def predict_proba(self, X):
                                    import numpy as np
                                    import xgboost as xgb
                                    if isinstance(X, np.ndarray):
                                        dmatrix = xgb.DMatrix(X)
                                        probs = self.booster.predict(dmatrix)
                                        # Convert to 2-column format
                                        return np.column_stack([1 - probs, probs])
                                    probs = self.booster.predict(xgb.DMatrix(X))
                                    return np.column_stack([1 - probs, probs])
                            
                            model = XGBClassifierWrapper(model)
                            logger.info("✅ XGBoost Booster wrapped successfully!")
                    elif hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                        model_type = 'sklearn'
                        logger.info("✅ Scikit-learn model detected!")
                    else:
                        model_type = 'unknown'
                        logger.warning("⚠️ Unknown model type - will attempt to use as-is")
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load with joblib from {path}: {e}")
                    
                    # Try loading as XGBoost native format
                    if XGB_AVAILABLE and path.endswith(('.json', '.ubj')):
                        try:
                            model = xgb.XGBClassifier()
                            model.load_model(path)
                            model_type = 'xgboost'
                            logger.info(f"✅ XGBoost model loaded natively from: {path}")
                            return True
                        except Exception as e2:
                            logger.warning(f"⚠️ Failed to load XGBoost natively: {e2}")
        
        logger.error("❌ No model file found or all loading attempts failed!")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load the model
if load_model():
    logger.info("✅ Model and preprocessor loaded successfully!")
else:
    logger.warning("⚠️ Model loading failed - predictions will not work")

# ============================================================
# PREDICTION FUNCTION (XGBoost Aware)
# ============================================================

def make_prediction(features):
    """Make prediction using loaded model (XGBoost aware)"""
    global model, preprocessor, model_type
    
    if model is None:
        raise ValueError("Model not loaded")
    
    # Transform features
    transformed = preprocessor.transform(features)
    
    # Make prediction based on model type
    if model_type == 'xgboost':
        try:
            # Try standard sklearn-like interface first
            pred = model.predict(transformed)[0]
            proba = model.predict_proba(transformed)[0]
            return pred, proba
        except (TypeError, AttributeError):
            # Handle raw XGBoost Booster
            import xgboost as xgb
            dmatrix = xgb.DMatrix(transformed)
            predictions = model.predict(dmatrix)
            # Convert to probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(dmatrix)
            else:
                # For binary classification, sigmoid of raw predictions
                probs = 1 / (1 + np.exp(-predictions))
                probs = np.column_stack([1 - probs, probs])
            
            pred = int((probs[:, 1] > 0.5)[0])
            proba = probs[0]
            return pred, proba
    else:
        # Standard sklearn model
        pred = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]
        return pred, proba

# ============================================================
# FEATURE VALIDATION
# ============================================================

REQUIRED_FEATURES = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day', 'month', 'duration',
    'campaign', 'pdays', 'previous', 'poutcome'
]

# ============================================================
# ROUTES - HOME
# ============================================================

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/app')
def app_main():
    return render_template('index.html')

# ============================================================
# ROUTES - PREDICTION
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    global model, preprocessor
    
    # Check if model is loaded
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'status': 'error'
        }), 503

    try:
        # ... rest of your prediction code
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing)}',
                'status': 'error'
            }), 400
        
        # Prepare data
        df = pd.DataFrame([data])
        numeric_fields = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field])
        
        # Make prediction
        prediction, probability = make_prediction(df)
        
        is_yes = prediction == 1
        prob_yes = float(round(probability[1] * 100, 1))
        prob_no = float(round(probability[0] * 100, 1))
        
        # Determine priority
        if prob_yes >= 70:
            priority = 'High Priority'
            message = '🎯 Call this client immediately!'
        elif prob_yes >= 45:
            priority = 'Medium Priority'
            message = '📞 Consider calling this client'
        else:
            priority = 'Low Priority'
            message = '⏳ Skip this client for now'
        
        confidence = float(min(round(abs(prob_yes - 50) * 2, 1), 95))
        
        # Save to database
        with db_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO leads 
                (client_data, prediction, probability_yes, probability_no, 
                confidence, priority, message, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                json.dumps(data),
                'Yes' if is_yes else 'No',
                prob_yes,
                prob_no,
                confidence,
                priority,
                message,
                'single'
            ))
            
            lead_id = cursor.lastrowid
            count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            conn.execute("UPDATE leads SET display_id = ? WHERE id = ?", (count, lead_id))
        
        logger.info(f"Prediction: {data.get('job', 'unknown')} -> {'Yes' if is_yes else 'No'} ({prob_yes}%)")
        
        return jsonify({
            'id': lead_id,
            'display_id': count,
            'prediction': 'Yes' if is_yes else 'No',
            'probability_yes': prob_yes,
            'probability_no': prob_no,
            'recommendation': priority,
            'priority': priority,
            'message': message,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

# ============================================================
# ROUTES - BULK PREDICTION
# ============================================================

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded.',
            'status': 'error'
        }), 503

    try:
        data = request.get_json()
        
        if not data or 'clients' not in data:
            return jsonify({
                'error': 'Missing "clients" array',
                'status': 'error'
            }), 400
        
        clients = data['clients']
        if not isinstance(clients, list):
            return jsonify({
                'error': '"clients" must be an array',
                'status': 'error'
            }), 400
        
        if len(clients) > 1000:
            return jsonify({
                'error': 'Maximum 1000 clients allowed',
                'status': 'error'
            }), 400
        
        results = []
        saved_count = 0
        
        for client in clients:
            try:
                df = pd.DataFrame([client])
                prediction, probability = make_prediction(df)
                
                is_yes = prediction == 1
                prob_yes = round(probability[1] * 100, 1)
                prob_no = round(probability[0] * 100, 1)
                
                if prob_yes >= 70:
                    priority = 'High Priority'
                    message = '🎯 Call this client immediately!'
                elif prob_yes >= 45:
                    priority = 'Medium Priority'
                    message = '📞 Consider calling this client'
                else:
                    priority = 'Low Priority'
                    message = '⏳ Skip this client for now'
                
                confidence = min(round(abs(prob_yes - 50) * 2, 1), 95)
                
                with db_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO leads 
                        (client_data, prediction, probability_yes, probability_no, 
                         confidence, priority, message, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        json.dumps(client),
                        'Yes' if is_yes else 'No',
                        prob_yes,
                        prob_no,
                        confidence,
                        priority,
                        message,
                        'bulk'
                    ))
                    
                    lead_id = cursor.lastrowid
                    count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
                    conn.execute("UPDATE leads SET display_id = ? WHERE id = ?", (count, lead_id))
                    saved_count += 1
                
                results.append({
                    'client': client,
                    'prediction': 'Yes' if is_yes else 'No',
                    'probability_yes': prob_yes,
                    'probability_no': prob_no,
                    'priority': priority,
                    'confidence': confidence,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'client': client,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'successful': sum(1 for r in results if r['status'] == 'success'),
                'high_potential': sum(1 for r in results if r.get('prediction') == 'Yes'),
                'saved': saved_count
            },
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Bulk prediction failed: {str(e)}',
            'status': 'error'
        }), 500

# ============================================================
# ROUTES - LEADS MANAGEMENT
# ============================================================

@app.route('/leads', methods=['GET'])
def get_leads():
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        prediction = request.args.get('prediction', None)
        priority = request.args.get('priority', None)
        from_date = request.args.get('from_date', None)
        to_date = request.args.get('to_date', None)
        
        query = "SELECT id, display_id, client_data, prediction, probability_yes, probability_no, confidence, priority, message, timestamp, source FROM leads"
        params = []
        conditions = []
        
        if prediction:
            conditions.append("prediction = ?")
            params.append(prediction)
        if priority:
            conditions.append("priority = ?")
            params.append(priority)
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY display_id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with db_connection() as conn:
            leads = conn.execute(query, params).fetchall()
            
            count_query = "SELECT COUNT(*) as total FROM leads"
            if conditions:
                count_query += " WHERE " + " AND ".join(conditions)
            total = conn.execute(count_query, params[:-2]).fetchone()['total']
        
        leads_list = []
        for lead in leads:
            lead_dict = dict(lead)
            client_data = lead_dict['client_data']
            if isinstance(client_data, bytes):
                client_data = client_data.decode('utf-8')
            lead_dict['client_data'] = json.loads(client_data)
            leads_list.append(lead_dict)
        
        return jsonify({
            'leads': leads_list,
            'total': total,
            'limit': limit,
            'offset': offset,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error fetching leads: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/leads/<int:lead_id>', methods=['DELETE'])
def delete_lead(lead_id):
    try:
        with db_connection() as conn:
            lead = conn.execute("SELECT id FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if not lead:
                return jsonify({'error': 'Lead not found', 'status': 'error'}), 404
            
            conn.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
            conn.execute("DELETE FROM chat_history WHERE lead_id = ?", (lead_id,))
            
            remaining = conn.execute("SELECT id FROM leads ORDER BY id ASC").fetchall()
            for new_display, (row_id,) in enumerate(remaining, 1):
                conn.execute("UPDATE leads SET display_id = ? WHERE id = ?", (new_display, row_id))
            
            return jsonify({'message': 'Lead deleted successfully', 'status': 'success'})
    except Exception as e:
        logger.error(f"Error deleting lead: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/leads/clear', methods=['DELETE'])
def clear_all_leads():
    try:
        confirm = request.args.get('confirm', 'false')
        if confirm != 'true':
            return jsonify({'error': 'Confirmation required. Use ?confirm=true', 'status': 'error'}), 400
        
        with db_connection() as conn:
            count = conn.execute("SELECT COUNT(*) as total FROM leads").fetchone()['total']
            conn.execute("DELETE FROM leads")
            conn.execute("DELETE FROM chat_history")
            
            return jsonify({
                'message': f'All {count} leads deleted successfully.',
                'deleted_count': count,
                'status': 'success'
            })
    except Exception as e:
        logger.error(f"Error clearing leads: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/leads/stats', methods=['GET'])
def get_lead_stats():
    try:
        with db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as total FROM leads").fetchone()['total']
            high = conn.execute("SELECT COUNT(*) as total FROM leads WHERE prediction = 'Yes'").fetchone()['total']
            low = total - high
            
            monthly = conn.execute('''
                SELECT strftime('%Y-%m', timestamp) as month,
                       COUNT(*) as total,
                       SUM(CASE WHEN prediction = 'Yes' THEN 1 ELSE 0 END) as yes_count
                FROM leads GROUP BY month ORDER BY month DESC LIMIT 12
            ''').fetchall()
            
            jobs = conn.execute('''
                SELECT json_extract(client_data, '$.job') as job,
                       COUNT(*) as total,
                       SUM(CASE WHEN prediction = 'Yes' THEN 1 ELSE 0 END) as yes_count
                FROM leads WHERE json_extract(client_data, '$.job') IS NOT NULL
                GROUP BY job ORDER BY total DESC LIMIT 10
            ''').fetchall()
            
            avg_confidence = conn.execute("SELECT AVG(confidence) as avg FROM leads").fetchone()['avg'] or 0
            
            return jsonify({
                'total': total,
                'high_potential': high,
                'low_priority': low,
                'conversion_rate': round((high / total * 100) if total > 0 else 0, 1),
                'avg_confidence': round(avg_confidence, 1),
                'monthly': [dict(m) for m in monthly],
                'jobs': [dict(j) for j in jobs],
                'status': 'success'
            })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ============================================================
# ROUTES - LEAD SCORING
# ============================================================

def calculate_lead_score(client_data):
    """Calculate detailed lead score with breakdown"""
    score = 0
    breakdown = []
    insights = []
    max_score = 100
    
    # 1. Previous Outcome (25 points)
    poutcome = client_data.get('poutcome', 'unknown')
    if poutcome == 'success':
        score += 25
        breakdown.append({'factor': 'Previous Success', 'score': 25, 'max_score': 25, 'level': 'high', 'icon': '✅', 'description': 'Client subscribed before'})
        insights.append('✅ Client had previous success - mention this in conversation')
    elif poutcome == 'failure':
        score += 5
        breakdown.append({'factor': 'Previous Outcome', 'score': 5, 'max_score': 25, 'level': 'low', 'icon': '⚠️', 'description': 'Previous attempt failed'})
        insights.append('⚠️ Previous attempt failed - try a different approach')
    else:
        score += 10
        breakdown.append({'factor': 'Previous Outcome', 'score': 10, 'max_score': 25, 'level': 'medium', 'icon': '❓', 'description': 'No previous contact'})
        insights.append('❓ No previous contact - this is a fresh opportunity')
    
    # 2. Call Duration (20 points)
    duration = client_data.get('duration', 0)
    if duration > 300:
        score += 20
        breakdown.append({'factor': 'Call Duration', 'score': 20, 'max_score': 20, 'level': 'high', 'icon': '📞', 'description': 'Long engagement (300s+)'})
        insights.append('📞 Long call duration - client is engaged and interested')
    elif duration > 150:
        score += 12
        breakdown.append({'factor': 'Call Duration', 'score': 12, 'max_score': 20, 'level': 'medium', 'icon': '📞', 'description': 'Moderate engagement (150-300s)'})
        insights.append('📞 Moderate engagement - keep the conversation flowing')
    else:
        score += 5
        breakdown.append({'factor': 'Call Duration', 'score': 5, 'max_score': 20, 'level': 'low', 'icon': '📞', 'description': 'Short engagement (<150s)'})
        insights.append('📞 Short call - need to build rapport quickly')
    
    # 3. Account Balance (15 points)
    balance = client_data.get('balance', 0)
    if balance > 5000:
        score += 15
        breakdown.append({'factor': 'Account Balance', 'score': 15, 'max_score': 15, 'level': 'high', 'icon': '💰', 'description': 'High balance ($5,000+)'})
        insights.append('💰 High balance client - suggest investment products')
    elif balance > 1000:
        score += 10
        breakdown.append({'factor': 'Account Balance', 'score': 10, 'max_score': 15, 'level': 'medium', 'icon': '💰', 'description': 'Medium balance ($1,000-5,000)'})
        insights.append('💰 Good balance - discuss savings options')
    else:
        score += 3
        breakdown.append({'factor': 'Account Balance', 'score': 3, 'max_score': 15, 'level': 'low', 'icon': '💰', 'description': 'Low balance (<$1,000)'})
        insights.append('💰 Low balance - consider lower entry products')
    
    # 4. Job Type (12 points)
    high_value_jobs = ['management', 'admin.', 'technician', 'entrepreneur']
    medium_value_jobs = ['services', 'self-employed', 'retired']
    job = client_data.get('job', 'unknown')
    
    if job in high_value_jobs:
        score += 12
        breakdown.append({'factor': 'Job Type', 'score': 12, 'max_score': 12, 'level': 'high', 'icon': '👔', 'description': f'High-value job: {job}'})
        insights.append(f'👔 {job} professional - likely has disposable income')
    elif job in medium_value_jobs:
        score += 8
        breakdown.append({'factor': 'Job Type', 'score': 8, 'max_score': 12, 'level': 'medium', 'icon': '👔', 'description': f'Medium-value job: {job}'})
        insights.append(f'👔 {job} - stable income source')
    else:
        score += 4
        breakdown.append({'factor': 'Job Type', 'score': 4, 'max_score': 12, 'level': 'low', 'icon': '👔', 'description': f'Entry-level job: {job}'})
        insights.append(f'👔 {job} - may need more convincing')
    
    # 5. Education (10 points)
    education = client_data.get('education', 'unknown')
    if education == 'tertiary':
        score += 10
        breakdown.append({'factor': 'Education', 'score': 10, 'max_score': 10, 'level': 'high', 'icon': '🎓', 'description': 'Tertiary education'})
        insights.append('🎓 Highly educated - use technical terms and data')
    elif education == 'secondary':
        score += 6
        breakdown.append({'factor': 'Education', 'score': 6, 'max_score': 10, 'level': 'medium', 'icon': '🎓', 'description': 'Secondary education'})
        insights.append('🎓 Good education level - explain benefits clearly')
    else:
        score += 3
        breakdown.append({'factor': 'Education', 'score': 3, 'max_score': 10, 'level': 'low', 'icon': '🎓', 'description': 'Primary or unknown education'})
        insights.append('🎓 Keep explanations simple and clear')
    
    # 6. Age Group (10 points)
    age = client_data.get('age', 0)
    if 35 <= age <= 55:
        score += 10
        breakdown.append({'factor': 'Age Group', 'score': 10, 'max_score': 10, 'level': 'high', 'icon': '📅', 'description': 'Prime age (35-55)'})
        insights.append('📅 Prime age - likely to have savings and investments')
    elif age > 55:
        score += 8
        breakdown.append({'factor': 'Age Group', 'score': 8, 'max_score': 10, 'level': 'medium', 'icon': '📅', 'description': 'Senior (55+)'})
        insights.append('📅 Senior client - emphasize security and stability')
    else:
        score += 3
        breakdown.append({'factor': 'Age Group', 'score': 3, 'max_score': 10, 'level': 'low', 'icon': '📅', 'description': 'Young adult (<35)'})
        insights.append('📅 Young client - focus on long-term benefits')
    
    # 7. Previous Contacts (8 points)
    previous = client_data.get('previous', 0)
    if previous > 2:
        score += 8
        breakdown.append({'factor': 'Previous Contacts', 'score': 8, 'max_score': 8, 'level': 'high', 'icon': '🔄', 'description': f'Multiple contacts ({previous})'})
        insights.append('🔄 Multiple previous contacts - they know us, use this')
    elif previous > 0:
        score += 5
        breakdown.append({'factor': 'Previous Contacts', 'score': 5, 'max_score': 8, 'level': 'medium', 'icon': '🔄', 'description': f'Some contacts ({previous})'})
        insights.append('🔄 Some previous contact - reference past interactions')
    else:
        score += 2
        breakdown.append({'factor': 'Previous Contacts', 'score': 2, 'max_score': 8, 'level': 'low', 'icon': '🔄', 'description': 'First contact'})
        insights.append('🔄 First contact - make a good first impression')
    
    return {
        'total_score': score,
        'max_score': max_score,
        'percentage': round((score / max_score) * 100, 1),
        'breakdown': breakdown,
        'insights': insights[:5]
    }

@app.route('/lead-score/<int:lead_id>', methods=['GET'])
def get_lead_score(lead_id):
    try:
        with db_connection() as conn:
            lead = conn.execute(
                "SELECT client_data, prediction, probability_yes, confidence FROM leads WHERE id = ?",
                (lead_id,)
            ).fetchone()
            
            if not lead:
                return jsonify({'error': 'Lead not found', 'status': 'error'}), 404
            
            client_data = json.loads(lead['client_data'])
            score_data = calculate_lead_score(client_data)
            score_data['prediction'] = lead['prediction']
            score_data['probability'] = lead['probability_yes']
            score_data['confidence'] = lead['confidence']
            
            return jsonify({'score': score_data, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting lead score: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ============================================================
# ROUTES - CHAT HISTORY
# ============================================================

@app.route('/chat/save', methods=['POST'])
def save_chat():
    try:
        data = request.get_json()
        lead_id = data.get('lead_id')
        question = data.get('question')
        answer = data.get('answer')
        
        if not lead_id or not question or not answer:
            return jsonify({'error': 'Missing required fields'}), 400
        
        with db_connection() as conn:
            conn.execute('''
                INSERT INTO chat_history (lead_id, question, answer)
                VALUES (?, ?, ?)
            ''', (lead_id, question, answer))
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/history/<int:lead_id>', methods=['GET'])
def get_chat_history(lead_id):
    try:
        with db_connection() as conn:
            history = conn.execute('''
                SELECT id, question, answer, timestamp
                FROM chat_history
                WHERE lead_id = ?
                ORDER BY timestamp ASC
            ''', (lead_id,)).fetchall()
        
        return jsonify({
            'history': [dict(h) for h in history],
            'count': len(history),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/clear/<int:lead_id>', methods=['DELETE'])
def clear_chat_history(lead_id):
    try:
        with db_connection() as conn:
            conn.execute('DELETE FROM chat_history WHERE lead_id = ?', (lead_id,))
        return jsonify({'status': 'success', 'message': 'Chat history cleared'})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTES - AI ASSISTANT (GENERAL ONLY)
# ============================================================

try:
    init_ai_database()
    ai_assistant = AILeadAssistant()
    if ai_assistant and ai_assistant.available:
        print("✅ AI Assistant (Groq) initialized successfully!")
    else:
        print("⚠️ AI Assistant initialized but Groq not available. Check API key.")
except Exception as e:
    print(f"❌ AI Assistant initialization error: {e}")
    ai_assistant = None

@app.route('/ai/ask', methods=['POST'])
def ai_ask():
    if not ai_assistant or not ai_assistant.available:
        return jsonify({
            'error': 'AI Assistant not available. Check GROQ_API_KEY in .env file.',
            'status': 'error'
        }), 503
    
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Missing question', 'status': 'error'}), 400
        
        response = ai_assistant.ask(None, None, question, 'general')
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ AI Ask Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/ai/top-leads', methods=['GET'])
def ai_top_leads():
    if not ai_assistant:
        return jsonify({'error': 'AI Assistant not available'}), 503
    
    try:
        limit = request.args.get('limit', 5, type=int)
        leads = ai_assistant.get_top_leads(limit)
        return jsonify({'leads': leads, 'count': len(leads), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/health', methods=['GET'])
def ai_health():
    return jsonify({
        'available': ai_assistant is not None and ai_assistant.available,
        'provider': 'Groq' if ai_assistant and ai_assistant.available else None,
        'model': ai_assistant.model if ai_assistant else None,
        'status': 'healthy' if ai_assistant and ai_assistant.available else 'unavailable'
    })


# ============================================================
# ROUTES - UTILITY
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    db_healthy = False
    try:
        with db_connection() as conn:
            conn.execute("SELECT 1")
            db_healthy = True
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type,
        'preprocessor_loaded': preprocessor is not None,
        'database_healthy': db_healthy,
        'ai_available': ai_assistant is not None and ai_assistant.available,
        'xgboost_available': XGB_AVAILABLE,
        'timestamp': datetime.now().isoformat(),
        'version': '3.0'
    })

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        'required_features': REQUIRED_FEATURES,
        'valid_jobs': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                      'management', 'retired', 'self-employed', 'services', 
                      'student', 'technician', 'unemployed', 'unknown'],
        'valid_months': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'numeric_features': ['age', 'balance', 'day', 'duration', 
                            'campaign', 'pdays', 'previous']
    })

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 LeadScout Server Starting...")
    print("="*60)
    print(f"📡 Port: {port}")
    print(f"🔧 Debug: {debug}")
    print(f"🗄️  Database: {DATABASE}")
    print(f"🤖 AI Assistant: {'✅ Available' if ai_assistant and ai_assistant.available else '❌ Not Available'}")
    print(f"📊 XGBoost: {'✅ Available' if XGB_AVAILABLE else '❌ Not Installed'}")
    print(f"📁 Model: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    print(f"📁 Model Type: {model_type if model_type else 'N/A'}")
    print("📝 Access at: http://localhost:" + str(port))
    print("="*60 + "\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)