#!/usr/bin/env python3
"""
Legal AI Contract Analyzer - Main Flask Application
Advanced contract analysis with persona-based summaries and risk assessment
"""

import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sqlite3
from pathlib import Path

# Import our custom modules
from models.contract_analyzer import ContractAnalyzer
from models.risk_assessor import RiskAssessor
from models.persona_summarizer import PersonaSummarizer
from utils.text_processing import TextProcessor
from utils.visualization import VisualizationGenerator
from data.cuad_processor import CUADProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'legal_ai_secret_key_change_in_production'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI components
contract_analyzer = None
risk_assessor = None
persona_summarizer = None
text_processor = None
viz_generator = None
cuad_processor = None

def init_models():
    """Initialize all AI models and processors"""
    global contract_analyzer, risk_assessor, persona_summarizer
    global text_processor, viz_generator, cuad_processor
    
    try:
        logger.info("Initializing AI models...")
        contract_analyzer = ContractAnalyzer()
        risk_assessor = RiskAssessor()
        persona_summarizer = PersonaSummarizer()
        text_processor = TextProcessor()
        viz_generator = VisualizationGenerator()
        cuad_processor = CUADProcessor()
        logger.info("All models initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def init_db():
    """Initialize SQLite database for storing analysis results"""
    conn = sqlite3.connect('legal_contracts.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contract_analyses (
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
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with persona selection"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_contract():
    """Handle contract upload and persona selection"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    # Handle POST request
    if 'contract_file' not in request.files and 'contract_text' not in request.form:
        return jsonify({'error': 'No file or text provided'}), 400
    
    # Get persona and context
    persona = request.form.get('persona', 'general')
    user_context = request.form.get('user_context', '')
    
    contract_text = ""
    filename = "pasted_text.txt"
    
    # Process file upload or pasted text
    if 'contract_file' in request.files:
        file = request.files['contract_file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            contract_text = text_processor.extract_text_from_file(file_path)
            
            # Clean up uploaded file
            os.remove(file_path)
    else:
        contract_text = request.form.get('contract_text', '')
    
    if not contract_text.strip():
        return jsonify({'error': 'No contract content found'}), 400
    
    try:
        # Analyze contract
        analysis_results = analyze_contract(
            contract_text, persona, user_context, filename
        )
        
        # Store in database
        store_analysis(filename, persona, user_context, contract_text, analysis_results)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def analyze_contract(contract_text, persona, user_context, filename):
    """Perform comprehensive contract analysis"""
    
    # 1. Basic contract analysis
    logger.info(f"Analyzing contract: {filename}")
    contract_analysis = contract_analyzer.analyze(contract_text)
    
    # 2. Risk assessment
    risk_analysis = risk_assessor.assess_risks(
        contract_text, contract_analysis, user_context
    )
    
    # 3. Persona-based summary
    persona_summary = persona_summarizer.generate_summary(
        contract_text, contract_analysis, persona, user_context
    )
    
    # 4. Generate visualizations
    visualizations = viz_generator.create_visualizations(
        contract_analysis, risk_analysis
    )
    
    # Compile results
    results = {
        'filename': filename,
        'persona': persona,
        'user_context': user_context,
        'timestamp': datetime.now().isoformat(),
        
        # Core analysis
        'contract_summary': contract_analysis.get('summary', ''),
        'key_parties': contract_analysis.get('parties', []),
        'contract_type': contract_analysis.get('type', 'Unknown'),
        'clauses': contract_analysis.get('clauses', []),
        
        # Persona-specific results
        'persona_summary': persona_summary.get('summary', ''),
        'persona_highlights': persona_summary.get('highlights', []),
        'persona_warnings': persona_summary.get('warnings', []),
        
        # Risk assessment
        'overall_risk_score': risk_analysis.get('overall_score', 0),
        'risk_breakdown': risk_analysis.get('breakdown', {}),
        'risk_heatmap': risk_analysis.get('heatmap', []),
        'personalized_advice': risk_analysis.get('advice', []),
        
        # Visualizations
        'knowledge_graph': visualizations.get('knowledge_graph', {}),
        'risk_visualization': visualizations.get('risk_chart', {}),
        'clause_timeline': visualizations.get('timeline', {}),
        
        # Interactive elements
        'clickable_clauses': _create_clickable_clauses(
            contract_analysis.get('clauses', []), risk_analysis
        )
    }
    
    return results

def _create_clickable_clauses(clauses, risk_analysis):
    """Create interactive clause data for frontend"""
    clickable_clauses = []
    
    for clause in clauses:
        clause_risk = risk_analysis.get('clause_risks', {}).get(clause['id'], {})
        
        clickable_clause = {
            'id': clause['id'],
            'text': clause['text'],
            'type': clause['type'],
            'risk_level': clause_risk.get('level', 'low'),
            'risk_score': clause_risk.get('score', 0),
            'explanation': clause_risk.get('explanation', ''),
            'plain_english': clause.get('plain_english', ''),
            'concerns': clause_risk.get('concerns', []),
            'suggestions': clause_risk.get('suggestions', [])
        }
        
        clickable_clauses.append(clickable_clause)
    
    return clickable_clauses

def store_analysis(filename, persona, user_context, contract_text, results):
    """Store analysis results in database"""
    try:
        conn = sqlite3.connect('legal_contracts.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO contract_analyses 
            (filename, persona, user_context, contract_text, summary, 
             risk_score, risk_analysis, clauses_json, visualization_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            persona,
            user_context,
            contract_text,
            results.get('persona_summary', ''),
            results.get('overall_risk_score', 0),
            json.dumps(results.get('risk_breakdown', {})),
            json.dumps(results.get('clauses', [])),
            json.dumps({
                'knowledge_graph': results.get('knowledge_graph', {}),
                'risk_visualization': results.get('risk_visualization', {})
            })
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Analysis stored for {filename}")
        
    except Exception as e:
        logger.error(f"Database storage error: {str(e)}")

@app.route('/results')
def show_results():
    """Display analysis results page"""
    return render_template('results.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """REST API endpoint for contract analysis"""
    data = request.get_json()
    
    if not data or 'contract_text' not in data:
        return jsonify({'error': 'No contract text provided'}), 400
    
    contract_text = data['contract_text']
    persona = data.get('persona', 'general')
    user_context = data.get('user_context', '')
    filename = data.get('filename', 'api_submission.txt')
    
    try:
        results = analyze_contract(contract_text, persona, user_context, filename)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Get analysis history for current user"""
    try:
        conn = sqlite3.connect('legal_contracts.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, upload_time, persona, risk_score
            FROM contract_analyses
            ORDER BY upload_time DESC
            LIMIT 20
        ''')
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'id': row[0],
                'filename': row[1],
                'upload_time': row[2],
                'persona': row[3],
                'risk_score': row[4]
            })
        
        conn.close()
        return jsonify(history)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': all([
            contract_analyzer is not None,
            risk_assessor is not None,
            persona_summarizer is not None
        ])
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    # Initialize database and models
    init_db()
    init_models()
    
    # Run the application
    logger.info("Starting Legal AI Contract Analyzer...")
    app.run(debug=True, host='0.0.0.0', port=5000)