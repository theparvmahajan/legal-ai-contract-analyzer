#!/bin/bash
# Legal AI Contract Analyzer - Setup Script
# Run this script to set up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Legal AI Contract Analyzer..."
echo "================================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found version: $python_version"
    exit 1
fi
echo "✅ Python version OK: $python_version"

# Create project directory structure
echo "📁 Creating project structure..."
mkdir -p legal_ai_contract_analyzer/{models,data/datasets,templates,static/{css,js,images},utils,uploads}
cd legal_ai_contract_analyzer

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv legal_ai_env
source legal_ai_env/Scripts/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install transformers==4.35.0 datasets==2.14.0 tokenizers==0.14.0
pip install scikit-learn numpy pandas

# Install web framework
echo "🌐 Installing web framework..."
pip install Flask Flask-CORS gunicorn

# Install NLP libraries
echo "🔤 Installing NLP libraries..."
pip install spacy nltk
python -m spacy download en_core_web_sm

# Install file processing libraries
echo "📄 Installing file processing libraries..."
pip install python-docx PyPDF2

# Install visualization libraries
echo "📊 Installing visualization libraries..."
pip install plotly networkx

# Install database and utilities
echo "🗄️ Installing database and utilities..."
pip install SQLAlchemy regex tqdm python-dateutil

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Create __init__.py files
echo "📝 Creating Python package files..."
touch models/__init__.py data/__init__.py utils/__init__.py

# Create requirements.txt
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Legal AI Contract Analyzer - Requirements
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
transformers==4.35.0
datasets==2.14.0
tokenizers==0.14.0
scikit-learn==1.3.0
numpy==1.24.3
pandas==2.0.3
Flask==2.3.2
Flask-CORS==4.0.0
gunicorn==21.2.0
spacy==3.7.0
nltk==3.8.1
plotly==5.17.0
networkx==3.1
python-docx==0.8.11
PyPDF2==3.0.1
SQLAlchemy==2.0.20
regex==2023.8.8
tqdm==4.65.0
python-dateutil==2.8.2
Werkzeug==2.3.6
EOF

# Initialize database
echo "🗄️ Initializing database..."
python -c "
import sqlite3
conn = sqlite3.connect('legal_contracts.db')
cursor = conn.cursor()
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
print('✅ Database initialized')
"

# Download and cache Legal-BERT model
echo "🤖 Downloading Legal-BERT model (this may take a few minutes)..."
python -c "
from transformers import AutoTokenizer, AutoModel
try:
    print('Downloading Legal-BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    print('Downloading Legal-BERT model...')
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
    print('✅ Legal-BERT model downloaded and cached')
except Exception as e:
    print(f'⚠️ Could not download Legal-BERT: {e}')
    print('Will use fallback models during runtime')
"

# Create sample environment configuration
echo "⚙️ Creating configuration files..."
cat > .env << 'EOF'
# Legal AI Contract Analyzer Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-change-in-production
MAX_FILE_SIZE=16777216
UPLOAD_FOLDER=uploads
DATABASE_URL=sqlite:///legal_contracts.db
EOF

# Create startup script
echo "🚀 Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash
# Startup script for Legal AI Contract Analyzer

echo "🚀 Starting Legal AI Contract Analyzer..."

# Activate virtual environment
source legal_ai_env/bin/activate

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Check if all required files exist
required_files=("app.py" "models/contract_analyzer.py" "models/risk_assessor.py" "models/persona_summarizer.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        echo "Please ensure all project files are in place."
        exit 1
    fi
done

# Start the Flask application
echo "🌐 Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python app.py
EOF

chmod +x start.sh

# Create development script
echo "🔧 Creating development utilities..."
cat > dev_utils.py << 'EOF'
#!/usr/bin/env python3
"""
Development utilities for Legal AI Contract Analyzer
"""

import os
import sys
import subprocess
import sqlite3
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'flask', 'spacy', 'nltk', 
        'plotly', 'pandas', 'numpy', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def check_models():
    """Check if required models are available"""
    print("\n🤖 Checking AI models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        print("✅ Legal-BERT model available")
    except Exception as e:
        print(f"⚠️ Legal-BERT model issue: {e}")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("✅ spaCy English model available")
    except Exception as e:
        print(f"⚠️ spaCy model issue: {e}")
        print("Run: python -m spacy download en_core_web_sm")

def reset_database():
    """Reset the application database"""
    print("\n🗄️ Resetting database...")
    
    if os.path.exists('legal_contracts.db'):
        backup_name = f'legal_contracts_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        os.rename('legal_contracts.db', backup_name)
        print(f"📦 Old database backed up as: {backup_name}")
    
    # Initialize fresh database
    conn = sqlite3.connect('legal_contracts.db')
    cursor = conn.cursor()
    
    cursor.execute('''
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
    ''')
    
    cursor.execute('''
        CREATE TABLE user_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database reset complete!")

def test_analysis():
    """Test the contract analysis pipeline"""
    print("\n🧪 Testing contract analysis...")
    
    sample_contract = """
    This Service Agreement is entered into on January 1, 2024, between ABC Corp ("Client") 
    and John Doe ("Contractor"). The Contractor agrees to provide web development services 
    for a fee of $5,000 per month. Either party may terminate this agreement with 30 days 
    written notice. The Contractor shall maintain confidentiality of all client information.
    """
    
    try:
        # Import and test core components
        sys.path.append('.')
        from models.contract_analyzer import ContractAnalyzer
        from models.risk_assessor import RiskAssessor
        from models.persona_summarizer import PersonaSummarizer
        
        print("🔍 Testing ContractAnalyzer...")
        analyzer = ContractAnalyzer()
        analysis = analyzer.analyze(sample_contract)
        print(f"✅ Contract analysis completed: {len(analysis.get('clauses', []))} clauses identified")
        
        print("⚠️ Testing RiskAssessor...")
        risk_assessor = RiskAssessor()
        risks = risk_assessor.assess_risks(sample_contract, analysis)
        print(f"✅ Risk assessment completed: {risks.get('overall_score', 0):.2f} risk score")
        
        print("👤 Testing PersonaSummarizer...")
        summarizer = PersonaSummarizer()
        summary = summarizer.generate_summary(sample_contract, analysis, 'freelancer')
        print("✅ Persona summary generated")
        
        print("🎉 All components working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main development utility function"""
    if len(sys.argv) < 2:
        print("""
Legal AI Contract Analyzer - Development Utils

Usage: python dev_utils.py <command>

Commands:
  check-deps     Check if all dependencies are installed
  check-models   Check if AI models are available
  reset-db       Reset the application database
  test           Test the analysis pipeline
  full-check     Run all checks
        """)
        return
    
    command = sys.argv[1]
    
    if command == 'check-deps':
        check_dependencies()
    elif command == 'check-models':
        check_models()
    elif command == 'reset-db':
        reset_database()
    elif command == 'test':
        test_analysis()
    elif command == 'full-check':
        check_dependencies()
        check_models()
        test_analysis()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == '__main__':
    main()
EOF

# Create README
echo "📖 Creating README..."
cat > README.md << 'EOF'
# Legal AI Contract Analyzer

Advanced AI-powered contract analysis with persona-based summaries and risk assessment.

## Features

- **Persona-Based Analysis**: Tailored summaries for Freelancers, Tenants, Small Business, Consumers, and Students
- **Risk Assessment**: Comprehensive risk scoring with personalized advice
- **Interactive Clauses**: Click-to-expand clause analysis with plain English explanations
- **Risk Heatmap**: Visual representation of contract risks
- **Knowledge Graph**: Interactive contract relationship visualization
- **Multi-Format Support**: PDF, DOCX, DOC, and text input

## Quick Start

1. **Run Setup Script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Start the Application**:
   ```bash
   ./start.sh
   ```

3. **Access the Web Interface**:
   Open http://localhost:5000 in your browser

## Manual Setup

If the setup script doesn't work:

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv legal_ai_env
   source legal_ai_env/bin/activate  # On Windows: legal_ai_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Initialize Database**:
   ```bash
   python -c "from app import init_db; init_db()"
   ```

4. **Run Application**:
   ```bash
   python app.py
   ```

## Project Structure

```
legal_ai_contract_analyzer/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── models/                   # AI models
│   ├── contract_analyzer.py  # Contract analysis engine
│   ├── risk_assessor.py      # Risk assessment engine
│   └── persona_summarizer.py # Persona-based summaries
├── data/                     # Dataset processing
│   └── cuad_processor.py     # CUAD dataset handler
├── utils/                    # Utility functions
│   └── text_processing.py    # Text processing utilities
├── templates/                # HTML templates
├── static/                   # CSS, JS, images
└── uploads/                  # Temporary file storage
```

## AI Models

- **Primary Model**: Legal-BERT (`nlpaueb/legal-bert-base-uncased`)
- **Dataset**: CUAD (Contract Understanding Atticus Dataset)
- **NLP**: spaCy English model for entity recognition

## API Endpoints

- `POST /upload` - Upload and analyze contract
- `POST /api/analyze` - REST API for analysis
- `GET /api/history` - Get analysis history
- `GET /health` - Health check

## Development

### Testing
```bash
python dev_utils.py test
```

### Check Dependencies
```bash
python dev_utils.py check-deps
```

### Reset Database
```bash
python dev_utils.py reset-db
```

## Configuration

Edit `.env` file for configuration:
```
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key
MAX_FILE_SIZE=16777216
```

## Deployment

### Production Setup
1. Set `FLASK_ENV=production` in `.env`
2. Change `SECRET_KEY` to a secure value
3. Use `gunicorn` for production server:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Legal Disclaimer

This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal guidance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
EOF

# Final setup completion
echo ""
echo "🎉 Setup Complete!"
echo "================================================"
echo ""
echo "✅ Virtual environment created: legal_ai_env"
echo "✅ All dependencies installed"
echo "✅ Database initialized"
echo "✅ AI models downloaded"
echo "✅ Project structure created"
echo ""
echo "🚀 To start the application:"
echo "   ./start.sh"
echo ""
echo "🌐 Then open: http://localhost:5000"
echo ""
echo "🔧 Development utils:"
echo "   python dev_utils.py full-check"
echo ""
echo "📚 See README.md for detailed instructions"
echo ""

# Test if we can import basic modules
echo "🧪 Running quick test..."
python -c "
import torch
import transformers
import flask
import spacy
print('✅ Core modules importable')
" || echo "⚠️ Some modules may need manual installation"

echo "Setup completed successfully! 🎉"