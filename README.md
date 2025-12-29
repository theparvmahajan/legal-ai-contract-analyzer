# 🏛️ Legal AI Contract Analyzer

<div align="center">

![Legal AI Banner](https://img.shields.io/badge/Legal_AI-Contract_Analyzer-667eea?style=for-the-badge&logo=scale&logoColor=white)

**AI-Powered Contract Analysis with Persona-Based Insights and Risk Assessment**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.35.0-FFD21E?style=flat)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#️-technology-stack)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Quick Install](#quick-install-automated)
  - [Manual Installation](#manual-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Models & Dataset](#-models--dataset)
- [Project Structure](#-project-structure)

---

## 🌟 Overview

**Legal AI Contract Analyzer** is an intelligent contract analysis platform that leverages state-of-the-art Natural Language Processing (NLP) to help individuals and businesses understand complex legal contracts. The system provides personalized insights based on user roles and generates comprehensive risk assessments with actionable recommendations.

### 🎯 Problem Statement

Legal contracts are often:
- ❌ Complex and filled with legal jargon
- ❌ Difficult for non-lawyers to understand
- ❌ Contain hidden risks and unfavorable terms
- ❌ Time-consuming to review thoroughly
- ❌ Expensive to have professionally analyzed

### 💡 Our Solution

This AI-powered tool:
- ✅ Translates legal language into plain English
- ✅ Identifies and highlights potential risks
- ✅ Provides persona-specific analysis and advice
- ✅ Visualizes contract structure and relationships
- ✅ Offers actionable recommendations
- ✅ Delivers instant analysis at no cost

## ✨ Key Features

### 🎭 Persona-Based Analysis

Choose your role for tailored insights:

| Persona | Focus Areas | Key Concerns |
|---------|-------------|--------------|
| 👨‍💻 **Freelancer** | Payment terms, IP ownership, scope | Cash flow, deliverables, timeline |
| 🏠 **Tenant** | Rent, maintenance, deposits | Lease duration, responsibilities |
| 🏢 **Small Business** | Liability, compliance, operations | Risk exposure, restrictions |
| 👤 **Consumer** | Fees, cancellation, warranties | Hidden costs, refund rights |
| 🎓 **Student** | Financial obligations, flexibility | Affordability, withdrawal terms |

### ⚠️ Intelligent Risk Assessment

- **Clause-Level Scoring**: Each clause rated on 0-100% risk scale
- **Context-Aware Analysis**: Personalized based on your situation
- **Risk Heatmap**: Visual representation of high-risk areas
- **Priority Ranking**: Focus on most critical issues first
- **Comparative Analysis**: Benchmarking against standard contracts

### 🔍 Advanced Contract Analysis

- **Entity Recognition**: Automatically identifies parties, dates, amounts
- **Clause Classification**: 41+ legal clause categories (CUAD dataset)
- **Plain English Translation**: Simplifies complex legal terminology
- **Key Term Extraction**: Highlights critical definitions and obligations
- **Timeline Visualization**: Maps important dates and deadlines

### 📊 Interactive Visualizations

- **Knowledge Graph**: Interactive network of contract relationships
- **Risk Heatmap**: Color-coded clause-level risk indicators
- **Timeline View**: Visual representation of key dates
- **Statistical Charts**: Distribution of clause types and risk levels

### 🚀 Additional Features

- Multi-format support (PDF, DOCX, DOC, TXT)
- Drag-and-drop file upload
- Real-time analysis progress
- Exportable reports (PDF, JSON)
- Analysis history tracking
- Mobile-responsive design
- Dark mode support

---

## 🛠️ Technology Stack

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core language |
| **Flask** | 2.3.2 | Web framework |
| **PyTorch** | 2.1.0 | Deep learning framework |
| **Transformers** | 4.35.0 | NLP model library |
| **spaCy** | 3.7.0 | Advanced NLP processing |
| **SQLite** | 3.x | Database |

### AI/ML Models

| Model | Size | Purpose |
|-------|------|---------|
| **Legal-BERT** | 110M params | Contract understanding |
| **spaCy en_core_web_sm** | 12MB | Entity recognition |
| **Custom Fine-tuned Models** | Various | Clause classification |

### Frontend

| Technology | Purpose |
|-----------|---------|
| **HTML5/CSS3** | Structure and styling |
| **JavaScript (ES6+)** | Interactivity |
| **D3.js** | Knowledge graph visualization |
| **Plotly.js** | Charts and graphs |
| **Font Awesome** | Icons |

### Dataset

- **CUAD (Contract Understanding Atticus Dataset)**
  - 500+ real contracts
  - 13,000+ expert annotations
  - 41 legal categories
  - Source: [Atticus Project](https://www.atticusprojectai.org/)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (HTML5 + CSS3 + JavaScript + D3.js + Plotly.js)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask Application                        │
│                      (app.py)                               │
└──┬───────────────┬────────────────┬────────────────────┬────┘
   │               │                │                    │
   ▼               ▼                ▼                    ▼
┌─────────┐  ┌──────────┐   ┌─────────────┐   ┌──────────────┐
│Contract │  │   Risk   │   │   Persona   │   │     Text     │
│Analyzer │  │ Assessor │   │ Summarizer  │   │  Processor   │
└────┬────┘  └────┬─────┘   └──────┬──────┘   └──────┬───────┘
     │            │                 │                  │
     └────────────┴─────────────────┴──────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      Legal-BERT Model       │
              │   (nlpaueb/legal-bert)      │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │       SQLite Database        │
              │  (contract_analyses table)   │
              └─────────────────────────────┘
```

### Data Flow

```
1. User uploads contract (PDF/DOCX/TXT)
        ↓
2. Text extraction (PyPDF2/python-docx)
        ↓
3. Preprocessing & cleaning
        ↓
4. Legal-BERT analysis
        ↓
5. Clause identification & classification
        ↓
6. Risk scoring (context-aware)
        ↓
7. Persona-based summarization
        ↓
8. Visualization generation
        ↓
9. Results displayed + Database storage
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Git**: Version control
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 5GB free space
- **OS**: Linux, macOS, or Windows

### Quick Install (Automated)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/legal-ai-contract-analyzer.git
cd legal-ai-contract-analyzer

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Start the application
./start.sh
```

### Manual Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/legal-ai-contract-analyzer.git
cd legal-ai-contract-analyzer
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv legal_ai_env

# Activate it
# On Linux/Mac:
source legal_ai_env/bin/activate

# On Windows:
# legal_ai_env\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Step 4: Initialize Database

```bash
python -c "
import sqlite3
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
conn.commit()
conn.close()
print('Database initialized!')
"
```

#### Step 5: Download AI Models

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading Legal-BERT model...')
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
print('Models downloaded successfully!')
"
```

#### Step 6: Verify Installation

```bash
# Run tests
python dev_utils.py full-check

# Expected output: All checks should pass 
```

---

## 🚀 Quick Start

### Basic Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

3. **Select your persona:**
   - Choose your role (Freelancer, Tenant, etc.)

4. **Upload a contract:**
   - Drag & drop a file, or
   - Paste contract text directly

5. **Add context (optional):**
   - Describe your situation for personalized advice

6. **Analyze:**
   - Click "Analyze Contract"
   - Wait 30-60 seconds for results

7. **Review results:**
   - Risk score and breakdown
   - Persona-specific summary
   - Interactive clause explorer
   - Knowledge graph
   - Actionable recommendations

---

### Understanding Results

#### Risk Score

| Score | Level | Meaning |
|-------|-------|---------|
| 0-39% | 🟢 Low | Generally favorable terms |
| 40-69% | 🟡 Medium | Some areas need attention |
| 70-100% | 🔴 High | Significant risks identified |

#### Clause Analysis

Each clause includes:
- **Type**: payment, termination, liability, etc.
- **Risk Level**: High/Medium/Low
- **Plain English**: Simplified explanation
- **Concerns**: Specific issues identified
- **Suggestions**: Recommended actions

#### Interactive Elements

1. **Click Clauses**: Expand for detailed analysis
2. **Hover Heatmap**: See risk scores
3. **Explore Graph**: Drag nodes, zoom in/out
4. **Export Report**: Download as PDF

---

## 🔌 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### 1. Analyze Contract

**POST** `/upload`

Upload and analyze a contract.

**Request:**
```http
POST /upload HTTP/1.1
Content-Type: multipart/form-data

contract_file: <file>
persona: freelancer
user_context: "Risk-averse, tight budget"
```

**Response:**
```json
{
  "filename": "contract.pdf",
  "persona": "freelancer",
  "timestamp": "2024-12-30T10:30:00",
  "overall_risk_score": 0.65,
  "risk_level": "medium",
  "persona_summary": "As a freelancer, here's what this contract means...",
  "key_parties": [
    {"name": "ABC Corp", "role": "client"},
    {"name": "John Doe", "role": "contractor"}
  ],
  "clauses": [...],
  "risk_breakdown": {...},
  "personalized_advice": [...]
}
```

#### 2. REST API Analysis

**POST** `/api/analyze`

Analyze contract via REST API.

**Request:**
```json
{
  "contract_text": "This Agreement...",
  "persona": "tenant",
  "user_context": "First time renter"
}
```

**Response:** Same as `/upload`

#### 3. Get Analysis History

**GET** `/api/history`

Retrieve past analyses.

**Response:**
```json
[
  {
    "id": 1,
    "filename": "contract1.pdf",
    "upload_time": "2024-12-30T10:00:00",
    "persona": "freelancer",
    "risk_score": 0.45
  },
  ...
]
```

#### 4. Health Check

**GET** `/health`

Check system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-30T10:30:00",
  "models_loaded": true
}
```

### Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 413 | File Too Large |
| 500 | Internal Server Error |

---

## 🤖 Models & Dataset

### Primary Model: Legal-BERT

**Source:** [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased)

**Specifications:**
- **Architecture**: BERT-base
- **Parameters**: 110M
- **Training Data**: Legal documents (contracts, court cases, legislation)
- **Vocabulary**: 30K tokens
- **Max Sequence Length**: 512 tokens
- **Use Case**: Legal text understanding and classification

**Why Legal-BERT?**
- ✅ Pre-trained on legal corpus
- ✅ Understands legal terminology
- ✅ Superior performance on legal tasks
- ✅ Lightweight (110M vs. GPT-3's 175B)
- ✅ Runs on CPU efficiently

### Dataset: CUAD

**Source:** [Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad)

**Statistics:**
- **Contracts**: 510
- **Annotations**: 13,000+
- **Categories**: 41 legal clause types
- **Format**: Question-answering pairs
- **Domain**: Commercial contracts

**Clause Categories:**
```
├── Agreement Date          ├── Parties
├── Competitive Restriction ├── Expiration Date
├── Governing Law          ├── Non-Compete
├── Exclusivity            ├── Termination Rights
├── IP Ownership           ├── License Grant
├── Liability Cap          ├── Payment Terms
├── Warranty Duration      ├── Insurance
└── ... (30+ more)
```

### NLP Processing Pipeline

```
Input Text
    ↓
[Tokenization] → BERT Tokenizer
    ↓
[Encoding] → Legal-BERT Embeddings
    ↓
[Classification] → Clause Type Identification
    ↓
[NER] → spaCy Entity Recognition
    ↓
[Risk Scoring] → Custom Algorithm
    ↓
Output Analysis
```

---

## 📂 Project Structure

```
legal_ai_contract_analyzer/
│
├── 📄 app.py                       # Main Flask application
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.sh                     # Automated setup script
├── 📄 start.sh                     # Application start script
├── 📄 dev_utils.py                 # Development utilities
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .env                         # Environment variables
│
├── 📁 models/                      # AI/ML Models
│   ├── __init__.py
│   ├── contract_analyzer.py        # Legal-BERT contract analysis
│   ├── risk_assessor.py            # Risk scoring engine
│   └── persona_summarizer.py       # Persona-based summaries
│
├── 📁 data/                        # Data processing
│   ├── __init__.py
│   ├── cuad_processor.py           # CUAD dataset handler
│   └── contract_samples.py         # Sample contracts
│
├── 📁 utils/                       # Utility functions
│   ├── __init__.py
│   ├── text_processing.py          # Text extraction & cleaning
│   └── visualization.py            # Chart/graph generation
│
├── 📁 templates/                   # HTML templates
│   └── index.html                  # Main web interface
│
├── 📁 static/                      # Static assets
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── main.js
│   │   └── visualizations.js
│   └── images/
│       └── logo.png
│
├── 📁 uploads/                     # Temporary file storage
│   └── .gitkeep
│
├── 📁 tests/                       # Unit tests
│   ├── test_analyzer.py
│   ├── test_risk_assessor.py
│   └── test_api.py
│
├── 📁 docs/                        # Documentation
│   ├── API.md
│   ├── MODELS.md
│   ├── CONTRIBUTING.md
│   └── demo.gif
│
└── 📁 scripts/                     # Utility scripts
    ├── download_models.py
    ├── process_dataset.py
    └── run_tests.sh
```
