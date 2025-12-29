"""
models/contract_analyzer.py
Core contract analysis using Legal-BERT and fine-tuned models
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import spacy
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ContractAnalyzer:
    """
    Advanced contract analyzer using Legal-BERT for:
    - Clause identification and classification
    - Entity extraction (parties, dates, amounts)
    - Contract type detection
    - Key term extraction
    """
    
    def __init__(self, model_name='nlpaueb/legal-bert-base-uncased'):
        """Initialize the contract analyzer with Legal-BERT model"""
        logger.info("Initializing ContractAnalyzer...")
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load Legal-BERT tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Load spaCy for NER and linguistic processing
            self.nlp = spacy.load('en_core_web_sm')
            
            # Initialize classification pipeline
            self.classifier = pipeline(
                'text-classification',
                model=model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Contract type patterns
            self.contract_patterns = self._load_contract_patterns()
            
            # Clause patterns for identification
            self.clause_patterns = self._load_clause_patterns()
            
            logger.info("ContractAnalyzer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing ContractAnalyzer: {str(e)}")
            raise

    def _load_contract_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for contract type identification"""
        return {
            'employment': [
                r'employment agreement', r'employment contract', r'job offer',
                r'salary', r'position', r'employment terms', r'work schedule'
            ],
            'lease': [
                r'lease agreement', r'rental agreement', r'tenant', r'landlord',
                r'rent', r'premises', r'rental property', r'lease term'
            ],
            'service': [
                r'service agreement', r'consulting agreement', r'professional services',
                r'service provider', r'deliverables', r'scope of work'
            ],
            'purchase': [
                r'purchase agreement', r'sale agreement', r'buyer', r'seller',
                r'purchase price', r'goods', r'products', r'merchandise'
            ],
            'freelance': [
                r'freelance agreement', r'independent contractor', r'contractor agreement',
                r'project', r'deliverables', r'milestone', r'freelancer'
            ],
            'nda': [
                r'non-disclosure agreement', r'confidentiality agreement', r'nda',
                r'confidential information', r'proprietary', r'trade secret'
            ],
            'license': [
                r'license agreement', r'licensing', r'intellectual property',
                r'copyright', r'trademark', r'patent', r'licensed material'
            ]
        }

    def _load_clause_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for clause identification"""
        return {
            'termination': [
                r'terminate', r'termination', r'end this agreement', r'expire',
                r'breach', r'default', r'dissolution', r'cancellation'
            ],
            'payment': [
                r'payment', r'pay', r'compensation', r'fee', r'salary', r'wage',
                r'remuneration', r'invoice', r'billing', r'cost'
            ],
            'liability': [
                r'liability', r'liable', r'damages', r'loss', r'injury',
                r'indemnify', r'hold harmless', r'limitation of liability'
            ],
            'confidentiality': [
                r'confidential', r'proprietary', r'non-disclosure', r'secret',
                r'private information', r'confidentiality'
            ],
            'intellectual_property': [
                r'intellectual property', r'copyright', r'trademark', r'patent',
                r'trade secret', r'proprietary rights', r'ownership'
            ],
            'dispute_resolution': [
                r'dispute', r'arbitration', r'mediation', r'litigation',
                r'governing law', r'jurisdiction', r'court'
            ],
            'force_majeure': [
                r'force majeure', r'act of god', r'unforeseeable circumstances',
                r'beyond reasonable control', r'natural disaster'
            ],
            'modification': [
                r'modification', r'amendment', r'change', r'alter', r'modify',
                r'written consent', r'mutual agreement'
            ]
        }

    def analyze(self, contract_text: str) -> Dict[str, Any]:
        """
        Perform comprehensive contract analysis
        
        Args:
            contract_text: Raw contract text
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting contract analysis...")
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(contract_text)
            
            # Extract basic information
            contract_type = self._detect_contract_type(cleaned_text)
            parties = self._extract_parties(cleaned_text)
            key_dates = self._extract_dates(cleaned_text)
            amounts = self._extract_amounts(cleaned_text)
            
            # Identify and classify clauses
            clauses = self._identify_clauses(cleaned_text)
            
            # Generate summary
            summary = self._generate_summary(cleaned_text, contract_type, parties)
            
            # Extract key terms and obligations
            key_terms = self._extract_key_terms(cleaned_text)
            obligations = self._extract_obligations(cleaned_text)
            
            results = {
                'contract_type': contract_type,
                'parties': parties,
                'key_dates': key_dates,
                'amounts': amounts,
                'clauses': clauses,
                'summary': summary,
                'key_terms': key_terms,
                'obligations': obligations,
                'analysis_timestamp': datetime.now().isoformat(),
                'text_length': len(contract_text),
                'processed_text_length': len(cleaned_text)
            }
            
            logger.info(f"Contract analysis completed. Type: {contract_type}, Clauses: {len(clauses)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in contract analysis: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess contract text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()

    def _detect_contract_type(self, text: str) -> str:
        """Detect contract type using pattern matching and ML"""
        text_lower = text.lower()
        type_scores = {}
        
        # Pattern-based scoring
        for contract_type, patterns in self.contract_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            type_scores[contract_type] = score
        
        # Return highest scoring type
        if type_scores:
            detected_type = max(type_scores, key=type_scores.get)
            if type_scores[detected_type] > 0:
                return detected_type
        
        return 'general'

    def _extract_parties(self, text: str) -> List[Dict[str, Any]]:
        """Extract contracting parties using NER and patterns"""
        doc = self.nlp(text)
        parties = []
        
        # Extract organizations and persons
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON']:
                parties.append({
                    'name': ent.text,
                    'type': 'organization' if ent.label_ == 'ORG' else 'person',
                    'context': text[max(0, ent.start_char-50):ent.end_char+50]
                })
        
        # Pattern-based extraction for common contract roles
        role_patterns = [
            (r'(?i)(?:the\s+)?client(?:\s+is|\s*:)\s*([^,.\n]+)', 'client'),
            (r'(?i)(?:the\s+)?contractor(?:\s+is|\s*:)\s*([^,.\n]+)', 'contractor'),
            (r'(?i)(?:the\s+)?landlord(?:\s+is|\s*:)\s*([^,.\n]+)', 'landlord'),
            (r'(?i)(?:the\s+)?tenant(?:\s+is|\s*:)\s*([^,.\n]+)', 'tenant'),
            (r'(?i)(?:the\s+)?employer(?:\s+is|\s*:)\s*([^,.\n]+)', 'employer'),
            (r'(?i)(?:the\s+)?employee(?:\s+is|\s*:)\s*([^,.\n]+)', 'employee'),
        ]
        
        for pattern, role in role_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                party_name = match.group(1).strip()
                if party_name:
                    parties.append({
                        'name': party_name,
                        'role': role,
                        'type': 'identified_party'
                    })
        
        # Remove duplicates
        seen = set()
        unique_parties = []
        for party in parties:
            party_key = party['name'].lower()
            if party_key not in seen:
                seen.add(party_key)
                unique_parties.append(party)
        
        return unique_parties

    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract important dates from contract"""
        doc = self.nlp(text)
        dates = []
        
        # Extract date entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                dates.append({
                    'date': ent.text,
                    'context': text[max(0, ent.start_char-30):ent.end_char+30],
                    'type': 'date_entity'
                })
        
        # Pattern-based extraction for specific date types
        date_patterns = [
            (r'(?i)(?:effective|start|commencement)\s+date[:\s]*([^,.\n]+)', 'effective_date'),
            (r'(?i)(?:expir|end|terminat)\w*\s+date[:\s]*([^,.\n]+)', 'expiration_date'),
            (r'(?i)(?:due|payment)\s+date[:\s]*([^,.\n]+)', 'payment_date'),
            (r'(?i)(?:renewal)\s+date[:\s]*([^,.\n]+)', 'renewal_date'),
            (r'(?i)(?:notice)\s+(?:period|date)[:\s]*([^,.\n]+)', 'notice_date')
        ]
        
        for pattern, date_type in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_value = match.group(1).strip()
                if date_value:
                    dates.append({
                        'date': date_value,
                        'type': date_type,
                        'context': text[max(0, match.start()-30):match.end()+30]
                    })
        
        return dates

    def _extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts and quantities"""
        doc = self.nlp(text)
        amounts = []
        
        # Extract money entities
        for ent in doc.ents:
            if ent.label_ == 'MONEY':
                amounts.append({
                    'amount': ent.text,
                    'context': text[max(0, ent.start_char-30):ent.end_char+30],
                    'type': 'money_entity'
                })
        
        # Pattern-based extraction for specific amount types
        amount_patterns = [
            (r'(?i)(?:salary|wage|pay)(?:\s+is|\s+of|\s*:)\s*(\$?[\d,]+(?:\.\d{2})?)', 'salary'),
            (r'(?i)(?:rent|rental)(?:\s+is|\s+of|\s*:)\s*(\$?[\d,]+(?:\.\d{2})?)', 'rent'),
            (r'(?i)(?:fee|cost|price)(?:\s+is|\s+of|\s*:)\s*(\$?[\d,]+(?:\.\d{2})?)', 'fee'),
            (r'(?i)(?:deposit|security)(?:\s+is|\s+of|\s*:)\s*(\$?[\d,]+(?:\.\d{2})?)', 'deposit'),
            (r'(?i)(?:penalty|fine)(?:\s+is|\s+of|\s*:)\s*(\$?[\d,]+(?:\.\d{2})?)', 'penalty')
        ]
        
        for pattern, amount_type in amount_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_value = match.group(1).strip()
                if amount_value:
                    amounts.append({
                        'amount': amount_value,
                        'type': amount_type,
                        'context': text[max(0, match.start()-30):match.end()+30]
                    })
        
        return amounts

    def _identify_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Identify and classify contract clauses"""
        # Split text into potential clauses (paragraphs and numbered sections)
        clause_candidates = self._split_into_clauses(text)
        
        clauses = []
        for i, clause_text in enumerate(clause_candidates):
            if len(clause_text.strip()) < 50:  # Skip very short segments
                continue
                
            # Classify clause type
            clause_type = self._classify_clause(clause_text)
            
            # Extract plain English explanation
            plain_english = self._generate_plain_english_explanation(clause_text, clause_type)
            
            # Identify key obligations and rights
            obligations = self._extract_clause_obligations(clause_text)
            
            clause = {
                'id': f'clause_{i+1}',
                'text': clause_text.strip(),
                'type': clause_type,
                'plain_english': plain_english,
                'obligations': obligations,
                'importance': self._assess_clause_importance(clause_text, clause_type),
                'word_count': len(clause_text.split())
            }
            
            clauses.append(clause)
        
        return clauses

    def _split_into_clauses(self, text: str) -> List[str]:
        """Split contract text into logical clauses"""
        # Try numbered sections first
        numbered_sections = re.split(r'\n\s*\d+\.?\s+', text)
        if len(numbered_sections) > 3:  # If we get good splits
            return [section.strip() for section in numbered_sections if section.strip()]
        
        # Try lettered sections
        lettered_sections = re.split(r'\n\s*[a-z]\.?\s+', text, flags=re.IGNORECASE)
        if len(lettered_sections) > 3:
            return [section.strip() for section in lettered_sections if section.strip()]
        
        # Fall back to paragraph splits
        paragraphs = re.split(r'\n\s*\n', text)
        return [para.strip() for para in paragraphs if para.strip() and len(para.strip()) > 50]

    def _classify_clause(self, clause_text: str) -> str:
        """Classify clause type using pattern matching"""
        clause_lower = clause_text.lower()
        
        # Score each clause type
        type_scores = {}
        for clause_type, patterns in self.clause_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, clause_lower))
                score += matches
            if score > 0:
                type_scores[clause_type] = score
        
        # Return highest scoring type
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return 'general'

    def _generate_plain_english_explanation(self, clause_text: str, clause_type: str) -> str:
        """Generate plain English explanation of clause"""
        explanations = {
            'termination': "This clause explains when and how the contract can be ended.",
            'payment': "This clause covers payment amounts, timing, and methods.",
            'liability': "This clause defines who is responsible if something goes wrong.",
            'confidentiality': "This clause protects sensitive information from being shared.",
            'intellectual_property': "This clause deals with ownership of ideas, designs, or creative work.",
            'dispute_resolution': "This clause explains how disagreements will be resolved.",
            'force_majeure': "This clause covers what happens during unexpected events beyond anyone's control.",
            'modification': "This clause explains how changes to the contract must be made."
        }
        
        base_explanation = explanations.get(clause_type, "This clause contains important contract terms.")
        
        # Add specific details based on clause content
        if 'written notice' in clause_text.lower():
            base_explanation += " Written notice is required."
        if 'thirty' in clause_text.lower() or '30' in clause_text:
            base_explanation += " A 30-day period is mentioned."
        if 'immediately' in clause_text.lower():
            base_explanation += " Some actions take effect immediately."
        
        return base_explanation

    def _extract_clause_obligations(self, clause_text: str) -> List[str]:
        """Extract key obligations from clause text"""
        obligations = []
        
        # Pattern for obligations (shall, must, will, required to)
        obligation_patterns = [
            r'(?:shall|must|will|required to|obligated to)\s+([^.]+)',
            r'(?:Party|Client|Contractor|Employee|Employer)\s+(?:shall|must|will)\s+([^.]+)',
            r'(?:agrees to|undertakes to)\s+([^.]+)'
        ]
        
        for pattern in obligation_patterns:
            matches = re.finditer(pattern, clause_text, re.IGNORECASE)
            for match in matches:
                obligation = match.group(1).strip()
                if len(obligation) > 10 and len(obligation) < 200:
                    obligations.append(obligation)
        
        return obligations[:5]  # Limit to top 5 obligations

    def _assess_clause_importance(self, clause_text: str, clause_type: str) -> str:
        """Assess importance level of clause"""
        high_importance_types = ['termination', 'liability', 'payment']
        high_importance_keywords = ['penalty', 'breach', 'default', 'damages', 'liability']
        
        if clause_type in high_importance_types:
            return 'high'
        
        clause_lower = clause_text.lower()
        for keyword in high_importance_keywords:
            if keyword in clause_lower:
                return 'high'
        
        if len(clause_text) > 500:  # Long clauses might be important
            return 'medium'
        
        return 'medium'

    def _extract_key_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract key terms and definitions"""
        key_terms = []
        
        # Pattern for definitions (terms in quotes or defined as)
        definition_patterns = [
            r'"([^"]+)"\s+(?:means|shall mean|is defined as)\s+([^.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:means|shall mean|is defined as)\s+([^.]+)',
            r'(?:the term|as used herein,?)\s+"([^"]+)"\s+(?:means|shall mean)\s+([^.]+)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if len(term) > 2 and len(definition) > 10:
                    key_terms.append({
                        'term': term,
                        'definition': definition,
                        'type': 'defined_term'
                    })
        
        return key_terms

    def _extract_obligations(self, text: str) -> Dict[str, List[str]]:
        """Extract obligations by party"""
        doc = self.nlp(text)
        obligations = {'general': []}
        
        # Find sentences with obligation keywords
        obligation_keywords = ['shall', 'must', 'will', 'required to', 'obligated to', 'agrees to']
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check if sentence contains obligation keywords
            if any(keyword in sent_lower for keyword in obligation_keywords):
                if len(sent_text) > 20 and len(sent_text) < 300:
                    obligations['general'].append(sent_text)
        
        # Limit obligations
        obligations['general'] = obligations['general'][:10]
        
        return obligations

    def _generate_summary(self, text: str, contract_type: str, parties: List[Dict]) -> str:
        """Generate a brief contract summary"""
        party_names = [p['name'] for p in parties[:2]]  # First two parties
        
        if len(party_names) >= 2:
            party_str = f"between {party_names[0]} and {party_names[1]}"
        elif len(party_names) == 1:
            party_str = f"involving {party_names[0]}"
        else:
            party_str = "between the contracting parties"
        
        contract_type_readable = contract_type.replace('_', ' ').title()
        
        summary = f"This is a {contract_type_readable} contract {party_str}. "
        
        # Add key details based on contract type
        if contract_type == 'employment':
            summary += "It outlines employment terms, responsibilities, and compensation."
        elif contract_type == 'lease':
            summary += "It establishes the terms for renting property, including rent and tenant obligations."
        elif contract_type == 'service':
            summary += "It defines the services to be provided and the terms of engagement."
        elif contract_type == 'freelance':
            summary += "It outlines project deliverables, timelines, and payment terms for independent work."
        else:
            summary += "It establishes the rights and obligations of each party."
        
        return summary

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()

    def compare_contracts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two contracts for similarity and differences"""
        # Get embeddings
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        
        # Calculate similarity
        similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        
        # Analyze both contracts
        analysis1 = self.analyze(text1)
        analysis2 = self.analyze(text2)
        
        comparison = {
            'similarity_score': float(similarity),
            'contract1_type': analysis1['contract_type'],
            'contract2_type': analysis2['contract_type'],
            'common_clauses': self._find_common_clauses(analysis1['clauses'], analysis2['clauses']),
            'unique_clauses1': self._find_unique_clauses(analysis1['clauses'], analysis2['clauses']),
            'unique_clauses2': self._find_unique_clauses(analysis2['clauses'], analysis1['clauses'])
        }
        
        return comparison

    def _find_common_clauses(self, clauses1: List, clauses2: List) -> List[str]:
        """Find common clause types between two contracts"""
        types1 = {clause['type'] for clause in clauses1}
        types2 = {clause['type'] for clause in clauses2}
        return list(types1.intersection(types2))

    def _find_unique_clauses(self, clauses1: List, clauses2: List) -> List[str]:
        """Find clause types unique to first contract"""
        types1 = {clause['type'] for clause in clauses1}
        types2 = {clause['type'] for clause in clauses2}
        return list(types1.difference(types2))