"""
utils/text_processing.py
Text processing utilities for contract analysis
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional
import docx
import PyPDF2
import spacy
from io import BytesIO

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processing utilities for contract documents
    - Extract text from various file formats
    - Clean and normalize text
    - Split into sections
    - Extract entities and key information
    """
    
    def __init__(self):
        """Initialize text processor"""
        logger.info("Initializing TextProcessor...")
        
        try:
            # Load spaCy model for NLP processing
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("TextProcessor initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing TextProcessor: {str(e)}")
            raise

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various file formats
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting from PDF: {str(e)}")
            raise

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {str(e)}")
            raise

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin1') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting from TXT: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize contract text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common footer patterns
        text = re.sub(r'Confidential and Proprietary.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'DRAFT.*$', '', text, flags=re.MULTILINE)
        
        # Normalize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Normalize section numbering
        text = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2.\3', text)
        
        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary of entity types and their instances
        """
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'MONEY': [],
            'DATE': [],
            'GPE': [],  # Countries, cities, states
            'PERCENT': [],
            'CARDINAL': []  # Numbers
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'context': text[max(0, ent.start_char-30):ent.end_char+30]
                })
        
        return entities

    def extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[Dict[str, Any]]:
        """
        Extract key phrases from contract text
        
        Args:
            text: Text to process
            max_phrases: Maximum number of phrases to return
            
        Returns:
            List of key phrases with metadata
        """
        doc = self.nlp(text)
        key_phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 5 and len(chunk.text) < 100:  # Reasonable length
                key_phrases.append({
                    'text': chunk.text,
                    'type': 'noun_phrase',
                    'start': chunk.start_char,
                    'end': chunk.end_char,
                    'root': chunk.root.text,
                    'root_pos': chunk.root.pos_
                })
        
        # Extract legal terms (pattern-based)
        legal_patterns = [
            r'(?i)(?:shall|must|will|required to|obligated to)\s+[^.]{10,100}',
            r'(?i)(?:party|parties|client|contractor|employee|employer)\s+[^.]{10,100}',
            r'(?i)(?:agreement|contract|terms|conditions)\s+[^.]{10,100}',
            r'(?i)(?:payment|compensation|fee|salary)\s+[^.]{10,100}',
            r'(?i)(?:termination|breach|default|violation)\s+[^.]{10,100}'
        ]
        
        for pattern in legal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key_phrases.append({
                    'text': match.group(0),
                    'type': 'legal_phrase',
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                })
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            phrase_key = phrase['text'].lower().strip()
            if phrase_key not in seen and len(phrase_key) > 5:
                seen.add(phrase_key)
                unique_phrases.append(phrase)
        
        # Score phrases by legal relevance
        legal_keywords = [
            'agreement', 'contract', 'party', 'shall', 'payment', 'termination',
            'liability', 'obligation', 'breach', 'default', 'confidential',
            'intellectual property', 'indemnify', 'arbitration', 'governing law'
        ]
        
        for phrase in unique_phrases:
            score = 0
            phrase_lower = phrase['text'].lower()
            for keyword in legal_keywords:
                if keyword in phrase_lower:
                    score += 1
            phrase['relevance_score'] = score
        
        # Sort by relevance and return top phrases
        unique_phrases.sort(key=lambda x: x['relevance_score'], reverse=True)
        return unique_phrases[:max_phrases]

    def split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split contract text into logical sections
        
        Args:
            text: Contract text to split
            
        Returns:
            List of sections with metadata
        """
        sections = []
        
        # Try different splitting strategies
        
        # Strategy 1: Numbered sections (1., 2., 3., etc.)
        numbered_pattern = r'\n\s*(\d+\.?\s+[^\n]+)'
        numbered_matches = list(re.finditer(numbered_pattern, text))
        
        if len(numbered_matches) >= 3:
            sections = self._split_by_numbered_sections(text, numbered_matches)
        else:
            # Strategy 2: Lettered sections (A., B., C., etc.)
            lettered_pattern = r'\n\s*([A-Z]\.?\s+[^\n]+)'
            lettered_matches = list(re.finditer(lettered_pattern, text))
            
            if len(lettered_matches) >= 3:
                sections = self._split_by_lettered_sections(text, lettered_matches)
            else:
                # Strategy 3: Header-based sections
                header_pattern = r'\n\s*([A-Z][A-Z\s]+[A-Z])\s*\n'
                header_matches = list(re.finditer(header_pattern, text))
                
                if len(header_matches) >= 2:
                    sections = self._split_by_headers(text, header_matches)
                else:
                    # Strategy 4: Paragraph-based sections
                    sections = self._split_by_paragraphs(text)
        
        return sections

    def _split_by_numbered_sections(self, text: str, matches: List) -> List[Dict[str, Any]]:
        """Split text by numbered sections"""
        sections = []
        
        for i, match in enumerate(matches):
            section_start = match.start()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[section_start:section_end].strip()
            section_title = match.group(1).strip()
            
            sections.append({
                'id': f'section_{i+1}',
                'title': section_title,
                'text': section_text,
                'type': 'numbered_section',
                'start': section_start,
                'end': section_end,
                'word_count': len(section_text.split())
            })
        
        return sections

    def _split_by_lettered_sections(self, text: str, matches: List) -> List[Dict[str, Any]]:
        """Split text by lettered sections"""
        sections = []
        
        for i, match in enumerate(matches):
            section_start = match.start()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[section_start:section_end].strip()
            section_title = match.group(1).strip()
            
            sections.append({
                'id': f'section_{chr(65+i)}',
                'title': section_title,
                'text': section_text,
                'type': 'lettered_section',
                'start': section_start,
                'end': section_end,
                'word_count': len(section_text.split())
            })
        
        return sections

    def _split_by_headers(self, text: str, matches: List) -> List[Dict[str, Any]]:
        """Split text by header sections"""
        sections = []
        
        for i, match in enumerate(matches):
            section_start = match.start()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[section_start:section_end].strip()
            section_title = match.group(1).strip()
            
            sections.append({
                'id': f'header_section_{i+1}',
                'title': section_title,
                'text': section_text,
                'type': 'header_section',
                'start': section_start,
                'end': section_end,
                'word_count': len(section_text.split())
            })
        
        return sections

    def _split_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split text by paragraphs as fallback"""
        paragraphs = re.split(r'\n\s*\n', text)
        sections = []
        
        current_pos = 0
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 100:  # Only include substantial paragraphs
                para_start = text.find(para, current_pos)
                para_end = para_start + len(para)
                
                # Create title from first few words
                words = para.split()[:8]
                title = ' '.join(words) + ('...' if len(words) == 8 else '')
                
                sections.append({
                    'id': f'paragraph_{i+1}',
                    'title': title,
                    'text': para.strip(),
                    'type': 'paragraph',
                    'start': para_start,
                    'end': para_end,
                    'word_count': len(para.split())
                })
                
                current_pos = para_end
        
        return sections

    def extract_definitions(self, text: str) -> Dict[str, str]:
        """
        Extract defined terms from contract
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary of term definitions
        """
        definitions = {}
        
        # Pattern 1: "Term" means definition
        pattern1 = r'"([^"]+)"\s+(?:means|shall mean|is defined as)\s+([^.]+\.)'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        
        for match in matches1:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            definitions[term] = definition
        
        # Pattern 2: Term (defined term) means definition
        pattern2 = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\([^)]+\)\s+(?:means|shall mean)\s+([^.]+\.)'
        matches2 = re.finditer(pattern2, text)
        
        for match in matches2:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            definitions[term] = definition
        
        # Pattern 3: As used herein, "term" means definition
        pattern3 = r'(?:as used herein|for purposes of this agreement),?\s*"([^"]+)"\s+(?:means|shall mean)\s+([^.]+\.)'
        matches3 = re.finditer(pattern3, text, re.IGNORECASE)
        
        for match in matches3:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            definitions[term] = definition
        
        return definitions

    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """
        Extract contact information from contract
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary of contact information types
        """
        contact_info = {
            'emails': [],
            'phones': [],
            'addresses': [],
            'websites': []
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['emails'] = list(set(emails))
        
        # Phone pattern (various formats)
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'  # International
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            contact_info['phones'].extend(phones)
        
        contact_info['phones'] = list(set(contact_info['phones']))
        
        # Website pattern
        website_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        websites = re.findall(website_pattern, text)
        contact_info['websites'] = list(set(websites))
        
        # Address pattern (simplified)
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)[^,\n]*(?:,\s*[A-Za-z\s]+)?(?:,\s*[A-Z]{2})?\s*\d{5}(?:-\d{4})?'
        addresses = re.findall(address_pattern, text)
        contact_info['addresses'] = list(set(addresses))
        
        return contact_info

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the contract
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary of readability scores
        """
        if not text:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        # Basic text statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        syllables = self._count_syllables(' '.join(words))
        
        if len(sentences) == 0 or len(words) == 0:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        # Flesch Reading Ease Score
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),  # Clamp between 0-100
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'readability_level': self._interpret_flesch_score(flesch_score)
        }

    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting"""
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in text.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if text.lower().endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(1, syllable_count)

    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch reading ease score"""
        if score >= 90:
            return 'Very Easy'
        elif score >= 80:
            return 'Easy'
        elif score >= 70:
            return 'Fairly Easy'
        elif score >= 60:
            return 'Standard'
        elif score >= 50:
            return 'Fairly Difficult'
        elif score >= 30:
            return 'Difficult'
        else:
            return 'Very Difficult'

    def extract_monetary_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all monetary terms and amounts
        
        Args:
            text: Contract text
            
        Returns:
            List of monetary terms with context
        """
        monetary_terms = []
        
        # Currency patterns
        currency_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # Dollar amounts
            r'USD\s*[\d,]+(?:\.\d{2})?',  # USD amounts
            r'€[\d,]+(?:\.\d{2})?',  # Euro amounts
            r'£[\d,]+(?:\.\d{2})?',  # Pound amounts
            r'[\d,]+(?:\.\d{2})?\s*(?:dollars|USD|euros|pounds)'  # Written amounts
        ]
        
        for pattern in currency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get context around the amount
                context_start = max(0, start_pos - 50)
                context_end = min(len(text), end_pos + 50)
                context = text[context_start:context_end]
                
                # Classify the type of monetary term
                term_type = self._classify_monetary_term(context)
                
                monetary_terms.append({
                    'amount': amount,
                    'type': term_type,
                    'context': context.strip(),
                    'start': start_pos,
                    'end': end_pos
                })
        
        return monetary_terms

    def _classify_monetary_term(self, context: str) -> str:
        """Classify the type of monetary term based on context"""
        context_lower = context.lower()
        
        if any(term in context_lower for term in ['salary', 'wage', 'compensation', 'pay']):
            return 'salary'
        elif any(term in context_lower for term in ['rent', 'rental', 'lease']):
            return 'rent'
        elif any(term in context_lower for term in ['fee', 'cost', 'charge', 'price']):
            return 'fee'
        elif any(term in context_lower for term in ['deposit', 'security']):
            return 'deposit'
        elif any(term in context_lower for term in ['penalty', 'fine', 'late fee']):
            return 'penalty'
        elif any(term in context_lower for term in ['bonus', 'incentive']):
            return 'bonus'
        else:
            return 'other'

    def validate_contract_structure(self, text: str) -> Dict[str, Any]:
        """
        Validate basic contract structure
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_parties': False,
            'has_consideration': False,
            'has_terms': False,
            'has_signatures': False,
            'has_dates': False,
            'structure_score': 0,
            'missing_elements': [],
            'warnings': []
        }
        
        text_lower = text.lower()
        
        # Check for parties
        party_patterns = [
            r'between.*and', r'party.*party', r'client.*contractor',
            r'landlord.*tenant', r'employer.*employee'
        ]
        if any(re.search(pattern, text_lower) for pattern in party_patterns):
            validation['has_parties'] = True
        else:
            validation['missing_elements'].append('Clear party identification')
        
        # Check for consideration (payment/exchange)
        consideration_patterns = [
            r'consideration', r'payment', r'compensation', r'exchange',
            r'valuable consideration', r'sum of'
        ]
        if any(pattern in text_lower for pattern in consideration_patterns):
            validation['has_consideration'] = True
        else:
            validation['missing_elements'].append('Consideration or payment terms')
        
        # Check for contract terms
        terms_patterns = [
            r'terms', r'conditions', r'obligations', r'shall', r'must', r'will'
        ]
        if any(pattern in text_lower for pattern in terms_patterns):
            validation['has_terms'] = True
        else:
            validation['missing_elements'].append('Clear terms and conditions')
        
        # Check for signature areas
        signature_patterns = [
            r'signature', r'signed', r'execute', r'witness', r'notary'
        ]
        if any(pattern in text_lower for pattern in signature_patterns):
            validation['has_signatures'] = True
        else:
            validation['missing_elements'].append('Signature areas')
        
        # Check for dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'january|february|march|april|may|june|july|august|september|october|november|december'
        ]
        if any(re.search(pattern, text_lower) for pattern in date_patterns):
            validation['has_dates'] = True
        else:
            validation['missing_elements'].append('Important dates')
        
        # Calculate structure score
        elements = [
            validation['has_parties'],
            validation['has_consideration'],
            validation['has_terms'],
            validation['has_signatures'],
            validation['has_dates']
        ]
        validation['structure_score'] = sum(elements) / len(elements)
        
        # Generate warnings
        if validation['structure_score'] < 0.6:
            validation['warnings'].append('Contract appears to be missing key structural elements')
        
        if len(text.split()) < 100:
            validation['warnings'].append('Contract appears unusually short')
        
        if len(text.split()) > 10000:
            validation['warnings'].append('Contract is very long - consider summarization')
        
        return validation