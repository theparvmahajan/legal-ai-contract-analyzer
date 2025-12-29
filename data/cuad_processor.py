"""
data/cuad_processor.py
CUAD (Contract Understanding Atticus Dataset) processor for training and fine-tuning
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Any
import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CUADProcessor:
    """
    Process CUAD dataset for contract analysis training
    - Load and preprocess CUAD dataset
    - Create training samples for legal clause classification
    - Generate contract samples for testing
    """
    
    def __init__(self):
        """Initialize CUAD processor"""
        logger.info("Initializing CUADProcessor...")
        
        self.dataset = None
        self.processed_data = None
        self.contract_samples = []
        
        # CUAD label categories (41 categories)
        self.cuad_labels = [
            'Agreement Date', 'Competitive Restriction Exception', 'Rofr/Rofo/Rofn',
            'Document Name', 'Parties', 'Expiration Date', 'Governing Law',
            'Most Favored Nation', 'Non-Compete', 'Exclusivity', 'No-Solicit Of Customers',
            'No-Solicit Of Employees', 'Non-Disparagement', 'Termination For Convenience',
            'Anti-Assignment', 'Revenue/Customer Sharing', 'Price Restrictions',
            'Minimum Commitment', 'Volume Restriction', 'Ip Ownership Assignment',
            'Joint Ip Ownership', 'License Grant', 'Non-Transferable License',
            'Affiliate License-Licensor', 'Affiliate License-Licensee', 'Unlimited/All-You-Can-Eat-License',
            'Irrevocable Or Perpetual License', 'Source Code Escrow', 'Post-Termination Services',
            'Audit Rights', 'Uncapped Liability', 'Cap On Liability', 'Liquidated Damages',
            'Warranty Duration', 'Insurance', 'Covenant Not To Sue', 'Third Party Beneficiary',
            'Effective Date', 'Notice Period To Terminate Renewal', 'Renewal Term', 'Auto Renewal'
        ]
        
        # Simplified clause mapping for our system
        self.clause_mapping = {
            'payment': ['Revenue/Customer Sharing', 'Price Restrictions', 'Minimum Commitment'],
            'termination': ['Termination For Convenience', 'Notice Period To Terminate Renewal', 'Expiration Date'],
            'liability': ['Uncapped Liability', 'Cap On Liability', 'Liquidated Damages', 'Insurance'],
            'intellectual_property': ['Ip Ownership Assignment', 'Joint Ip Ownership', 'License Grant'],
            'confidentiality': ['Non-Disparagement', 'Source Code Escrow'],
            'dispute_resolution': ['Governing Law', 'Covenant Not To Sue', 'Audit Rights'],
            'modification': ['Anti-Assignment'],
            'general': ['Agreement Date', 'Document Name', 'Parties', 'Effective Date']
        }
        
        logger.info("CUADProcessor initialized successfully!")

    def load_cuad_dataset(self, cache_dir: str = './data/cuad_cache') -> bool:
        """
        Load CUAD dataset from Hugging Face
        
        Args:
            cache_dir: Directory to cache dataset
            
        Returns:
            True if loaded successfully
        """
        try:
            logger.info("Loading CUAD dataset...")
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load CUAD dataset (this might take a few minutes first time)
            self.dataset = load_dataset('cuad', cache_dir=cache_dir)
            
            logger.info(f"CUAD dataset loaded: {len(self.dataset['train'])} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CUAD dataset: {str(e)}")
            # Create sample data if CUAD is not available
            self._create_sample_data()
            return False

    def _create_sample_data(self):
        """Create sample contract data for development/testing"""
        logger.info("Creating sample contract data...")
        
        sample_contracts = [
            {
                'context': '''This Service Agreement ("Agreement") is entered into on January 15, 2024, 
                between TechCorp Inc., a Delaware corporation ("Company"), and John Smith ("Contractor"). 
                The Contractor agrees to provide software development services for a fee of $5,000 per month. 
                Either party may terminate this agreement with 30 days written notice. The Contractor shall 
                not disclose any confidential information received during the term of this agreement. 
                This agreement shall be governed by the laws of California.''',
                'title': 'Software Development Service Agreement',
                'parties': ['TechCorp Inc.', 'John Smith'],
                'contract_type': 'service'
            },
            {
                'context': '''Residential Lease Agreement dated March 1, 2024, between Sarah Johnson 
                ("Landlord") and Mike Wilson ("Tenant"). Monthly rent is $2,500, due on the first of each month. 
                Security deposit of $2,500 is required. Lease term is 12 months ending February 28, 2025. 
                Tenant is responsible for utilities. No pets allowed. Landlord may terminate lease for 
                non-payment of rent with 3 days notice.''',
                'title': 'Residential Lease Agreement',
                'parties': ['Sarah Johnson', 'Mike Wilson'],
                'contract_type': 'lease'
            },
            {
                'context': '''Employment Agreement between DataTech Solutions LLC ("Employer") and 
                Jennifer Brown ("Employee") dated April 10, 2024. Annual salary of $85,000 payable bi-weekly. 
                Employment is at-will and may be terminated by either party at any time. Employee agrees to 
                maintain confidentiality of proprietary information. Non-compete clause prohibits working 
                for competitors within 50 miles for 1 year after termination.''',
                'title': 'Employment Agreement',
                'parties': ['DataTech Solutions LLC', 'Jennifer Brown'],
                'contract_type': 'employment'
            },
            {
                'context': '''Non-Disclosure Agreement entered into on May 5, 2024, between InnovaCorp 
                ("Disclosing Party") and Alex Rodriguez ("Receiving Party"). Receiving Party agrees not to 
                disclose confidential information for a period of 3 years. Confidential information includes 
                trade secrets, business plans, and technical data. Breach of this agreement may result in 
                monetary damages and injunctive relief.''',
                'title': 'Non-Disclosure Agreement',
                'parties': ['InnovaCorp', 'Alex Rodriguez'],
                'contract_type': 'nda'
            },
            {
                'context': '''Freelance Web Design Contract between WebStudio Inc. ("Client") and 
                Creative Designs LLC ("Designer") dated June 20, 2024. Total project cost is $15,000 
                with 50% due upfront and 50% upon completion. Project timeline is 8 weeks. Client owns 
                all intellectual property rights to the final design. Designer provides 2 rounds of 
                revisions included in the base price.''',
                'title': 'Freelance Web Design Contract',
                'parties': ['WebStudio Inc.', 'Creative Designs LLC'],
                'contract_type': 'freelance'
            }
        ]
        
        # Convert to dataset format
        self.dataset = {
            'train': sample_contracts,
            'test': sample_contracts[:2]  # Use subset for testing
        }
        
        logger.info(f"Created {len(sample_contracts)} sample contracts")

    def process_for_classification(self) -> Dict[str, Any]:
        """
        Process dataset for clause classification training
        
        Returns:
            Processed data for training
        """
        logger.info("Processing CUAD data for classification...")
        
        processed_samples = []
        
        # Use sample data if CUAD dataset not available
        dataset_to_use = self.dataset['train'] if self.dataset else []
        
        for sample in dataset_to_use:
            context = sample.get('context', '')
            
            # Extract clauses using simple heuristics
            clauses = self._extract_clauses_from_context(context)
            
            for clause in clauses:
                # Classify clause type
                clause_type = self._classify_clause_simple(clause['text'])
                
                processed_samples.append({
                    'text': clause['text'],
                    'label': clause_type,
                    'contract_type': sample.get('contract_type', 'general'),
                    'source': 'cuad_processed'
                })
        
        # Add manual examples for better coverage
        manual_examples = self._get_manual_training_examples()
        processed_samples.extend(manual_examples)
        
        # Split into train/validation
        train_data, val_data = train_test_split(processed_samples, test_size=0.2, random_state=42)
        
        self.processed_data = {
            'train': train_data,
            'validation': val_data,
            'labels': list(set([sample['label'] for sample in processed_samples]))
        }
        
        logger.info(f"Processed {len(train_data)} training samples, {len(val_data)} validation samples")
        return self.processed_data

    def _extract_clauses_from_context(self, context: str) -> List[Dict[str, str]]:
        """Extract clause-like segments from contract text"""
        import re
        
        # Split by sentences and filter for clause-like content
        sentences = re.split(r'[.!?]+', context)
        clauses = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                clauses.append({
                    'id': f'clause_{i}',
                    'text': sentence,
                    'type': 'extracted'
                })
        
        return clauses

    def _classify_clause_simple(self, text: str) -> str:
        """Simple rule-based clause classification"""
        text_lower = text.lower()
        
        # Payment-related
        if any(term in text_lower for term in ['payment', 'fee', 'cost', 'salary', 'rent']):
            return 'payment'
        
        # Termination-related
        if any(term in text_lower for term in ['terminate', 'end', 'expire', 'cancel']):
            return 'termination'
        
        # Liability-related
        if any(term in text_lower for term in ['liable', 'liability', 'damages', 'indemnify']):
            return 'liability'
        
        # IP-related
        if any(term in text_lower for term in ['intellectual property', 'copyright', 'trademark', 'patent', 'owns']):
            return 'intellectual_property'
        
        # Confidentiality-related
        if any(term in text_lower for term in ['confidential', 'non-disclosure', 'proprietary', 'secret']):
            return 'confidentiality'
        
        # Dispute-related
        if any(term in text_lower for term in ['dispute', 'arbitration', 'court', 'governing law']):
            return 'dispute_resolution'
        
        return 'general'

    def _get_manual_training_examples(self) -> List[Dict[str, str]]:
        """Get manually curated training examples for better model performance"""
        
        examples = [
            # Payment examples
            {
                'text': 'The Client shall pay the Contractor $5,000 per month for services rendered.',
                'label': 'payment',
                'contract_type': 'service',
                'source': 'manual'
            },
            {
                'text': 'Monthly rent of $2,500 is due on the first day of each month.',
                'label': 'payment',
                'contract_type': 'lease',
                'source': 'manual'
            },
            {
                'text': 'Late payment fees of $50 will be charged for payments received after the due date.',
                'label': 'payment',
                'contract_type': 'general',
                'source': 'manual'
            },
            
            # Termination examples
            {
                'text': 'Either party may terminate this agreement with 30 days written notice.',
                'label': 'termination',
                'contract_type': 'service',
                'source': 'manual'
            },
            {
                'text': 'This agreement shall automatically terminate on December 31, 2024.',
                'label': 'termination',
                'contract_type': 'general',
                'source': 'manual'
            },
            {
                'text': 'The Company may terminate Employee immediately for cause.',
                'label': 'termination',
                'contract_type': 'employment',
                'source': 'manual'
            },
            
            # Liability examples
            {
                'text': 'Contractor shall indemnify Client against all claims arising from the services.',
                'label': 'liability',
                'contract_type': 'service',
                'source': 'manual'
            },
            {
                'text': 'In no event shall liability exceed the total amount paid under this agreement.',
                'label': 'liability',
                'contract_type': 'general',
                'source': 'manual'
            },
            {
                'text': 'Each party shall be liable for damages caused by their own negligence.',
                'label': 'liability',
                'contract_type': 'general',
                'source': 'manual'
            },
            
            # IP examples
            {
                'text': 'All intellectual property created during the project shall belong to the Client.',
                'label': 'intellectual_property',
                'contract_type': 'service',
                'source': 'manual'
            },
            {
                'text': 'Employee assigns all work product and inventions to the Company.',
                'label': 'intellectual_property',
                'contract_type': 'employment',
                'source': 'manual'
            },
            
            # Confidentiality examples
            {
                'text': 'Recipient shall not disclose confidential information to any third party.',
                'label': 'confidentiality',
                'contract_type': 'nda',
                'source': 'manual'
            },
            {
                'text': 'This confidentiality obligation shall survive termination of this agreement.',
                'label': 'confidentiality',
                'contract_type': 'nda',
                'source': 'manual'
            },
            
            # Dispute resolution examples
            {
                'text': 'Any disputes shall be resolved through binding arbitration in New York.',
                'label': 'dispute_resolution',
                'contract_type': 'general',
                'source': 'manual'
            },
            {
                'text': 'This agreement shall be governed by the laws of California.',
                'label': 'dispute_resolution',
                'contract_type': 'general',
                'source': 'manual'
            }
        ]
        
        return examples

    def get_contract_samples(self) -> List[Dict[str, Any]]:
        """Get sample contracts for testing the system"""
        
        if self.contract_samples:
            return self.contract_samples
        
        # Create comprehensive test samples
        self.contract_samples = [
            {
                'title': 'Freelance Software Development Agreement',
                'text': '''FREELANCE SOFTWARE DEVELOPMENT AGREEMENT

This Agreement is entered into on March 15, 2024, between TechStart Inc., a Delaware corporation ("Client"), and Maria Garcia, an individual ("Developer").

SERVICES: Developer agrees to create a mobile application for Client's e-commerce platform including user authentication, product catalog, shopping cart, and payment integration.

COMPENSATION: Client shall pay Developer a total of $25,000 for the completed project. Payment schedule: $12,500 upon signing this agreement, $6,250 at 50% completion milestone, and $6,250 upon final delivery and acceptance.

TIMELINE: Project shall be completed within 12 weeks from the start date. Developer will provide weekly progress reports.

INTELLECTUAL PROPERTY: All work product, including source code, designs, and documentation, shall be owned exclusively by Client upon final payment.

REVISIONS: Developer includes up to 3 rounds of revisions in the base price. Additional revisions will be charged at $150 per hour.

CONFIDENTIALITY: Developer shall not disclose any confidential information received from Client including business plans, user data, and proprietary algorithms.

TERMINATION: Either party may terminate this agreement with 2 weeks written notice. Client shall pay for all work completed up to the termination date.

LIABILITY: Developer's liability is limited to the total contract amount. Developer maintains professional liability insurance of $1 million.

GOVERNING LAW: This agreement shall be governed by California law. Any disputes shall be resolved through mediation first, then binding arbitration.

SIGNATURES:
Client: _________________ Date: _______
Developer: ______________ Date: _______''',
                'contract_type': 'freelance',
                'complexity': 'medium',
                'key_risks': ['IP ownership transfer', 'Limited liability', 'Fixed price project'],
                'parties': ['TechStart Inc.', 'Maria Garcia']
            },
            
            {
                'title': 'High-Risk Commercial Lease Agreement',
                'text': '''COMMERCIAL LEASE AGREEMENT

This Lease Agreement is made on January 1, 2024, between Metro Properties LLC ("Landlord") and Startup Cafe Inc. ("Tenant").

PREMISES: 2,500 square feet of retail space at 123 Main Street, Downtown City.

RENT: Base rent of $8,000 per month plus additional rent equal to 5% of gross monthly sales exceeding $50,000. Rent increases by 4% annually.

SECURITY DEPOSIT: $50,000 security deposit required, non-refundable if lease is terminated early by Tenant.

LEASE TERM: 5 years beginning January 1, 2024, with automatic renewal for additional 5-year terms unless either party gives 180 days notice.

TENANT IMPROVEMENTS: Tenant responsible for all improvements and modifications at Tenant's sole expense. All improvements become property of Landlord.

MAINTENANCE: Tenant responsible for ALL maintenance, repairs, utilities, insurance, and property taxes. Landlord has no maintenance obligations.

ASSIGNMENT: Tenant may not assign lease or sublet without Landlord's prior written consent, which may be withheld for any reason.

DEFAULT: Any rent payment more than 5 days late constitutes default. Landlord may terminate lease immediately upon default without cure period.

PERSONAL GUARANTEE: Tenant's CEO must provide unlimited personal guarantee for all lease obligations.

INDEMNIFICATION: Tenant indemnifies Landlord against ALL claims, damages, or losses arising from Tenant's use of premises.

NO WARRANTIES: Premises leased "AS IS" with no warranties of habitability, fitness for purpose, or compliance with applicable laws.

GOVERNING LAW: Delaware law governs. Tenant waives right to jury trial.''',
                'contract_type': 'lease',
                'complexity': 'high',
                'key_risks': ['Personal guarantee', 'No cure period', 'Unlimited liability', 'Automatic renewal'],
                'parties': ['Metro Properties LLC', 'Startup Cafe Inc.']
            },
            
            {
                'title': 'Standard Employment Agreement',
                'text': '''EMPLOYMENT AGREEMENT

This Employment Agreement is entered into on April 1, 2024, between GrowthTech Corporation ("Company") and Jennifer Kim ("Employee").

POSITION: Employee is hired as Senior Marketing Manager reporting to the Vice President of Marketing.

COMPENSATION: Annual salary of $95,000 payable bi-weekly. Employee eligible for annual performance bonus up to 15% of base salary.

BENEFITS: Health insurance (Company pays 80%), dental and vision insurance, 401k with 4% company match, 15 days PTO annually.

EMPLOYMENT TERM: Employment is at-will and may be terminated by either party with or without cause and with or without notice.

CONFIDENTIALITY: Employee shall maintain confidentiality of all proprietary information including customer lists, business strategies, and technical data.

NON-COMPETE: For 12 months after termination, Employee shall not work for direct competitors within 25-mile radius of Company's offices.

INTELLECTUAL PROPERTY: All work product created during employment belongs to Company. Employee assigns all rights to inventions and creative works.

SEVERANCE: If terminated without cause, Employee entitled to 4 weeks severance pay and continuation of health benefits for 30 days.

DISPUTE RESOLUTION: Employment disputes shall be resolved through confidential arbitration. Company pays arbitration costs.

GOVERNING LAW: This agreement governed by laws of the state where Company is headquartered.''',
                'contract_type': 'employment',
                'complexity': 'medium',
                'key_risks': ['At-will employment', 'Non-compete clause', 'IP assignment'],
                'parties': ['GrowthTech Corporation', 'Jennifer Kim']
            }
        ]
        
        return self.contract_samples

    def export_training_data(self, output_file: str = 'processed_cuad_data.json'):
        """Export processed data for model training"""
        
        if not self.processed_data:
            logger.warning("No processed data available. Run process_for_classification() first.")
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training data exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting training data: {str(e)}")

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed dataset"""
        
        if not self.processed_data:
            return {'error': 'No processed data available'}
        
        train_data = self.processed_data['train']
        val_data = self.processed_data['validation']
        
        # Label distribution
        train_labels = [sample['label'] for sample in train_data]
        val_labels = [sample['label'] for sample in val_data]
        
        from collections import Counter
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)
        
        stats = {
            'total_samples': len(train_data) + len(val_data),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'unique_labels': len(self.processed_data['labels']),
            'label_distribution': {
                'train': dict(train_label_counts),
                'validation': dict(val_label_counts)
            },
            'contract_types': list(set([sample.get('contract_type', 'unknown') for sample in train_data]))
        }
        
        return stats

    def create_fine_tuning_dataset(self, model_name: str = 'nlpaueb/legal-bert-base-uncased'):
        """Create dataset formatted for fine-tuning Legal-BERT"""
        
        if not self.processed_data:
            logger.error("No processed data available")
            return None
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
            
            # Convert to Hugging Face datasets
            train_dataset = Dataset.from_list(self.processed_data['train'])
            val_dataset = Dataset.from_list(self.processed_data['validation'])
            
            # Tokenize datasets
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
            
            # Create label mapping
            label_to_id = {label: i for i, label in enumerate(self.processed_data['labels'])}
            id_to_label = {i: label for label, i in label_to_id.items()}
            
            def encode_labels(examples):
                return {'labels': [label_to_id[label] for label in examples['label']]}
            
            train_dataset = train_dataset.map(encode_labels, batched=True)
            val_dataset = val_dataset.map(encode_labels, batched=True)
            
            return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'num_labels': len(self.processed_data['labels'])
            }
            
        except Exception as e:
            logger.error(f"Error creating fine-tuning dataset: {str(e)}")
            return None