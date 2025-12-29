"""
models/persona_summarizer.py
Persona-based contract summaries tailored to different user types
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PersonaSummarizer:
    """
    Generate persona-specific contract summaries and highlights
    - Freelancer-focused summaries
    - Tenant-specific analysis
    - Small business considerations
    - General consumer protection
    - Student-friendly explanations
    """
    
    def __init__(self):
        """Initialize the persona summarizer with templates and focus areas"""
        logger.info("Initializing PersonaSummarizer...")
        
        # Persona-specific focus areas
        self.persona_focus = self._load_persona_focus_areas()
        
        # Summary templates for different personas
        self.summary_templates = self._load_summary_templates()
        
        # Warning patterns for each persona
        self.persona_warnings = self._load_persona_warnings()
        
        logger.info("PersonaSummarizer initialized successfully!")

    def _load_persona_focus_areas(self) -> Dict[str, Dict[str, List[str]]]:
        """Load focus areas for different personas"""
        return {
            'freelancer': {
                'high_priority': ['payment', 'intellectual_property', 'termination', 'scope_of_work'],
                'key_concerns': ['payment_terms', 'deliverables', 'ownership_rights', 'timeline'],
                'important_clauses': ['milestone_payments', 'revision_limits', 'late_penalties']
            },
            'tenant': {
                'high_priority': ['payment', 'termination', 'liability', 'maintenance'],
                'key_concerns': ['rent_amount', 'deposit', 'lease_term', 'maintenance_responsibility'],
                'important_clauses': ['early_termination', 'rent_increases', 'security_deposit']
            },
            'small_business': {
                'high_priority': ['liability', 'termination', 'payment', 'intellectual_property'],
                'key_concerns': ['liability_limits', 'business_continuity', 'cash_flow', 'competition'],
                'important_clauses': ['limitation_of_liability', 'non_compete', 'force_majeure']
            },
            'general_consumer': {
                'high_priority': ['payment', 'termination', 'liability', 'dispute_resolution'],
                'key_concerns': ['hidden_fees', 'cancellation_rights', 'personal_liability', 'dispute_costs'],
                'important_clauses': ['cooling_off_period', 'automatic_renewal', 'dispute_resolution']
            },
            'student': {
                'high_priority': ['payment', 'termination', 'liability'],
                'key_concerns': ['affordability', 'flexibility', 'personal_risk'],
                'important_clauses': ['payment_plans', 'early_termination', 'liability_caps']
            }
        }

    def _load_summary_templates(self) -> Dict[str, str]:
        """Load summary templates for different personas"""
        return {
            'freelancer': """
As a freelancer, here's what this contract means for you:

**Payment & Money**: {payment_summary}
**Your Work & Rights**: {work_summary}
**Project Timeline**: {timeline_summary}
**What Happens If Things Go Wrong**: {risk_summary}
**Key Things to Negotiate**: {negotiation_points}
            """.strip(),
            
            'tenant': """
As a tenant, here's what this lease means for you:

**Rent & Costs**: {payment_summary}
**Your Living Situation**: {living_summary}
**Responsibilities**: {responsibility_summary}
**Moving Out**: {termination_summary}
**Important Rights**: {rights_summary}
            """.strip(),
            
            'small_business': """
As a small business owner, here's what this contract means:

**Financial Impact**: {financial_summary}
**Business Risks**: {risk_summary}
**Operational Requirements**: {operational_summary}
**Exit Strategy**: {termination_summary}
**Legal Protection**: {protection_summary}
            """.strip(),
            
            'general_consumer': """
Here's what this contract means in plain English:

**What You're Paying**: {payment_summary}
**What You're Getting**: {service_summary}
**Your Rights**: {rights_summary}
**Risks to Watch**: {risk_summary}
**How to Cancel**: {cancellation_summary}
            """.strip(),
            
            'student': """
Here's what this contract means (explained simply):

**Costs & Payment**: {payment_summary}
**What You Need to Do**: {obligations_summary}
**What Could Go Wrong**: {risk_summary}
**Getting Out Early**: {exit_summary}
**Getting Help**: {support_summary}
            """.strip()
        }

    def _load_persona_warnings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load warning patterns specific to each persona"""
        return {
            'freelancer': {
                'payment_risks': [
                    r'payment.*60.*days', r'net.*90', r'payment.*completion',
                    r'no.*payment.*until', r'final.*payment.*upon'
                ],
                'ip_risks': [
                    r'work.*for.*hire', r'all.*rights.*client', r'assign.*rights',
                    r'intellectual.*property.*client'
                ],
                'scope_risks': [
                    r'unlimited.*revisions', r'scope.*may.*change', r'additional.*work.*no.*charge'
                ]
            },
            'tenant': {
                'rent_risks': [
                    r'rent.*increase.*any.*time', r'rent.*adjustment.*discretion',
                    r'additional.*fees', r'utilities.*not.*included'
                ],
                'termination_risks': [
                    r'no.*refund.*deposit', r'forfeit.*deposit', r'early.*termination.*penalty',
                    r'eviction.*discretion'
                ],
                'maintenance_risks': [
                    r'tenant.*responsible.*all.*repairs', r'landlord.*not.*responsible.*maintenance'
                ]
            },
            'small_business': {
                'liability_risks': [
                    r'unlimited.*liability', r'personal.*guarantee', r'jointly.*severally',
                    r'indemnify.*all.*damages'
                ],
                'operational_risks': [
                    r'exclusive.*dealing', r'non.*compete.*any.*business', r'territorial.*restrictions'
                ],
                'termination_risks': [
                    r'terminate.*without.*cause', r'immediate.*termination', r'no.*compensation.*termination'
                ]
            },
            'general_consumer': {
                'payment_risks': [
                    r'automatic.*renewal', r'cancellation.*fee', r'non.*refundable',
                    r'price.*increase.*any.*time'
                ],
                'service_risks': [
                    r'service.*may.*be.*discontinued', r'no.*guarantee.*service',
                    r'modify.*service.*any.*time'
                ]
            },
            'student': {
                'financial_risks': [
                    r'parent.*guarantor', r'personal.*guarantee', r'late.*fees',
                    r'collection.*costs'
                ],
                'academic_risks': [
                    r'no.*refund.*withdrawal', r'academic.*dismissal', r'grade.*requirements'
                ]
            }
        }

    def generate_summary(self, contract_text: str, contract_analysis: Dict[str, Any], 
                        persona: str, user_context: str = "") -> Dict[str, Any]:
        """
        Generate persona-specific contract summary
        
        Args:
            contract_text: Raw contract text
            contract_analysis: Results from ContractAnalyzer
            persona: User persona (freelancer, tenant, small_business, etc.)
            user_context: Additional user context
            
        Returns:
            Dictionary containing persona-specific summary and highlights
        """
        logger.info(f"Generating summary for persona: {persona}")
        
        try:
            # Get persona focus areas
            focus_areas = self.persona_focus.get(persona, self.persona_focus['general_consumer'])
            
            # Extract persona-specific information
            persona_info = self._extract_persona_info(contract_analysis, focus_areas, persona)
            
            # Generate main summary
            summary = self._generate_main_summary(persona_info, persona)
            
            # Identify key highlights
            highlights = self._identify_highlights(contract_analysis, persona, focus_areas)
            
            # Generate warnings
            warnings = self._generate_warnings(contract_text, contract_analysis, persona)
            
            # Create action items
            action_items = self._create_action_items(persona_info, persona, warnings)
            
            # Generate questions to ask
            questions = self._generate_questions(persona, contract_analysis)
            
            results = {
                'summary': summary,
                'highlights': highlights,
                'warnings': warnings,
                'action_items': action_items,
                'questions_to_ask': questions,
                'persona': persona,
                'focus_areas': focus_areas,
                'key_metrics': persona_info.get('metrics', {}),
                'summary_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Persona summary generated successfully for {persona}")
            return results
            
        except Exception as e:
            logger.error(f"Error generating persona summary: {str(e)}")
            raise

    def _extract_persona_info(self, contract_analysis: Dict[str, Any], 
                            focus_areas: Dict[str, List[str]], persona: str) -> Dict[str, Any]:
        """Extract information relevant to the specific persona"""
        
        clauses = contract_analysis.get('clauses', [])
        amounts = contract_analysis.get('amounts', [])
        dates = contract_analysis.get('key_dates', [])
        
        persona_info = {
            'key_clauses': {},
            'financial_terms': [],
            'important_dates': [],
            'obligations': [],
            'rights': [],
            'risks': [],
            'metrics': {}
        }
        
        # Extract clauses relevant to persona
        high_priority_types = focus_areas.get('high_priority', [])
        for clause in clauses:
            if clause['type'] in high_priority_types:
                persona_info['key_clauses'][clause['type']] = {
                    'text': clause['text'],
                    'plain_english': clause.get('plain_english', ''),
                    'importance': clause.get('importance', 'medium')
                }
        
        # Extract financial information
        for amount in amounts:
            persona_info['financial_terms'].append({
                'type': amount['type'],
                'amount': amount['amount'],
                'context': amount.get('context', '')
            })
        
        # Extract important dates
        for date in dates:
            persona_info['important_dates'].append({
                'type': date['type'],
                'date': date['date'],
                'context': date.get('context', '')
            })
        
        # Calculate persona-specific metrics
        persona_info['metrics'] = self._calculate_persona_metrics(
            contract_analysis, persona, persona_info
        )
        
        return persona_info

    def _calculate_persona_metrics(self, contract_analysis: Dict[str, Any], 
                                 persona: str, persona_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics specific to each persona"""
        
        metrics = {}
        clauses = contract_analysis.get('clauses', [])
        
        if persona == 'freelancer':
            payment_clauses = [c for c in clauses if c['type'] == 'payment']
            ip_clauses = [c for c in clauses if c['type'] == 'intellectual_property']
            
            metrics.update({
                'payment_terms_count': len(payment_clauses),
                'ip_protection_level': 'high' if ip_clauses else 'low',
                'contract_complexity': 'high' if len(clauses) > 10 else 'medium'
            })
            
        elif persona == 'tenant':
            payment_clauses = [c for c in clauses if c['type'] == 'payment']
            termination_clauses = [c for c in clauses if c['type'] == 'termination']
            
            metrics.update({
                'rent_clauses_count': len(payment_clauses),
                'termination_flexibility': 'low' if termination_clauses else 'medium',
                'tenant_protection_level': self._assess_tenant_protection(clauses)
            })
            
        elif persona == 'small_business':
            liability_clauses = [c for c in clauses if c['type'] == 'liability']
            termination_clauses = [c for c in clauses if c['type'] == 'termination']
            
            metrics.update({
                'liability_exposure': 'high' if liability_clauses else 'medium',
                'business_continuity_risk': 'high' if termination_clauses else 'low',
                'operational_restrictions': len([c for c in clauses if 'non-compete' in c['text'].lower()])
            })
        
        return metrics

    def _assess_tenant_protection(self, clauses: List[Dict]) -> str:
        """Assess level of tenant protection in lease"""
        protection_score = 0
        
        for clause in clauses:
            clause_text = clause['text'].lower()
            if any(term in clause_text for term in ['tenant rights', 'landlord responsible', 'reasonable notice']):
                protection_score += 1
            if any(term in clause_text for term in ['tenant solely responsible', 'no warranty', 'as is']):
                protection_score -= 1
        
        if protection_score >= 2:
            return 'high'
        elif protection_score >= 0:
            return 'medium'
        else:
            return 'low'

    def _generate_main_summary(self, persona_info: Dict[str, Any], persona: str) -> str:
        """Generate the main persona-specific summary"""
        
        template = self.summary_templates.get(persona, self.summary_templates['general_consumer'])
        
        # Prepare summary components based on persona
        if persona == 'freelancer':
            components = self._prepare_freelancer_components(persona_info)
        elif persona == 'tenant':
            components = self._prepare_tenant_components(persona_info)
        elif persona == 'small_business':
            components = self._prepare_business_components(persona_info)
        elif persona == 'student':
            components = self._prepare_student_components(persona_info)
        else:
            components = self._prepare_general_components(persona_info)
        
        # Fill template with components
        try:
            summary = template.format(**components)
        except KeyError as e:
            # Fallback if template keys don't match
            logger.warning(f"Template key missing: {e}, using fallback summary")
            summary = self._generate_fallback_summary(persona_info, persona)
        
        return summary

    def _prepare_freelancer_components(self, persona_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepare summary components for freelancer persona"""
        
        # Payment summary
        payment_terms = persona_info.get('financial_terms', [])
        if payment_terms:
            payment_summary = f"You'll be paid {payment_terms[0]['amount']} for this work. "
            payment_details = [term for term in payment_terms if 'payment' in term['type'].lower()]
            if payment_details:
                payment_summary += "Check when and how you'll receive payment - this is crucial for your cash flow."
        else:
            payment_summary = "Payment terms are not clearly specified. This needs clarification before you start work."
        
        # Work and IP summary
        ip_clauses = persona_info.get('key_clauses', {}).get('intellectual_property', {})
        if ip_clauses:
            work_summary = "The contract addresses ownership of your work. " + ip_clauses.get('plain_english', 'Review who owns what you create.')
        else:
            work_summary = "Intellectual property ownership is not clearly defined. Clarify who owns your work and any code/designs you create."
        
        # Timeline summary
        dates = persona_info.get('important_dates', [])
        timeline_summary = "Key deadlines: "
        if dates:
            timeline_summary += ", ".join([f"{date['type']}: {date['date']}" for date in dates[:3]])
        else:
            timeline_summary += "No specific deadlines mentioned. Consider adding milestone dates."
        
        # Risk summary
        risks = persona_info.get('risks', [])
        risk_summary = "Main concerns: "
        if risks:
            risk_summary += ". ".join(risks[:2]) + "."
        else:
            risk_summary += "Review termination clauses and what happens if the project scope changes."
        
        # Negotiation points
        negotiation_points = self._get_freelancer_negotiation_points(persona_info)
        
        return {
            'payment_summary': payment_summary,
            'work_summary': work_summary,
            'timeline_summary': timeline_summary,
            'risk_summary': risk_summary,
            'negotiation_points': negotiation_points
        }

    def _prepare_tenant_components(self, persona_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepare summary components for tenant persona"""
        
        # Payment summary (rent)
        payment_terms = persona_info.get('financial_terms', [])
        rent_info = [term for term in payment_terms if 'rent' in term['type'].lower()]
        if rent_info:
            payment_summary = f"Monthly rent is {rent_info[0]['amount']}. "
            deposit_info = [term for term in payment_terms if 'deposit' in term['type'].lower()]
            if deposit_info:
                payment_summary += f"Security deposit: {deposit_info[0]['amount']}."
        else:
            payment_summary = "Rent amount is not clearly specified in the reviewed sections."
        
        # Living situation
        living_summary = "This lease covers your rights and responsibilities as a tenant. "
        maintenance_clauses = persona_info.get('key_clauses', {}).get('maintenance', {})
        if maintenance_clauses:
            living_summary += "Pay attention to who handles repairs and maintenance."
        else:
            living_summary += "Maintenance responsibilities are not clearly defined."
        
        # Responsibilities
        obligations = persona_info.get('obligations', [])
        if obligations:
            responsibility_summary = "Your main responsibilities: " + ". ".join(obligations[:2]) + "."
        else:
            responsibility_summary = "Your specific responsibilities need clarification."
        
        # Termination summary
        termination_clause = persona_info.get('key_clauses', {}).get('termination', {})
        if termination_clause:
            termination_summary = termination_clause.get('plain_english', 'Review the lease termination terms carefully.')
        else:
            termination_summary = "Lease termination terms are not clearly specified. This is important for planning your move."
        
        # Rights summary
        rights_summary = self._get_tenant_rights_summary(persona_info)
        
        return {
            'payment_summary': payment_summary,
            'living_summary': living_summary,
            'responsibility_summary': responsibility_summary,
            'termination_summary': termination_summary,
            'rights_summary': rights_summary
        }

    def _prepare_business_components(self, persona_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepare summary components for small business persona"""
        
        # Financial impact
        payment_terms = persona_info.get('financial_terms', [])
        if payment_terms:
            amounts = [term['amount'] for term in payment_terms if term['amount']]
            financial_summary = f"Financial commitments include: {', '.join(amounts[:3])}. Consider the impact on your cash flow."
        else:
            financial_summary = "Financial terms need clarification. Ensure all costs are clearly defined."
        
        # Business risks
        liability_clause = persona_info.get('key_clauses', {}).get('liability', {})
        if liability_clause:
            risk_summary = "Liability exposure: " + liability_clause.get('plain_english', 'Review liability terms carefully.')
        else:
            risk_summary = "Liability terms are not clearly defined. This could expose your business to unexpected risks."
        
        # Operational requirements
        obligations = persona_info.get('obligations', [])
        if obligations:
            operational_summary = "Key operational requirements: " + ". ".join(obligations[:2]) + "."
        else:
            operational_summary = "Operational requirements need clarification."
        
        # Termination
        termination_clause = persona_info.get('key_clauses', {}).get('termination', {})
        if termination_clause:
            termination_summary = "Exit terms: " + termination_clause.get('plain_english', 'Review termination carefully.')
        else:
            termination_summary = "Exit strategy is not clearly defined. Plan for how to end this relationship."
        
        # Legal protection
        protection_summary = self._get_business_protection_summary(persona_info)
        
        return {
            'financial_summary': financial_summary,
            'risk_summary': risk_summary,
            'operational_summary': operational_summary,
            'termination_summary': termination_summary,
            'protection_summary': protection_summary
        }

    def _prepare_student_components(self, persona_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepare summary components for student persona"""
        
        # Cost and payment
        payment_terms = persona_info.get('financial_terms', [])
        if payment_terms:
            payment_summary = f"You'll need to pay: {', '.join([term['amount'] for term in payment_terms[:2]])}. "
            payment_summary += "Check if payment plans are available and what happens if you can't pay on time."
        else:
            payment_summary = "Costs are not clearly specified. Get a clear breakdown of all fees."
        
        # Obligations
        obligations = persona_info.get('obligations', [])
        if obligations:
            obligations_summary = "What you need to do: " + ". ".join(obligations[:2]) + "."
        else:
            obligations_summary = "Your responsibilities are not clearly outlined."
        
        # Risks in simple terms
        risks = persona_info.get('risks', [])
        risk_summary = "Things that could cause problems: "
        if risks:
            risk_summary += ". ".join(risks[:2]) + ". Make sure you understand these before signing."
        else:
            risk_summary += "Late payments, breaking rules, or not meeting requirements could have consequences."
        
        # Getting out early
        termination_clause = persona_info.get('key_clauses', {}).get('termination', {})
        if termination_clause:
            exit_summary = "If you need to leave early: " + termination_clause.get('plain_english', 'Check the cancellation policy.')
        else:
            exit_summary = "Cancellation terms are unclear. Find out what happens if you need to withdraw."
        
        # Support
        support_summary = "If you have problems, look for dispute resolution or customer service terms. Don't hesitate to ask questions if anything is confusing."
        
        return {
            'payment_summary': payment_summary,
            'obligations_summary': obligations_summary,
            'risk_summary': risk_summary,
            'exit_summary': exit_summary,
            'support_summary': support_summary
        }

    def _prepare_general_components(self, persona_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepare summary components for general consumer persona"""
        
        # Payment
        payment_terms = persona_info.get('financial_terms', [])
        if payment_terms:
            payment_summary = f"Costs: {', '.join([term['amount'] for term in payment_terms[:3]])}."
        else:
            payment_summary = "Payment terms need clarification."
        
        # Service
        service_summary = "This contract outlines the services or products you'll receive and your obligations."
        
        # Rights
        rights_summary = "Your rights include reasonable service delivery and dispute resolution options."
        
        # Risks
        risks = persona_info.get('risks', [])
        risk_summary = "Watch out for: " + ", ".join(risks[:3]) if risks else "Review all terms carefully."
        
        # Cancellation
        termination_clause = persona_info.get('key_clauses', {}).get('termination', {})
        cancellation_summary = termination_clause.get('plain_english', 'Check cancellation and refund policies.') if termination_clause else "Cancellation terms are not clearly specified."
        
        return {
            'payment_summary': payment_summary,
            'service_summary': service_summary,
            'rights_summary': rights_summary,
            'risk_summary': risk_summary,
            'cancellation_summary': cancellation_summary
        }

    def _get_freelancer_negotiation_points(self, persona_info: Dict[str, Any]) -> str:
        """Get negotiation points for freelancers"""
        points = []
        
        if not persona_info.get('financial_terms'):
            points.append("Clear payment schedule with milestones")
        
        ip_clauses = persona_info.get('key_clauses', {}).get('intellectual_property')
        if not ip_clauses:
            points.append("Intellectual property ownership rights")
        
        points.extend([
            "Scope change process and additional payment",
            "Reasonable revision limits",
            "Clear project timeline and deliverables"
        ])
        
        return ". ".join(points[:3]) + "."

    def _get_tenant_rights_summary(self, persona_info: Dict[str, Any]) -> str:
        """Get tenant rights summary"""
        rights = [
            "Right to peaceful enjoyment of the property",
            "Right to proper notice before landlord entry",
            "Right to habitable living conditions"
        ]
        
        # Add specific rights based on contract content
        key_clauses = persona_info.get('key_clauses', {})
        if 'privacy' in str(key_clauses).lower():
            rights.append("Privacy protections are specified")
        
        return ". ".join(rights[:3]) + "."

    def _get_business_protection_summary(self, persona_info: Dict[str, Any]) -> str:
        """Get business protection summary"""
        protections = []
        
        key_clauses = persona_info.get('key_clauses', {})
        if key_clauses.get('liability'):
            protections.append("Liability limitations are defined")
        else:
            protections.append("Consider adding liability caps")
        
        if key_clauses.get('force_majeure'):
            protections.append("Force majeure protections included")
        else:
            protections.append("Consider adding force majeure clause")
        
        protections.append("Review insurance requirements")
        
        return ". ".join(protections[:3]) + "."

    def _generate_fallback_summary(self, persona_info: Dict[str, Any], persona: str) -> str:
        """Generate fallback summary when template fails"""
        summary = f"Summary for {persona.replace('_', ' ').title()}:\n\n"
        
        if persona_info.get('financial_terms'):
            summary += f"Financial terms: {len(persona_info['financial_terms'])} payment-related clauses found.\n"
        
        if persona_info.get('key_clauses'):
            clause_types = list(persona_info['key_clauses'].keys())
            summary += f"Key areas covered: {', '.join(clause_types)}.\n"
        
        if persona_info.get('important_dates'):
            summary += f"Important dates: {len(persona_info['important_dates'])} timeline items identified.\n"
        
        summary += "\nRecommendation: Review all terms carefully and consider consulting with a legal professional."
        
        return summary

    def _identify_highlights(self, contract_analysis: Dict[str, Any], persona: str, 
                          focus_areas: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Identify key highlights for the persona"""
        highlights = []
        clauses = contract_analysis.get('clauses', [])
        amounts = contract_analysis.get('amounts', [])
        
        # Priority clause highlights
        high_priority = focus_areas.get('high_priority', [])
        for clause in clauses:
            if clause['type'] in high_priority:
                highlights.append({
                    'type': 'clause',
                    'category': clause['type'],
                    'title': f"{clause['type'].replace('_', ' ').title()} Clause",
                    'description': clause.get('plain_english', 'Important contract terms'),
                    'importance': clause.get('importance', 'high'),
                    'clause_id': clause['id']
                })
        
        # Financial highlights
        if amounts:
            for amount in amounts[:3]:  # Top 3 amounts
                highlights.append({
                    'type': 'financial',
                    'category': 'payment',
                    'title': f"{amount['type'].replace('_', ' ').title()}",
                    'description': f"Amount: {amount['amount']}",
                    'importance': 'high'
                })
        
        # Persona-specific highlights
        if persona == 'freelancer':
            ip_clauses = [c for c in clauses if c['type'] == 'intellectual_property']
            if not ip_clauses:
                highlights.append({
                    'type': 'warning',
                    'category': 'missing_clause',
                    'title': 'IP Rights Not Specified',
                    'description': 'Intellectual property ownership is not clearly defined',
                    'importance': 'high'
                })
        
        elif persona == 'tenant':
            maintenance_clauses = [c for c in clauses if 'maintenance' in c['text'].lower()]
            if maintenance_clauses:
                highlights.append({
                    'type': 'info',
                    'category': 'responsibility',
                    'title': 'Maintenance Responsibilities',
                    'description': 'Contract specifies maintenance and repair responsibilities',
                    'importance': 'medium'
                })
        
        # Limit to most important highlights
        highlights = sorted(highlights, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['importance']], reverse=True)
        return highlights[:8]

    def _generate_warnings(self, contract_text: str, contract_analysis: Dict[str, Any], persona: str) -> List[Dict[str, str]]:
        """Generate persona-specific warnings"""
        warnings = []
        text_lower = contract_text.lower()
        
        # Get persona warning patterns
        persona_warning_patterns = self.persona_warnings.get(persona, {})
        
        for warning_category, patterns in persona_warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    warning_msg = self._get_warning_message(warning_category, persona, pattern)
                    warnings.append({
                        'category': warning_category,
                        'message': warning_msg,
                        'severity': 'high' if 'unlimited' in pattern or 'guarantee' in pattern else 'medium'
                    })
                    break  # One warning per category
        
        # Universal warnings
        universal_patterns = {
            'Automatic renewal': r'automatic.*renew|auto.*renew',
            'Personal guarantee': r'personal.*guarantee|personally.*liable',
            'No refund policy': r'no.*refund|non.*refundable',
            'Immediate termination': r'immediate.*terminat|terminate.*immediately'
        }
        
        for warning_name, pattern in universal_patterns.items():
            if re.search(pattern, text_lower):
                warnings.append({
                    'category': 'universal_risk',
                    'message': f"{warning_name}: This could significantly impact your rights or obligations.",
                    'severity': 'medium'
                })
        
        return warnings[:6]  # Limit to most important warnings

    def _get_warning_message(self, warning_category: str, persona: str, pattern: str) -> str:
        """Get appropriate warning message for category and persona"""
        
        warning_messages = {
            'freelancer': {
                'payment_risks': 'Payment terms may negatively impact your cash flow.',
                'ip_risks': 'You may lose ownership rights to your work.',
                'scope_risks': 'Project scope could expand without additional compensation.'
            },
            'tenant': {
                'rent_risks': 'Rent terms may change unfavorably during your tenancy.',
                'termination_risks': 'You may lose rights or money when moving out.',
                'maintenance_risks': 'You may be responsible for costly repairs.'
            },
            'small_business': {
                'liability_risks': 'Your business may face unlimited financial exposure.',
                'operational_risks': 'Your business operations may be severely restricted.',
                'termination_risks': 'Contract termination could disrupt business continuity.'
            },
            'general_consumer': {
                'payment_risks': 'You may face unexpected costs or payment obligations.',
                'service_risks': 'Service quality or availability may not meet expectations.'
            },
            'student': {
                'financial_risks': 'You or your family may face unexpected financial obligations.',
                'academic_risks': 'Your academic progress may be at risk.'
            }
        }
        
        persona_messages = warning_messages.get(persona, {})
        return persona_messages.get(warning_category, 'This clause may pose risks to your interests.')

    def _create_action_items(self, persona_info: Dict[str, Any], persona: str, warnings: List[Dict]) -> List[str]:
        """Create actionable items for the persona"""
        actions = []
        
        # Actions based on warnings
        high_severity_warnings = [w for w in warnings if w.get('severity') == 'high']
        if high_severity_warnings:
            actions.append("🚨 Address high-risk clauses before signing")
        
        # Persona-specific actions
        if persona == 'freelancer':
            if not persona_info.get('financial_terms'):
                actions.append("💰 Clarify payment terms and schedule")
            if not persona_info.get('key_clauses', {}).get('intellectual_property'):
                actions.append("🎨 Define intellectual property ownership")
            actions.append("📋 Request detailed scope of work document")
        
        elif persona == 'tenant':
            actions.extend([
                "🏠 Inspect property condition before signing",
                "💰 Understand all fees and deposits",
                "📋 Clarify maintenance responsibilities"
            ])
        
        elif persona == 'small_business':
            actions.extend([
                "🛡️ Review liability and insurance requirements",
                "📊 Assess financial impact on cash flow",
                "🤝 Consider reciprocal terms where possible"
            ])
        
        elif persona == 'student':
            actions.extend([
                "💰 Understand all costs and payment options",
                "👨‍👩‍👧 Discuss with parents/guardians if needed",
                "❓ Ask questions about anything unclear"
            ])
        
        # General actions
        actions.extend([
            "📖 Read the entire contract carefully",
            "⚖️ Consider legal consultation if needed",
            "📝 Document any changes or agreements"
        ])
        
        return actions[:8]  # Limit to most important actions

    def _generate_questions(self, persona: str, contract_analysis: Dict[str, Any]) -> List[str]:
        """Generate relevant questions for the persona to ask"""
        
        questions = []
        
        # Persona-specific questions
        if persona == 'freelancer':
            questions.extend([
                "When exactly will I be paid for each milestone?",
                "Who owns the intellectual property I create?",
                "What happens if the project scope changes?",
                "How many revisions are included in the price?",
                "What are the consequences if deadlines are missed?"
            ])
        
        elif persona == 'tenant':
            questions.extend([
                "What utilities are included in the rent?",
                "How much notice is required to terminate the lease?",
                "Who is responsible for maintenance and repairs?",
                "Under what conditions can rent be increased?",
                "What happens to my security deposit?"
            ])
        
        elif persona == 'small_business':
            questions.extend([
                "What are the liability limits and insurance requirements?",
                "How can the contract be terminated by either party?",
                "Are there any restrictions on working with competitors?",
                "What happens if circumstances beyond our control prevent performance?",
                "How are disputes resolved and what are the costs?"
            ])
        
        elif persona == 'student':
            questions.extend([
                "What payment options and financial aid are available?",
                "What happens if I need to withdraw or take a leave?",
                "Are there any additional fees not mentioned?",
                "What academic standards must I maintain?",
                "Who can I contact if I have problems or questions?"
            ])
        
        # Universal questions
        questions.extend([
            "Can any terms be negotiated or modified?",
            "What are the consequences of breaking this contract?",
            "Are there any automatic renewal or extension clauses?"
        ])
        
        return questions[:10]  # Limit to most relevant questions