"""
models/risk_assessor.py
Advanced risk assessment for legal contracts with personalized analysis
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskAssessor:
    """
    Comprehensive risk assessment for legal contracts
    - Clause-level risk scoring
    - Context-aware risk analysis
    - Personalized risk advice
    - Risk heatmap generation
    """
    
    def __init__(self):
        """Initialize the risk assessor with risk patterns and scoring rules"""
        logger.info("Initializing RiskAssessor...")
        
        # Risk patterns for different risk categories
        self.risk_patterns = self._load_risk_patterns()
        
        # Risk weights for different clause types
        self.clause_weights = self._load_clause_weights()
        
        # Context-specific risk factors
        self.context_factors = self._load_context_factors()
        
        logger.info("RiskAssessor initialized successfully!")

    def _load_risk_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load patterns that indicate different types of risks"""
        return {
            'high_risk': {
                'termination': [
                    r'immediate termination', r'without notice', r'at will',
                    r'sole discretion', r'terminate for any reason'
                ],
                'payment': [
                    r'non-refundable', r'upfront payment', r'penalty',
                    r'late fee', r'interest.*\d+%', r'collection costs'
                ],
                'liability': [
                    r'unlimited liability', r'personal guarantee', r'jointly and severally',
                    r'indemnify.*all', r'hold harmless.*any', r'consequential damages'
                ],
                'confidentiality': [
                    r'perpetual', r'indefinite', r'permanent', r'forever',
                    r'survival.*termination'
                ],
                'dispute': [
                    r'binding arbitration', r'waive.*jury trial', r'waive.*class action',
                    r'attorney.*fees.*prevailing party'
                ],
                'modification': [
                    r'unilateral', r'sole discretion', r'without consent',
                    r'modify.*any time'
                ]
            },
            'medium_risk': {
                'termination': [
                    r'30.*day.*notice', r'breach.*cure', r'material breach',
                    r'convenience'
                ],
                'payment': [
                    r'net 30', r'advance payment', r'milestone payments',
                    r'escrow'
                ],
                'liability': [
                    r'limitation.*liability', r'mutual indemnification',
                    r'reasonable efforts'
                ],
                'confidentiality': [
                    r'5.*year', r'return.*destroy', r'specific exceptions'
                ],
                'dispute': [
                    r'mediation first', r'governing law', r'specific jurisdiction'
                ]
            },
            'protective': {
                'termination': [
                    r'written notice', r'opportunity to cure', r'good cause',
                    r'mutual termination'
                ],
                'payment': [
                    r'net 15', r'invoice upon delivery', r'progress payments',
                    r'dispute resolution.*payment'
                ],
                'liability': [
                    r'cap.*liability', r'exclude.*consequential', r'insurance requirement'
                ],
                'confidentiality': [
                    r'standard exceptions', r'publicly available', r'independently developed'
                ]
            }
        }

    def _load_clause_weights(self) -> Dict[str, float]:
        """Load importance weights for different clause types"""
        return {
            'termination': 0.9,
            'payment': 0.95,
            'liability': 0.85,
            'confidentiality': 0.7,
            'intellectual_property': 0.8,
            'dispute_resolution': 0.75,
            'force_majeure': 0.6,
            'modification': 0.65,
            'general': 0.5
        }

    def _load_context_factors(self) -> Dict[str, Dict[str, float]]:
        """Load context-specific risk adjustment factors"""
        return {
            'risk_tolerance': {
                'very_low': 1.3,
                'low': 1.15,
                'moderate': 1.0,
                'high': 0.85,
                'very_high': 0.7
            },
            'business_size': {
                'individual': 1.2,
                'small_business': 1.1,
                'medium_business': 1.0,
                'large_business': 0.9
            },
            'financial_situation': {
                'tight_budget': 1.25,
                'limited_resources': 1.1,
                'adequate_resources': 1.0,
                'strong_financial': 0.9
            },
            'experience_level': {
                'beginner': 1.2,
                'intermediate': 1.0,
                'experienced': 0.9,
                'expert': 0.8
            }
        }

    def assess_risks(self, contract_text: str, contract_analysis: Dict[str, Any], 
                    user_context: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment
        
        Args:
            contract_text: Raw contract text
            contract_analysis: Results from ContractAnalyzer
            user_context: User-provided context for personalized assessment
            
        Returns:
            Dictionary containing risk assessment results
        """
        logger.info("Starting risk assessment...")
        
        try:
            # Parse user context
            context_factors = self._parse_user_context(user_context)
            
            # Assess risks for each clause
            clause_risks = self._assess_clause_risks(
                contract_analysis.get('clauses', []), context_factors
            )
            
            # Calculate overall risk score
            overall_score = self._calculate_overall_risk(clause_risks, contract_analysis)
            
            # Generate risk heatmap data
            risk_heatmap = self._generate_risk_heatmap(clause_risks, contract_text)
            
            # Create personalized advice
            personalized_advice = self._generate_personalized_advice(
                clause_risks, context_factors, contract_analysis
            )
            
            # Identify top risk areas
            top_risks = self._identify_top_risks(clause_risks)
            
            # Generate risk breakdown by category
            risk_breakdown = self._create_risk_breakdown(clause_risks)
            
            results = {
                'overall_score': overall_score,
                'risk_level': self._categorize_risk_level(overall_score),
                'clause_risks': clause_risks,
                'risk_heatmap': risk_heatmap,
                'advice': personalized_advice,
                'top_risks': top_risks,
                'breakdown': risk_breakdown,
                'context_factors': context_factors,
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Risk assessment completed. Overall score: {overall_score:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise

    def _parse_user_context(self, user_context: str) -> Dict[str, str]:
        """Parse user context to extract relevant factors"""
        context = user_context.lower() if user_context else ""
        factors = {}
        
        # Risk tolerance
        if any(word in context for word in ['very risk-averse', 'extremely cautious']):
            factors['risk_tolerance'] = 'very_low'
        elif any(word in context for word in ['risk-averse', 'cautious', 'conservative']):
            factors['risk_tolerance'] = 'low'
        elif any(word in context for word in ['risk-taking', 'aggressive', 'high-risk']):
            factors['risk_tolerance'] = 'high'
        else:
            factors['risk_tolerance'] = 'moderate'
        
        # Business size
        if any(word in context for word in ['freelancer', 'individual', 'solo', 'myself']):
            factors['business_size'] = 'individual'
        elif any(word in context for word in ['small business', 'startup', 'few employees']):
            factors['business_size'] = 'small_business'
        elif any(word in context for word in ['medium business', 'growing company']):
            factors['business_size'] = 'medium_business'
        elif any(word in context for word in ['large company', 'corporation', 'enterprise']):
            factors['business_size'] = 'large_business'
        else:
            factors['business_size'] = 'small_business'
        
        # Financial situation
        if any(word in context for word in ['tight budget', 'limited funds', 'financial constraints']):
            factors['financial_situation'] = 'tight_budget'
        elif any(word in context for word in ['strong financial', 'well-funded', 'good resources']):
            factors['financial_situation'] = 'strong_financial'
        else:
            factors['financial_situation'] = 'adequate_resources'
        
        # Experience level
        if any(word in context for word in ['new to', 'first time', 'beginner', 'inexperienced']):
            factors['experience_level'] = 'beginner'
        elif any(word in context for word in ['experienced', 'familiar with', 'done this before']):
            factors['experience_level'] = 'experienced'
        elif any(word in context for word in ['expert', 'specialist', 'professional']):
            factors['experience_level'] = 'expert'
        else:
            factors['experience_level'] = 'intermediate'
        
        return factors

    def _assess_clause_risks(self, clauses: List[Dict], context_factors: Dict[str, str]) -> Dict[str, Dict]:
        """Assess risk level for each clause"""
        clause_risks = {}
        
        for clause in clauses:
            clause_id = clause['id']
            clause_text = clause['text'].lower()
            clause_type = clause['type']
            
            # Base risk score from pattern matching
            risk_score = self._calculate_clause_risk_score(clause_text, clause_type)
            
            # Apply context adjustments
            adjusted_score = self._apply_context_adjustments(risk_score, clause_type, context_factors)
            
            # Generate risk explanation
            explanation = self._generate_risk_explanation(clause_text, clause_type, adjusted_score)
            
            # Identify specific concerns
            concerns = self._identify_clause_concerns(clause_text, clause_type)
            
            # Generate suggestions
            suggestions = self._generate_clause_suggestions(clause_type, concerns, adjusted_score)
            
            clause_risks[clause_id] = {
                'score': adjusted_score,
                'level': self._categorize_risk_level(adjusted_score),
                'explanation': explanation,
                'concerns': concerns,
                'suggestions': suggestions,
                'clause_type': clause_type,
                'weight': self.clause_weights.get(clause_type, 0.5)
            }
        
        return clause_risks

    def _calculate_clause_risk_score(self, clause_text: str, clause_type: str) -> float:
        """Calculate base risk score for a clause"""
        risk_score = 0.0
        
        # Check high-risk patterns
        high_risk_patterns = self.risk_patterns['high_risk'].get(clause_type, [])
        for pattern in high_risk_patterns:
            if re.search(pattern, clause_text):
                risk_score += 0.3
        
        # Check medium-risk patterns
        medium_risk_patterns = self.risk_patterns['medium_risk'].get(clause_type, [])
        for pattern in medium_risk_patterns:
            if re.search(pattern, clause_text):
                risk_score += 0.15
        
        # Check protective patterns (reduce risk)
        protective_patterns = self.risk_patterns['protective'].get(clause_type, [])
        for pattern in protective_patterns:
            if re.search(pattern, clause_text):
                risk_score -= 0.1
        
        # Additional risk factors
        if 'waive' in clause_text or 'waiver' in clause_text:
            risk_score += 0.2
        if 'irrevocable' in clause_text:
            risk_score += 0.25
        if 'unlimited' in clause_text:
            risk_score += 0.3
        if 'sole discretion' in clause_text:
            risk_score += 0.2
        
        # Normalize score to 0-1 range
        return min(max(risk_score, 0.0), 1.0)

    def _apply_context_adjustments(self, base_score: float, clause_type: str, context_factors: Dict[str, str]) -> float:
        """Apply user context adjustments to risk score"""
        adjusted_score = base_score
        
        # Apply context factor adjustments
        for factor_type, factor_value in context_factors.items():
            if factor_type in self.context_factors:
                multiplier = self.context_factors[factor_type].get(factor_value, 1.0)
                adjusted_score *= multiplier
        
        # Special adjustments for specific clause types
        if clause_type == 'payment' and context_factors.get('financial_situation') == 'tight_budget':
            adjusted_score *= 1.2
        
        if clause_type == 'termination' and context_factors.get('business_size') == 'individual':
            adjusted_score *= 1.15
        
        return min(adjusted_score, 1.0)

    def _generate_risk_explanation(self, clause_text: str, clause_type: str, risk_score: float) -> str:
        """Generate explanation for the risk assessment"""
        
        if risk_score >= 0.7:
            severity = "High Risk"
            impact = "This clause poses significant risk and should be carefully reviewed or negotiated."
        elif risk_score >= 0.4:
            severity = "Medium Risk"
            impact = "This clause has some risk factors that warrant attention."
        else:
            severity = "Low Risk"
            impact = "This clause appears to have minimal risk."
        
        explanations = {
            'termination': f"{severity}: This termination clause could affect your ability to maintain the contract. {impact}",
            'payment': f"{severity}: The payment terms in this clause could impact your cash flow or financial obligations. {impact}",
            'liability': f"{severity}: This liability clause determines your exposure to potential damages or losses. {impact}",
            'confidentiality': f"{severity}: The confidentiality requirements could restrict your future business activities. {impact}",
            'dispute_resolution': f"{severity}: This clause determines how disputes will be resolved, which could affect costs and outcomes. {impact}",
            'intellectual_property': f"{severity}: This clause affects ownership and usage rights of intellectual property. {impact}"
        }
        
        return explanations.get(clause_type, f"{severity}: This clause contains terms that could impact your rights or obligations. {impact}")

    def _identify_clause_concerns(self, clause_text: str, clause_type: str) -> List[str]:
        """Identify specific concerns in a clause"""
        concerns = []
        
        # Common concerning patterns
        concerning_patterns = {
            'Unlimited liability': r'unlimited.*liability',
            'No notice required': r'without.*notice|no.*notice.*required',
            'Immediate termination': r'immediate.*terminat',
            'Non-refundable payment': r'non-refundable',
            'Sole discretion': r'sole.*discretion',
            'Waiver of rights': r'waive.*rights?|waiver.*rights?',
            'Personal guarantee': r'personal.*guarantee',
            'Binding arbitration': r'binding.*arbitration',
            'No jury trial': r'waive.*jury.*trial',
            'Attorney fees': r'attorney.*fees.*prevailing',
            'Perpetual term': r'perpetual|indefinite|permanent',
            'Broad indemnification': r'indemnify.*all|hold.*harmless.*any',
            'Consequential damages': r'consequential.*damages',
            'Unilateral modification': r'unilateral.*modif|modify.*sole.*discretion'
        }
        
        for concern_name, pattern in concerning_patterns.items():
            if re.search(pattern, clause_text):
                concerns.append(concern_name)
        
        # Clause-specific concerns
        if clause_type == 'payment':
            if re.search(r'late.*fee|penalty.*fee', clause_text):
                concerns.append('Late payment penalties')
            if re.search(r'interest.*\d+%', clause_text):
                concerns.append('High interest rates')
        
        elif clause_type == 'termination':
            if re.search(r'at.*will|any.*reason', clause_text):
                concerns.append('Termination without cause')
            if re.search(r'no.*cure.*period', clause_text):
                concerns.append('No opportunity to fix issues')
        
        return concerns

    def _generate_clause_suggestions(self, clause_type: str, concerns: List[str], risk_score: float) -> List[str]:
        """Generate suggestions for addressing clause risks"""
        suggestions = []
        
        # General suggestions based on risk level
        if risk_score >= 0.7:
            suggestions.append("Consider negotiating this clause to reduce risk")
            suggestions.append("Seek legal advice before accepting these terms")
        elif risk_score >= 0.4:
            suggestions.append("Review this clause carefully and consider modifications")
        
        # Specific suggestions based on concerns
        concern_suggestions = {
            'Unlimited liability': "Request a liability cap to limit your exposure",
            'No notice required': "Negotiate for written notice requirements",
            'Immediate termination': "Request a cure period to fix any issues",
            'Non-refundable payment': "Negotiate for partial refund provisions",
            'Sole discretion': "Request mutual agreement or reasonable standards",
            'Waiver of rights': "Consider keeping important legal rights",
            'Personal guarantee': "Limit guarantees to business assets only",
            'Binding arbitration': "Consider mediation as an alternative",
            'No jury trial': "Evaluate if you want to preserve jury trial rights",
            'Attorney fees': "Negotiate for mutual attorney fee provisions",
            'Perpetual term': "Request specific time limits",
            'Broad indemnification': "Limit indemnification to specific scenarios",
            'Consequential damages': "Clarify what damages are excluded",
            'Unilateral modification': "Require mutual consent for changes"
        }
        
        for concern in concerns:
            if concern in concern_suggestions:
                suggestions.append(concern_suggestions[concern])
        
        # Clause-specific suggestions
        if clause_type == 'payment':
            if risk_score >= 0.5:
                suggestions.append("Consider milestone-based payments instead of upfront payments")
                suggestions.append("Negotiate reasonable payment terms (Net 30)")
        
        elif clause_type == 'confidentiality':
            if risk_score >= 0.5:
                suggestions.append("Ensure standard exceptions are included (publicly available information)")
                suggestions.append("Request reasonable time limits on confidentiality")
        
        return suggestions[:5]  # Limit to top 5 suggestions

    def _calculate_overall_risk(self, clause_risks: Dict[str, Dict], contract_analysis: Dict[str, Any]) -> float:
        """Calculate overall contract risk score"""
        if not clause_risks:
            return 0.5  # Default moderate risk
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for clause_id, risk_data in clause_risks.items():
            score = risk_data['score']
            weight = risk_data['weight']
            
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        overall_score = weighted_sum / total_weight
        
        # Apply contract type adjustments
        contract_type = contract_analysis.get('contract_type', 'general')
        if contract_type in ['employment', 'lease']:
            overall_score *= 1.1  # These tend to be riskier
        elif contract_type in ['nda', 'service']:
            overall_score *= 0.9  # These tend to be less risky
        
        return min(overall_score, 1.0)

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_risk_heatmap(self, clause_risks: Dict[str, Dict], contract_text: str) -> List[Dict]:
        """Generate risk heatmap data for visualization"""
        heatmap_data = []
        
        # Find clause positions in text for heatmap
        text_position = 0
        
        for clause_id, risk_data in clause_risks.items():
            risk_level = risk_data['level']
            risk_score = risk_data['score']
            
            # Find approximate position in text (simplified)
            clause_num = int(clause_id.split('_')[1]) if '_' in clause_id else 1
            estimated_position = (clause_num - 1) / len(clause_risks)
            
            heatmap_data.append({
                'clause_id': clause_id,
                'position': estimated_position,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'color': self._get_risk_color(risk_level),
                'concerns': risk_data.get('concerns', [])
            })
        
        return sorted(heatmap_data, key=lambda x: x['position'])

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors = {
            'high': '#FF4444',      # Red
            'medium': '#FFA500',    # Orange
            'low': '#90EE90'        # Light Green
        }
        return colors.get(risk_level, '#CCCCCC')

    def _generate_personalized_advice(self, clause_risks: Dict[str, Dict], 
                                    context_factors: Dict[str, str], 
                                    contract_analysis: Dict[str, Any]) -> List[Dict]:
        """Generate personalized advice based on user context"""
        advice = []
        
        # High-priority advice based on highest risks
        high_risk_clauses = [
            (clause_id, risk_data) for clause_id, risk_data in clause_risks.items() 
            if risk_data['level'] == 'high'
        ]
        
        if high_risk_clauses:
            advice.append({
                'priority': 'high',
                'category': 'Critical Issues',
                'message': f"Found {len(high_risk_clauses)} high-risk clauses that need immediate attention.",
                'action': 'Review and negotiate these clauses before signing.',
                'clauses': [clause_id for clause_id, _ in high_risk_clauses]
            })
        
        # Context-specific advice
        risk_tolerance = context_factors.get('risk_tolerance', 'moderate')
        business_size = context_factors.get('business_size', 'small_business')
        financial_situation = context_factors.get('financial_situation', 'adequate_resources')
        
        if risk_tolerance == 'very_low' and any(risk['level'] == 'medium' for risk in clause_risks.values()):
            advice.append({
                'priority': 'medium',
                'category': 'Risk Tolerance',
                'message': 'Given your conservative approach, consider additional protections.',
                'action': 'Negotiate for more favorable terms or add protective clauses.',
                'context': 'risk_averse'
            })
        
        if financial_situation == 'tight_budget':
            payment_risks = [
                clause_id for clause_id, risk_data in clause_risks.items()
                if risk_data['clause_type'] == 'payment' and risk_data['score'] >= 0.4
            ]
            if payment_risks:
                advice.append({
                    'priority': 'high',
                    'category': 'Financial Impact',
                    'message': 'Payment terms may strain your budget.',
                    'action': 'Negotiate better payment schedules or milestone-based payments.',
                    'clauses': payment_risks
                })
        
        if business_size == 'individual':
            liability_risks = [
                clause_id for clause_id, risk_data in clause_risks.items()
                if risk_data['clause_type'] == 'liability' and risk_data['score'] >= 0.5
            ]
            if liability_risks:
                advice.append({
                    'priority': 'high',
                    'category': 'Personal Protection',
                    'message': 'As an individual, liability clauses pose significant personal risk.',
                    'action': 'Consider liability insurance or negotiate caps on personal liability.',
                    'clauses': liability_risks
                })
        
        # Contract type specific advice
        contract_type = contract_analysis.get('contract_type', 'general')
        if contract_type == 'freelance':
            advice.append({
                'priority': 'medium',
                'category': 'Freelancer Protection',
                'message': 'Ensure intellectual property rights and payment terms are clearly defined.',
                'action': 'Verify ownership of work product and payment schedules.'
            })
        
        elif contract_type == 'lease':
            advice.append({
                'priority': 'medium',
                'category': 'Tenant Rights',
                'message': 'Review termination, deposit, and maintenance responsibilities carefully.',
                'action': 'Understand your rights and obligations as a tenant.'
            })
        
        return advice

    def _identify_top_risks(self, clause_risks: Dict[str, Dict]) -> List[Dict]:
        """Identify the top risk areas in the contract"""
        # Sort clauses by risk score
        sorted_risks = sorted(
            clause_risks.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_risks = []
        for clause_id, risk_data in sorted_risks[:5]:  # Top 5 risks
            if risk_data['score'] >= 0.3:  # Only include significant risks
                top_risks.append({
                    'clause_id': clause_id,
                    'clause_type': risk_data['clause_type'],
                    'risk_score': risk_data['score'],
                    'risk_level': risk_data['level'],
                    'main_concern': risk_data['concerns'][0] if risk_data['concerns'] else 'General risk',
                    'top_suggestion': risk_data['suggestions'][0] if risk_data['suggestions'] else 'Review carefully'
                })
        
        return top_risks

    def _create_risk_breakdown(self, clause_risks: Dict[str, Dict]) -> Dict[str, Any]:
        """Create risk breakdown by category"""
        breakdown = {}
        
        # Group by clause type
        clause_types = {}
        for clause_id, risk_data in clause_risks.items():
            clause_type = risk_data['clause_type']
            if clause_type not in clause_types:
                clause_types[clause_type] = []
            clause_types[clause_type].append(risk_data['score'])
        
        # Calculate average risk by type
        for clause_type, scores in clause_types.items():
            avg_score = sum(scores) / len(scores)
            breakdown[clause_type] = {
                'average_risk': avg_score,
                'risk_level': self._categorize_risk_level(avg_score),
                'clause_count': len(scores),
                'max_risk': max(scores),
                'min_risk': min(scores)
            }
        
        # Overall statistics
        all_scores = [risk['score'] for risk in clause_risks.values()]
        if all_scores:
            breakdown['overall_stats'] = {
                'total_clauses': len(all_scores),
                'average_risk': sum(all_scores) / len(all_scores),
                'high_risk_count': len([s for s in all_scores if s >= 0.7]),
                'medium_risk_count': len([s for s in all_scores if 0.4 <= s < 0.7]),
                'low_risk_count': len([s for s in all_scores if s < 0.4])
            }
        
        return breakdown

    def generate_risk_report(self, assessment_results: Dict[str, Any]) -> str:
        """Generate a comprehensive risk report"""
        overall_score = assessment_results.get('overall_score', 0)
        risk_level = assessment_results.get('risk_level', 'medium')
        top_risks = assessment_results.get('top_risks', [])
        advice = assessment_results.get('advice', [])
        
        report = f"""
CONTRACT RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL RISK ASSESSMENT
Risk Score: {overall_score:.2f}/1.00
Risk Level: {risk_level.upper()}

EXECUTIVE SUMMARY
This contract has been assessed as {risk_level} risk. """
        
        if risk_level == 'high':
            report += "Significant concerns have been identified that require immediate attention before signing."
        elif risk_level == 'medium':
            report += "Several areas warrant careful review and potential negotiation."
        else:
            report += "The contract appears to have reasonable terms with minimal risk."
        
        if top_risks:
            report += f"\n\nTOP RISK AREAS ({len(top_risks)} identified):\n"
            for i, risk in enumerate(top_risks, 1):
                report += f"{i}. {risk['clause_type'].title()} Clause - {risk['main_concern']}\n"
                report += f"   Risk Level: {risk['risk_level'].title()} ({risk['risk_score']:.2f})\n"
                report += f"   Recommendation: {risk['top_suggestion']}\n\n"
        
        if advice:
            report += "PERSONALIZED RECOMMENDATIONS:\n"
            for adv in advice:
                report += f"• {adv['category']}: {adv['message']}\n"
                report += f"  Action: {adv['action']}\n\n"
        
        report += "\nDISCLAIMER: This assessment is for informational purposes only and does not constitute legal advice. Consult with a qualified attorney for legal guidance."
        
        return report