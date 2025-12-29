"""
utils/visualization.py
Visualization generator for contract analysis results
"""

import json
import logging
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generate interactive visualizations for contract analysis
    - Risk heatmaps
    - Knowledge graphs
    - Timeline visualizations
    - Risk distribution charts
    """
    
    def __init__(self):
        """Initialize visualization generator"""
        logger.info("Initializing VisualizationGenerator...")
        
        self.color_schemes = {
            'risk': {
                'high': '#e53e3e',
                'medium': '#dd6b20', 
                'low': '#38a169'
            },
            'clause_types': {
                'payment': '#3182ce',
                'termination': '#e53e3e',
                'liability': '#d69e2e',
                'confidentiality': '#805ad5',
                'intellectual_property': '#319795',
                'dispute_resolution': '#dd6b20',
                'general': '#718096'
            }
        }
        
        logger.info("VisualizationGenerator initialized successfully!")

    def create_visualizations(self, contract_analysis: Dict[str, Any], 
                            risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create all visualizations for contract analysis
        
        Args:
            contract_analysis: Contract analysis results
            risk_analysis: Risk assessment results
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            visualizations = {
                'risk_chart': self._create_risk_distribution_chart(risk_analysis),
                'knowledge_graph': self._create_knowledge_graph(contract_analysis),
                'timeline': self._create_timeline_chart(contract_analysis),
                'clause_breakdown': self._create_clause_breakdown_chart(contract_analysis),
                'heatmap_data': self._create_heatmap_data(risk_analysis)
            }
            
            logger.info("All visualizations created successfully")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return self._create_fallback_visualizations()

    def _create_risk_distribution_chart(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk distribution pie chart"""
        try:
            breakdown = risk_analysis.get('breakdown', {})
            overall_stats = breakdown.get('overall_stats', {})
            
            if not overall_stats:
                return self._create_empty_chart("No risk data available")
            
            # Data for pie chart
            labels = ['High Risk', 'Medium Risk', 'Low Risk']
            values = [
                overall_stats.get('high_risk_count', 0),
                overall_stats.get('medium_risk_count', 0),
                overall_stats.get('low_risk_count', 0)
            ]
            colors = ['#e53e3e', '#dd6b20', '#38a169']
            
            # Create Plotly figure
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Risk Distribution by Clause",
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            
            return {
                'type': 'pie_chart',
                'data': fig.to_dict(),
                'summary': f"Total clauses analyzed: {sum(values)}"
            }
            
        except Exception as e:
            logger.error(f"Error creating risk chart: {str(e)}")
            return self._create_empty_chart("Error creating risk chart")

    def _create_knowledge_graph(self, contract_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph of contract relationships"""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add contract as central node
            contract_type = contract_analysis.get('contract_type', 'Contract')
            G.add_node('Contract', type='contract', label=contract_type.title())
            
            # Add parties
            parties = contract_analysis.get('parties', [])
            for i, party in enumerate(parties[:4]):  # Limit to 4 parties
                party_id = f"party_{i}"
                G.add_node(party_id, type='party', label=party.get('name', f'Party {i+1}'))
                G.add_edge('Contract', party_id, relationship='involves')
            
            # Add key clauses
            clauses = contract_analysis.get('clauses', [])
            clause_types = set()
            for clause in clauses[:8]:  # Limit to 8 clauses
                clause_type = clause.get('type', 'general')
                if clause_type not in clause_types:
                    clause_types.add(clause_type)
                    clause_id = f"clause_{clause_type}"
                    G.add_node(clause_id, type='clause', label=clause_type.replace('_', ' ').title())
                    G.add_edge('Contract', clause_id, relationship='contains')
            
            # Add key dates
            dates = contract_analysis.get('key_dates', [])
            for i, date in enumerate(dates[:3]):  # Limit to 3 dates
                date_id = f"date_{i}"
                date_label = date.get('type', 'Date').replace('_', ' ').title()
                G.add_node(date_id, type='date', label=date_label)
                G.add_edge('Contract', date_id, relationship='specifies')
            
            # Convert to visualization format
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Prepare node and edge data
            nodes = []
            edges = []
            
            for node_id, data in G.nodes(data=True):
                x, y = pos[node_id]
                nodes.append({
                    'id': node_id,
                    'label': data.get('label', node_id),
                    'type': data.get('type', 'default'),
                    'x': float(x),
                    'y': float(y),
                    'color': self._get_node_color(data.get('type', 'default'))
                })
            
            for source, target, data in G.edges(data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'relationship': data.get('relationship', 'connected')
                })
            
            return {
                'type': 'network_graph',
                'nodes': nodes,
                'edges': edges,
                'summary': f"Graph with {len(nodes)} entities and {len(edges)} relationships"
            }
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {str(e)}")
            return {
                'type': 'network_graph',
                'nodes': [],
                'edges': [],
                'error': 'Could not generate knowledge graph'
            }

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type"""
        colors = {
            'contract': '#3182ce',
            'party': '#38a169',
            'clause': '#d69e2e',
            'date': '#805ad5',
            'default': '#718096'
        }
        return colors.get(node_type, colors['default'])

    def _create_timeline_chart(self, contract_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create timeline visualization of key dates"""
        try:
            dates = contract_analysis.get('key_dates', [])
            
            if not dates:
                return {
                    'type': 'timeline',
                    'data': [],
                    'message': 'No timeline data available'
                }
            
            # Prepare timeline data
            timeline_events = []
            for i, date_info in enumerate(dates):
                timeline_events.append({
                    'date': date_info.get('date', 'Unknown'),
                    'type': date_info.get('type', 'event').replace('_', ' ').title(),
                    'context': date_info.get('context', '')[:100] + '...' if len(date_info.get('context', '')) > 100 else date_info.get('context', ''),
                    'order': i
                })
            
            return {
                'type': 'timeline',
                'data': timeline_events,
                'summary': f"{len(timeline_events)} key dates identified"
            }
            
        except Exception as e:
            logger.error(f"Error creating timeline: {str(e)}")
            return {
                'type': 'timeline',
                'data': [],
                'error': 'Could not generate timeline'
            }

    def _create_clause_breakdown_chart(self, contract_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create clause type breakdown bar chart"""
        try:
            clauses = contract_analysis.get('clauses', [])
            
            if not clauses:
                return self._create_empty_chart("No clauses to analyze")
            
            # Count clause types
            clause_counts = {}
            for clause in clauses:
                clause_type = clause.get('type', 'general')
                clause_counts[clause_type] = clause_counts.get(clause_type, 0) + 1
            
            # Prepare data for bar chart
            clause_types = list(clause_counts.keys())
            counts = list(clause_counts.values())
            colors = [self.color_schemes['clause_types'].get(ct, '#718096') for ct in clause_types]
            
            # Create Plotly bar chart
            fig = go.Figure(data=[go.Bar(
                x=clause_types,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Clause Types Distribution",
                xaxis_title="Clause Type",
                yaxis_title="Count",
                height=400,
                xaxis_tickangle=-45,
                font=dict(size=12)
            )
            
            return {
                'type': 'bar_chart',
                'data': fig.to_dict(),
                'summary': f"{len(clause_types)} different clause types found"
            }
            
        except Exception as e:
            logger.error(f"Error creating clause breakdown: {str(e)}")
            return self._create_empty_chart("Error creating clause breakdown")

    def _create_heatmap_data(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap data for risk visualization"""
        try:
            clause_risks = risk_analysis.get('clause_risks', {})
            
            heatmap_data = []
            for clause_id, risk_info in clause_risks.items():
                heatmap_data.append({
                    'clause_id': clause_id,
                    'clause_type': risk_info.get('clause_type', 'general'),
                    'risk_score': risk_info.get('score', 0),
                    'risk_level': risk_info.get('level', 'low'),
                    'color': self.color_schemes['risk'].get(risk_info.get('level', 'low')),
                    'explanation': risk_info.get('explanation', 'No explanation available')[:100]
                })
            
            return {
                'type': 'heatmap',
                'data': heatmap_data,
                'summary': f"Risk heatmap for {len(heatmap_data)} clauses"
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmap data: {str(e)}")
            return {
                'type': 'heatmap',
                'data': [],
                'error': 'Could not generate heatmap data'
            }

    def _create_empty_chart(self, message: str) -> Dict[str, Any]:
        """Create empty chart placeholder"""
        return {
            'type': 'empty',
            'message': message,
            'data': None
        }

    def _create_fallback_visualizations(self) -> Dict[str, Any]:
        """Create fallback visualizations when main generation fails"""
        return {
            'risk_chart': self._create_empty_chart("Risk chart not available"),
            'knowledge_graph': {
                'type': 'network_graph',
                'nodes': [],
                'edges': [],
                'message': 'Knowledge graph not available'
            },
            'timeline': {
                'type': 'timeline',
                'data': [],
                'message': 'Timeline not available'
            },
            'clause_breakdown': self._create_empty_chart("Clause breakdown not available"),
            'heatmap_data': {
                'type': 'heatmap',
                'data': [],
                'message': 'Heatmap not available'
            }
        }

    def export_visualization(self, viz_data: Dict[str, Any], 
                           output_path: str, viz_type: str = 'html') -> bool:
        """Export visualization to file"""
        try:
            if viz_type == 'html' and 'data' in viz_data and viz_data['data']:
                # Export Plotly chart as HTML
                fig_dict = viz_data['data']
                fig = go.Figure(fig_dict)
                fig.write_html(output_path)
                return True
            elif viz_type == 'json':
                # Export as JSON
                with open(output_path, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                return True
            else:
                logger.warning(f"Unsupported export type: {viz_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting visualization: {str(e)}")
            return False

    def create_summary_dashboard(self, all_visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary dashboard combining all visualizations"""
        try:
            dashboard = {
                'title': 'Contract Analysis Dashboard',
                'timestamp': datetime.now().isoformat(),
                'sections': []
            }
            
            # Add each visualization as a section
            for viz_name, viz_data in all_visualizations.items():
                section = {
                    'name': viz_name.replace('_', ' ').title(),
                    'type': viz_data.get('type', 'unknown'),
                    'data': viz_data,
                    'summary': viz_data.get('summary', f"{viz_name} visualization")
                }
                dashboard['sections'].append(section)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {
                'title': 'Contract Analysis Dashboard',
                'error': 'Could not create dashboard',
                'sections': []
            }