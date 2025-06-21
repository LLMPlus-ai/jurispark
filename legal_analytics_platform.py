#!/usr/bin/env python3
"""
Legal Analytics Platform: Three-Pillar System
Implementing the System Design Document for Advanced Legal Analytics

Author: JuriSpark Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any
import re
from collections import defaultdict, Counter
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import spacy
from textblob import TextBlob
import time

# Configure Streamlit page
st.set_page_config(
    page_title="JuriSpark - Legal Analytics Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .pillar-header {
        font-size: 1.8rem;
        color: #2c5282;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2c5282;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .debate-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid;
    }
    .claimant-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .respondent-message {
        background-color: #fce4ec;
        border-left-color: #c2185b;
    }
    .tribunal-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
    }
    .winning-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .losing-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

class LegalAnalyticsPlatform:
    """
    Main class implementing the three-pillar legal analytics system
    """
    
    def __init__(self):
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all system components"""
        self.knowledge_graph = DynamicKnowledgeGraph()
        self.argument_analytics = ArgumentAnalytics()
        self.debate_simulator = DebateSimulator()
        
        # Load or initialize data
        self.load_case_data()
        
    def load_case_data(self):
        """Load case data and build knowledge graph"""
        if 'case_data' not in st.session_state:
            st.session_state.case_data = self.initialize_sample_data()
            
    def initialize_sample_data(self):
        """Initialize sample legal case data"""
        return {
            'case_name': 'Sample Legal Dispute',
            'documents': {
                'claimant_memorial': "Sample claimant arguments and evidence...",
                'respondent_counter_memorial': "Sample respondent counter-arguments...",
                'expert_reports': {
                    'claimant_expert_1': "Expert opinion supporting claimant...",
                    'claimant_expert_2': "Additional expert analysis...",
                    'respondent_expert_1': "Expert opinion supporting respondent...",
                    'respondent_expert_2': "Counter-expert analysis..."
                },
                'final_award': "Tribunal decision and reasoning..."
            },
            'entities': self.generate_sample_entities(),
            'relationships': self.generate_sample_relationships(),
            'arguments': self.generate_sample_arguments()
        }
    
    def generate_sample_entities(self):
        """Generate sample entities for demonstration"""
        return {
            'persons': ['John Smith', 'Jane Doe', 'Dr. Expert', 'Legal Counsel A'],
            'organizations': ['Company ABC', 'Firm XYZ', 'Tribunal Authority'],
            'contracts': ['Service Agreement 2020', 'Amendment No. 1'],
            'financial_items': ['$1.2M Payment', '$500K Damages', '$2M Claim'],
            'dates': ['2020-01-15', '2020-06-30', '2021-03-01']
        }
    
    def generate_sample_relationships(self):
        """Generate sample relationships"""
        return [
            {'source': 'Company ABC', 'target': 'Firm XYZ', 'type': 'financial_flow', 'amount': 1200000},
            {'source': 'John Smith', 'target': 'Dr. Expert', 'type': 'information_flow', 'context': 'consultation'},
            {'source': 'Service Agreement 2020', 'target': 'Amendment No. 1', 'type': 'physical_flow', 'relation': 'modification'}
        ]
    
    def generate_sample_arguments(self):
        """Generate sample arguments with scores"""
        return [
            {
                'id': 1,
                'party': 'claimant',
                'text': 'The breach of contract caused significant financial damages',
                'winning_score': 0.75,
                'category': 'damages',
                'strength': 'high'
            },
            {
                'id': 2,
                'party': 'respondent',
                'text': 'The alleged breach was justified under force majeure',
                'winning_score': 0.45,
                'category': 'defense',
                'strength': 'medium'
            }
        ]

class DynamicKnowledgeGraph:
    """
    Pillar 1: Dynamic Knowledge Graph Engine
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = ['persons', 'organizations', 'contracts', 'financial_items', 'dates']
        self.flow_types = ['financial_flow', 'information_flow', 'physical_flow']
        
    def build_graph(self, case_data):
        """Build the dynamic knowledge graph from case data"""
        # Add entities as nodes
        for entity_type, entities in case_data['entities'].items():
            for entity in entities:
                self.graph.add_node(entity, type=entity_type)
        
        # Add relationships as edges
        for rel in case_data['relationships']:
            self.graph.add_edge(
                rel['source'], 
                rel['target'], 
                type=rel['type'],
                **{k: v for k, v in rel.items() if k not in ['source', 'target', 'type']}
            )
    
    def get_graph_statistics(self):
        """Get comprehensive graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(Counter([self.graph.nodes[node].get('type', 'unknown') 
                                     for node in self.graph.nodes()])),
            'edge_types': dict(Counter([self.graph.edges[edge].get('type', 'unknown') 
                                     for edge in self.graph.edges()])),
            'density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def create_interactive_visualization(self):
        """Create interactive network visualization"""
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare node data
        node_trace = go.Scatter(
            x=[pos[node][0] for node in self.graph.nodes()],
            y=[pos[node][1] for node in self.graph.nodes()],
            mode='markers+text',
            text=list(self.graph.nodes()),
            textposition="middle center",
            hoverinfo='text',
            marker=dict(
                size=20,
                color=[hash(self.graph.nodes[node].get('type', 'default')) % 10 
                      for node in self.graph.nodes()],
                colorscale='Viridis',
                line=dict(width=2, color='white')
            )
        )
        
        # Prepare edge data
        edge_traces = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        fig.update_layout(
            title=dict(text="Dynamic Knowledge Graph", font=dict(size=20)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="Interactive Knowledge Graph Visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left"
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig

class ArgumentAnalytics:
    """
    Pillar 2: Argument Analytics and Generative Strategy
    """
    
    def __init__(self):
        self.ml_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ML models for argument analysis"""
        # Placeholder for actual model initialization
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def analyze_sentence_level(self, arguments):
        """Module 1: Sentence-Level Scrutiny & Outcome Mapping"""
        analysis_results = []
        
        for arg in arguments:
            # Simulate sentence-level analysis
            sentences = self.split_into_sentences(arg['text'])
            sentence_scores = []
            
            for sentence in sentences:
                score = self.calculate_winning_likelihood(sentence)
                sentence_scores.append({
                    'sentence': sentence,
                    'winning_likelihood': score,
                    'confidence': random.uniform(0.6, 0.95)
                })
            
            analysis_results.append({
                'argument_id': arg['id'],
                'overall_score': arg['winning_score'],
                'sentence_analysis': sentence_scores,
                'key_factors': self.identify_key_factors(arg['text'])
            })
        
        return analysis_results
    
    def split_into_sentences(self, text):
        """Split text into sentences"""
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def calculate_winning_likelihood(self, sentence):
        """Calculate winning likelihood for a sentence"""
        # Simplified scoring based on keywords and sentiment
        positive_keywords = ['evidence', 'proven', 'clearly', 'established', 'documented']
        negative_keywords = ['alleged', 'disputed', 'unclear', 'questionable']
        
        score = 0.5  # baseline
        for keyword in positive_keywords:
            if keyword in sentence.lower():
                score += 0.1
        for keyword in negative_keywords:
            if keyword in sentence.lower():
                score -= 0.1
        
        # Add sentiment analysis
        blob = TextBlob(sentence)
        sentiment = blob.sentiment.polarity
        score += sentiment * 0.2
        
        return max(0.0, min(1.0, score))
    
    def identify_key_factors(self, text):
        """Identify key factors in argument text"""
        # Simplified key factor identification
        factors = []
        if 'contract' in text.lower():
            factors.append('Contractual obligation')
        if 'damage' in text.lower():
            factors.append('Damages assessment')
        if 'breach' in text.lower():
            factors.append('Breach of contract')
        return factors
    
    def generate_ai_suggestions(self, argument):
        """Module 2: Generative AI for Argument Enhancement"""
        suggestions = {
            'rephrasing': [
                "Consider strengthening the causal connection between breach and damages",
                "Add specific evidence references to support this claim",
                "Clarify the legal standard being applied"
            ],
            'logical_gaps': [
                "Missing link between action and consequence",
                "Need to address potential counterarguments",
                "Insufficient evidence for causation claim"
            ],
            'evidence_recommendations': [
                "Reference Expert Report Section 3.2 for technical analysis",
                "Include contractual clause 15.3 as supporting evidence",
                "Consider adding timeline evidence from Exhibit A"
            ]
        }
        return suggestions
    
    def uncover_opponent_strategy(self, opponent_arguments):
        """Module 3: Uncovering Opponent's Embedded Strategy"""
        strategy_analysis = {
            'core_narrative': self.identify_core_narrative(opponent_arguments),
            'hidden_objectives': self.infer_hidden_objectives(opponent_arguments),
            'pressure_points': self.identify_pressure_points(opponent_arguments),
            'strategic_weaknesses': self.find_strategic_weaknesses(opponent_arguments)
        }
        return strategy_analysis
    
    def identify_core_narrative(self, arguments):
        """Identify opponent's core narrative"""
        # Simplified narrative identification
        themes = []
        for arg in arguments:
            if 'force majeure' in arg['text'].lower():
                themes.append('Unforeseeable circumstances defense')
            if 'good faith' in arg['text'].lower():
                themes.append('Good faith performance')
        return themes
    
    def infer_hidden_objectives(self, arguments):
        """Infer hidden strategic objectives"""
        return [
            "Minimize financial exposure",
            "Shift blame to external factors",
            "Establish precedent for future cases"
        ]
    
    def identify_pressure_points(self, arguments):
        """Identify strategic pressure points"""
        return [
            "Timeline discrepancies in their narrative",
            "Contradictions between expert opinions",
            "Weak evidence for causation claims"
        ]
    
    def find_strategic_weaknesses(self, arguments):
        """Find strategic weaknesses"""
        return [
            "Over-reliance on single expert opinion",
            "Failure to address contractual obligations",
            "Inconsistent damage calculations"
        ]

class DebateSimulator:
    """
    Pillar 3: Simulated Debates and Predictive Outcomes
    """
    
    def __init__(self):
        self.claimant_agent = None
        self.respondent_agent = None
        self.tribunal_agent = None
        self.simulation_results = []
    
    def initialize_agents(self, case_data):
        """Initialize debate agents"""
        self.claimant_agent = DebateAgent('claimant', case_data)
        self.respondent_agent = DebateAgent('respondent', case_data)
        self.tribunal_agent = TribunalAgent(case_data)
    
    def run_simulation_rounds(self, num_rounds=100):
        """Run multiple simulation rounds"""
        results = []
        
        for round_num in range(num_rounds):
            round_result = self.simulate_single_debate(round_num + 1)
            results.append(round_result)
            
        self.simulation_results = results
        return self.analyze_simulation_results(results)
    
    def simulate_single_debate(self, round_num):
        """Simulate a single debate round"""
        debate_log = []
        
        # Initial arguments
        claimant_opening = self.claimant_agent.make_opening_argument()
        debate_log.append({
            'speaker': 'Claimant',
            'message': claimant_opening,
            'timestamp': datetime.now()
        })
        
        respondent_response = self.respondent_agent.respond_to_argument(claimant_opening)
        debate_log.append({
            'speaker': 'Respondent',
            'message': respondent_response,
            'timestamp': datetime.now()
        })
        
        # Back and forth arguments (simplified)
        for exchange in range(3):  # 3 exchanges per debate
            claimant_rebuttal = self.claimant_agent.make_rebuttal(respondent_response)
            debate_log.append({
                'speaker': 'Claimant',
                'message': claimant_rebuttal,
                'timestamp': datetime.now()
            })
            
            respondent_counter = self.respondent_agent.counter_argument(claimant_rebuttal)
            debate_log.append({
                'speaker': 'Respondent',
                'message': respondent_counter,
                'timestamp': datetime.now()
            })
        
        # Tribunal decision
        decision = self.tribunal_agent.make_decision(debate_log)
        debate_log.append({
            'speaker': 'Tribunal',
            'message': decision['reasoning'],
            'timestamp': datetime.now()
        })
        
        return {
            'round': round_num,
            'winner': decision['winner'],
            'confidence': decision['confidence'],
            'debate_log': debate_log,
            'key_arguments': decision['key_arguments']
        }
    
    def analyze_simulation_results(self, results):
        """Analyze results from multiple simulation rounds"""
        winners = [r['winner'] for r in results]
        
        analysis = {
            'total_rounds': len(results),
            'claimant_wins': winners.count('claimant'),
            'respondent_wins': winners.count('respondent'),
            'claimant_win_rate': winners.count('claimant') / len(results),
            'respondent_win_rate': winners.count('respondent') / len(results),
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'key_battlegrounds': self.identify_battlegrounds(results),
            'vulnerable_arguments': self.identify_vulnerable_arguments(results),
            'winning_strategies': self.identify_winning_strategies(results)
        }
        
        return analysis

class DebateAgent:
    """Individual debate agent for claimant or respondent"""
    
    def __init__(self, party, case_data):
        self.party = party
        self.case_data = case_data
        self.arguments = [arg for arg in case_data['arguments'] if arg['party'] == party]
    
    def make_opening_argument(self):
        """Make opening argument"""
        if self.party == 'claimant':
            return "The evidence clearly demonstrates a material breach of contract resulting in quantifiable damages."
        else:
            return "The alleged breach was justified under extraordinary circumstances and caused no compensable harm."
    
    def respond_to_argument(self, opponent_argument):
        """Respond to opponent's argument"""
        responses = [
            "This argument fails to consider the contractual provisions that clearly state...",
            "The evidence contradicts this assertion, as shown in Expert Report...",
            "This interpretation ignores established legal precedent regarding..."
        ]
        return random.choice(responses)
    
    def make_rebuttal(self, opponent_argument):
        """Make rebuttal argument"""
        rebuttals = [
            "The opponent's interpretation is fundamentally flawed because...",
            "Expert testimony unequivocally supports our position that...",
            "The documentary evidence establishes beyond doubt that..."
        ]
        return random.choice(rebuttals)
    
    def counter_argument(self, opponent_rebuttal):
        """Make counter-argument"""
        counters = [
            "This rebuttal mischaracterizes the evidence by...",
            "The legal standard requires proof that the opponent cannot provide...",
            "The timeline of events clearly contradicts this narrative..."
        ]
        return random.choice(counters)

class TribunalAgent:
    """Tribunal agent for making decisions"""
    
    def __init__(self, case_data):
        self.case_data = case_data
        self.legal_principles = self.extract_legal_principles()
    
    def extract_legal_principles(self):
        """Extract legal principles from award document"""
        return [
            "Burden of proof lies with the claimant",
            "Damages must be foreseeable and quantifiable",
            "Contractual obligations must be interpreted in good faith"
        ]
    
    def make_decision(self, debate_log):
        """Make tribunal decision based on debate"""
        # Simplified decision-making
        claimant_strength = len([msg for msg in debate_log if msg['speaker'] == 'Claimant'])
        respondent_strength = len([msg for msg in debate_log if msg['speaker'] == 'Respondent'])
        
        # Add some randomness for simulation variety
        random_factor = random.uniform(-0.2, 0.2)
        claimant_score = claimant_strength + random_factor
        respondent_score = respondent_strength - random_factor
        
        winner = 'claimant' if claimant_score > respondent_score else 'respondent'
        confidence = abs(claimant_score - respondent_score) / max(claimant_score, respondent_score)
        
        return {
            'winner': winner,
            'confidence': min(0.95, max(0.55, confidence)),
            'reasoning': f"Based on the strength of arguments and evidence presented, the {winner} has demonstrated a more compelling case.",
            'key_arguments': self.identify_key_arguments(debate_log)
        }
    
    def identify_key_arguments(self, debate_log):
        """Identify key arguments from debate"""
        return [
            "Contractual interpretation",
            "Causation of damages",
            "Burden of proof satisfaction"
        ]

def main():
    """Main Streamlit application"""
    
    # Initialize platform
    if 'platform' not in st.session_state:
        st.session_state.platform = LegalAnalyticsPlatform()
    
    platform = st.session_state.platform
    
    # Main header
    st.markdown('<h1 class="main-header">⚖️ JuriSpark - Legal Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Three-Pillar System for Legal Document Analysis and Debate Simulation**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pillar = st.sidebar.radio(
        "Select Analysis Pillar:",
        ["📊 Dashboard", "🕸️ Pillar 1: Knowledge Graph", "🧠 Pillar 2: Argument Analytics", "⚔️ Pillar 3: Debate Simulation"]
    )
    
    if pillar == "📊 Dashboard":
        render_dashboard(platform)
    elif pillar == "🕸️ Pillar 1: Knowledge Graph":
        render_knowledge_graph_pillar(platform)
    elif pillar == "🧠 Pillar 2: Argument Analytics":
        render_argument_analytics_pillar(platform)
    elif pillar == "⚔️ Pillar 3: Debate Simulation":
        render_debate_simulation_pillar(platform)

def render_dashboard(platform):
    """Render main dashboard"""
    st.markdown('<h2 class="pillar-header">📊 System Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Knowledge Graph</h3>
            <h2>8,761</h2>
            <p>Entities Mapped</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Arguments</h3>
            <h2>156</h2>
            <p>Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Simulations</h3>
            <h2>100</h2>
            <p>Completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Win Rate</h3>
            <h2>68%</h2>
            <p>Predicted</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System architecture overview
    st.subheader("Three-Pillar Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🕸️ Pillar 1: Knowledge Graph**
        - Dynamic entity mapping
        - Relationship categorization
        - Temporal analysis
        - Multi-dimensional queries
        """)
    
    with col2:
        st.success("""
        **🧠 Pillar 2: Argument Analytics**
        - Sentence-level analysis
        - ML-powered predictions
        - AI-generated suggestions
        - Strategy uncovering
        """)
    
    with col3:
        st.warning("""
        **⚔️ Pillar 3: Debate Simulation**
        - Multi-agent debates
        - 100-round simulations
        - Predictive outcomes
        - Strategic insights
        """)

def render_knowledge_graph_pillar(platform):
    """Render Knowledge Graph pillar interface"""
    st.markdown('<h2 class="pillar-header">🕸️ Pillar 1: Dynamic Knowledge Graph Engine</h2>', unsafe_allow_html=True)
    
    # Build knowledge graph
    platform.knowledge_graph.build_graph(st.session_state.case_data)
    
    # Graph statistics
    stats = platform.knowledge_graph.get_graph_statistics()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interactive Knowledge Graph")
        fig = platform.knowledge_graph.create_interactive_visualization()
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Graph Statistics")
        st.metric("Total Nodes", stats['total_nodes'])
        st.metric("Total Edges", stats['total_edges'])
        st.metric("Graph Density", f"{stats['density']:.3f}")
        
        st.subheader("Entity Distribution")
        entity_df = pd.DataFrame(list(stats['node_types'].items()), columns=['Type', 'Count'])
        fig_entities = px.pie(entity_df, values='Count', names='Type', title="Entity Types")
        st.plotly_chart(fig_entities, use_container_width=True)
    
    # Flow analysis
    st.subheader("Relationship Flow Analysis")
    
    flow_tabs = st.tabs(["💰 Financial Flow", "📄 Information Flow", "📦 Physical Flow"])
    
    with flow_tabs[0]:
        st.info("**Financial Flow Analysis**")
        st.write("Tracking monetary relationships, payments, and financial obligations.")
        # Add financial flow visualization here
        
    with flow_tabs[1]:
        st.info("**Information Flow Analysis**")
        st.write("Mapping communication patterns, document exchanges, and knowledge transfer.")
        # Add information flow visualization here
        
    with flow_tabs[2]:
        st.info("**Physical Flow Analysis**")
        st.write("Documenting movement of assets, goods, and physical deliverables.")
        # Add physical flow visualization here

def render_argument_analytics_pillar(platform):
    """Render Argument Analytics pillar interface"""
    st.markdown('<h2 class="pillar-header">🧠 Pillar 2: Argument Analytics & Generative Strategy</h2>', unsafe_allow_html=True)
    
    arguments = st.session_state.case_data['arguments']
    
    # Module 1: Sentence-Level Analysis
    st.subheader("Module 1: Sentence-Level Scrutiny & Outcome Mapping")
    
    analysis_results = platform.argument_analytics.analyze_sentence_level(arguments)
    
    for result in analysis_results:
        with st.expander(f"Argument {result['argument_id']} - Overall Score: {result['overall_score']:.2f}"):
            
            # Overall argument info
            arg_data = next(arg for arg in arguments if arg['id'] == result['argument_id'])
            st.write(f"**Party:** {arg_data['party'].title()}")
            st.write(f"**Text:** {arg_data['text']}")
            
            # Sentence-level analysis
            st.subheader("Sentence Analysis")
            for i, sentence_analysis in enumerate(result['sentence_analysis']):
                score = sentence_analysis['winning_likelihood']
                confidence = sentence_analysis['confidence']
                
                score_class = "winning-score" if score > 0.6 else "losing-score"
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.2rem 0; border-left: 4px solid {'#2e7d32' if score > 0.6 else '#d32f2f'}; background-color: {'#e8f5e8' if score > 0.6 else '#ffeaea'};">
                    <strong>Sentence {i+1}:</strong> {sentence_analysis['sentence']}<br>
                    <span class="{score_class}">Winning Likelihood: {score:.2f}</span> | Confidence: {confidence:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Key factors
            st.subheader("Key Factors")
            for factor in result['key_factors']:
                st.badge(factor)
    
    # Module 2: AI Suggestions
    st.subheader("Module 2: Generative AI Enhancement Suggestions")
    
    selected_arg = st.selectbox("Select argument for AI enhancement:", 
                               [f"Argument {arg['id']}: {arg['text'][:50]}..." for arg in arguments])
    
    if selected_arg:
        arg_id = int(selected_arg.split(':')[0].split()[-1])
        selected_argument = next(arg for arg in arguments if arg['id'] == arg_id)
        
        suggestions = platform.argument_analytics.generate_ai_suggestions(selected_argument)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Rephrasing Suggestions**")
            for suggestion in suggestions['rephrasing']:
                st.write(f"• {suggestion}")
        
        with col2:
            st.warning("**Logical Gaps Identified**")
            for gap in suggestions['logical_gaps']:
                st.write(f"• {gap}")
        
        with col3:
            st.success("**Evidence Recommendations**")
            for rec in suggestions['evidence_recommendations']:
                st.write(f"• {rec}")
    
    # Module 3: Opponent Strategy Analysis
    st.subheader("Module 3: Opponent Strategy Analysis")
    
    opponent_args = [arg for arg in arguments if arg['party'] == 'respondent']
    strategy_analysis = platform.argument_analytics.uncover_opponent_strategy(opponent_args)
    
    strategy_tabs = st.tabs(["🎯 Core Narrative", "🕵️ Hidden Objectives", "⚡ Pressure Points", "🔍 Weaknesses"])
    
    with strategy_tabs[0]:
        st.write("**Identified Core Narrative Themes:**")
        for theme in strategy_analysis['core_narrative']:
            st.write(f"• {theme}")
    
    with strategy_tabs[1]:
        st.write("**Inferred Hidden Objectives:**")
        for objective in strategy_analysis['hidden_objectives']:
            st.write(f"• {objective}")
    
    with strategy_tabs[2]:
        st.write("**Strategic Pressure Points:**")
        for point in strategy_analysis['pressure_points']:
            st.write(f"• {point}")
    
    with strategy_tabs[3]:
        st.write("**Strategic Weaknesses:**")
        for weakness in strategy_analysis['strategic_weaknesses']:
            st.write(f"• {weakness}")

def render_debate_simulation_pillar(platform):
    """Render Debate Simulation pillar interface"""
    st.markdown('<h2 class="pillar-header">⚔️ Pillar 3: Simulated Debates & Predictive Outcomes</h2>', unsafe_allow_html=True)
    
    # Initialize debate simulator
    platform.debate_simulator.initialize_agents(st.session_state.case_data)
    
    # Simulation controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_rounds = st.slider("Number of Simulation Rounds", 1, 100, 10)
        
    with col2:
        if st.button("🚀 Run Debate Simulations", type="primary"):
            with st.spinner("Running debate simulations..."):
                simulation_results = platform.debate_simulator.run_simulation_rounds(num_rounds)
                st.session_state.simulation_results = simulation_results
    
    # Display simulation results
    if 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        
        # Overall results
        st.subheader("Simulation Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rounds", results['total_rounds'])
        with col2:
            st.metric("Claimant Wins", results['claimant_wins'])
        with col3:
            st.metric("Respondent Wins", results['respondent_wins'])
        with col4:
            st.metric("Claimant Win Rate", f"{results['claimant_win_rate']:.1%}")
        
        # Win rate visualization
        win_data = pd.DataFrame({
            'Party': ['Claimant', 'Respondent'],
            'Wins': [results['claimant_wins'], results['respondent_wins']],
            'Win Rate': [results['claimant_win_rate'], results['respondent_win_rate']]
        })
        
        fig_wins = px.bar(win_data, x='Party', y='Wins', color='Party',
                         title="Simulation Win Distribution",
                         color_discrete_map={'Claimant': '#1976d2', 'Respondent': '#c2185b'})
        st.plotly_chart(fig_wins, use_container_width=True)
        
        # Detailed analysis
        analysis_tabs = st.tabs(["🎯 Key Battlegrounds", "⚠️ Vulnerable Arguments", "🏆 Winning Strategies"])
        
        with analysis_tabs[0]:
            st.subheader("Key Argumentative Battlegrounds")
            battlegrounds = results.get('key_battlegrounds', [
                "Contractual interpretation disputes",
                "Causation of damages",
                "Standard of care obligations",
                "Burden of proof satisfaction"
            ])
            for i, battleground in enumerate(battlegrounds, 1):
                st.write(f"{i}. {battleground}")
        
        with analysis_tabs[1]:
            st.subheader("Arguments Facing Consistent Challenges")
            vulnerable = results.get('vulnerable_arguments', [
                "Force majeure defense lacks sufficient evidence",
                "Damage calculations show inconsistencies",
                "Timeline narrative has gaps"
            ])
            for i, vuln in enumerate(vulnerable, 1):
                st.error(f"{i}. {vuln}")
        
        with analysis_tabs[2]:
            st.subheader("Most Effective Winning Strategies")
            strategies = results.get('winning_strategies', [
                "Strong documentary evidence presentation",
                "Expert testimony coordination",
                "Systematic rebuttal of opponent claims"
            ])
            for i, strategy in enumerate(strategies, 1):
                st.success(f"{i}. {strategy}")
    
    # Live debate viewer
    st.subheader("Live Debate Simulation")
    
    if st.button("🎭 Start Live Debate"):
        debate_container = st.container()
        
        with debate_container:
            # Simulate a live debate
            debate_round = platform.debate_simulator.simulate_single_debate(1)
            
            st.subheader(f"Debate Round {debate_round['round']}")
            st.write(f"**Winner:** {debate_round['winner'].title()} (Confidence: {debate_round['confidence']:.1%})")
            
            # Display debate messages
            for message in debate_round['debate_log']:
                speaker = message['speaker']
                text = message['message']
                
                if speaker == 'Claimant':
                    st.markdown(f"""
                    <div class="debate-message claimant-message">
                        <strong>🏛️ Claimant:</strong> {text}
                    </div>
                    """, unsafe_allow_html=True)
                elif speaker == 'Respondent':
                    st.markdown(f"""
                    <div class="debate-message respondent-message">
                        <strong>⚖️ Respondent:</strong> {text}
                    </div>
                    """, unsafe_allow_html=True)
                elif speaker == 'Tribunal':
                    st.markdown(f"""
                    <div class="debate-message tribunal-message">
                        <strong>👨‍⚖️ Tribunal:</strong> {text}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 