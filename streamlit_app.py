"""
Streamlit Web Interface for Dynamic Knowledge Graph
================================================

Interactive web application for exploring the legal expert reports knowledge graph.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
from dynamic_knowledge_graph import DynamicKnowledgeGraph

# Page configuration
st.set_page_config(
    page_title="Dynamic Knowledge Graph Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .query-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_knowledge_graph():
    """Load the knowledge graph data."""
    kg = DynamicKnowledgeGraph("/Users/top1/Downloads/Hackathon documents")
    
    # Check if processed data exists
    if os.path.exists("knowledge_graph_output"):
        try:
            # Load pre-processed data
            with open("knowledge_graph_output/entities.json", 'r') as f:
                entities = json.load(f)
            with open("knowledge_graph_output/relationships.json", 'r') as f:
                relationships = json.load(f)
            with open("knowledge_graph_output/arguments.json", 'r') as f:
                arguments = json.load(f)
            with open("knowledge_graph_output/temporal_events.json", 'r') as f:
                temporal_events = json.load(f)
            
            kg.entities = entities
            kg.relationships = relationships
            kg.arguments = arguments
            kg.temporal_events = temporal_events
            kg.build_graph()
            
            return kg
        except:
            pass
    
    # Process documents if no cached data
    kg.process_all_pdfs()
    kg.build_graph()
    kg.save_data()
    
    return kg

def create_network_visualization(kg, selected_entities=None):
    """Create network visualization using Plotly."""
    G = kg.graph
    
    # Filter graph if specific entities selected
    if selected_entities:
        # Create subgraph with selected entities and their neighbors
        nodes_to_include = set(selected_entities)
        for entity in selected_entities:
            if entity in G:
                nodes_to_include.update(G.neighbors(entity))
        G = G.subgraph(nodes_to_include)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    color_map = {
        'PERSON': '#ff6b6b',
        'ORG': '#4ecdc4', 
        'MONEY': '#45b7d1',
        'DATE': '#96ceb4',
        'LEGAL_TERM': '#feca57',
        'PERCENTAGE': '#ff9ff3',
        'CASE': '#54a0ff'
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        node_text.append(f"{node}<br>Type: {node_data.get('type', 'Unknown')}<br>Documents: {len(node_data.get('documents', []))}")
        node_color.append(color_map.get(node_data.get('type', 'OTHER'), '#ddd'))
        node_size.append(10 + len(node_data.get('documents', [])) * 2)
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.split()[-1] if len(node.split()) > 1 else node for node in G.nodes()],
        textposition="middle center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(text='Dynamic Knowledge Graph Network', font=dict(size=16)),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Hover over nodes for details",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left"
                       ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def create_temporal_analysis(kg):
    """Create temporal analysis visualization."""
    temporal_events = kg.temporal_events
    
    if not temporal_events:
        return None
    
    # Convert to DataFrame
    df_data = []
    for event in temporal_events:
        if 'date' in event and event['date']:
            try:
                date_obj = datetime.fromisoformat(str(event['date']).replace('Z', '+00:00'))
                df_data.append({
                    'date': date_obj,
                    'document': event.get('source_document', 'Unknown'),
                    'context': event.get('context', '')[:100] + '...',
                    'year': date_obj.year
                })
            except:
                continue
    
    if not df_data:
        return None
    
    df = pd.DataFrame(df_data)
    
    # Create timeline
    fig = px.scatter(df, 
                    x='date', 
                    y='document',
                    hover_data=['context'],
                    title='Temporal Events Timeline',
                    height=400)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Document"
    )
    
    return fig

def create_entity_analysis(kg):
    """Create entity analysis charts."""
    # Entity type distribution
    entity_types = defaultdict(int)
    for entities in kg.entities.values():
        for entity_type, entity_list in entities.items():
            entity_types[entity_type] += len(entity_list)
    
    # Create pie chart
    fig_pie = px.pie(
        values=list(entity_types.values()),
        names=list(entity_types.keys()),
        title="Entity Type Distribution"
    )
    
    # Flow categories
    flow_categories = Counter(rel['flow_category'] for rel in kg.relationships)
    
    fig_flow = px.bar(
        x=list(flow_categories.keys()),
        y=list(flow_categories.values()),
        title="Relationship Flow Categories"
    )
    
    return fig_pie, fig_flow

def create_argument_analysis(kg):
    """Create argument analysis visualization."""
    if not kg.arguments:
        return None
    
    # Argument types
    arg_types = Counter(arg['argument_type'] for arg in kg.arguments)
    
    fig = px.bar(
        x=list(arg_types.keys()),
        y=list(arg_types.values()),
        title="Argument Types Distribution",
        color=list(arg_types.keys()),
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🕸️ Dynamic Knowledge Graph Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Analysis of Legal Expert Reports")
    
    # Load data
    with st.spinner("Loading knowledge graph... This may take a moment."):
        kg = load_knowledge_graph()
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Summary metrics
    st.sidebar.markdown("### 📊 Summary")
    st.sidebar.metric("Documents", len(kg.documents))
    st.sidebar.metric("Total Entities", sum(len(entities) for entities in kg.entities.values()))
    st.sidebar.metric("Relationships", len(kg.relationships))
    st.sidebar.metric("Arguments", len(kg.arguments))
    st.sidebar.metric("Graph Nodes", kg.graph.number_of_nodes())
    st.sidebar.metric("Graph Edges", kg.graph.number_of_edges())
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query Explorer", "🕸️ Network View", "📈 Analytics", "⏰ Temporal Analysis", "💬 Arguments"])
    
    with tab1:
        st.header("Query the Knowledge Graph")
        
        # Query input
        query = st.text_input("Enter your query:", placeholder="e.g., damages calculation, Jeffrey Cohen opinion, Uruguay arbitration")
        
        if query:
            with st.spinner("Searching..."):
                results = kg.query_graph(query)
            
            st.markdown(f'<div class="query-box"><strong>Query:</strong> "{query}"</div>', unsafe_allow_html=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Relevant Entities")
                if results['entities']:
                    for entity in results['entities'][:10]:
                        with st.expander(f"{entity['name']} ({entity.get('type', 'Unknown')})"):
                            st.write(f"**Type:** {entity.get('type', 'Unknown')}")
                            st.write(f"**Documents:** {', '.join(entity.get('documents', []))}")
                else:
                    st.info("No relevant entities found.")
            
            with col2:
                st.subheader("🔗 Relevant Relationships")
                if results['relationships']:
                    for i, rel in enumerate(results['relationships'][:5]):
                        with st.expander(f"Relationship {i+1}: {rel['source']} → {rel['target']}"):
                            st.write(f"**Type:** {rel['relationship_type']}")
                            st.write(f"**Flow:** {rel['flow_category']}")
                            st.write(f"**Context:** {rel['context'][:200]}...")
                            st.write(f"**Document:** {rel['source_document']}")
                else:
                    st.info("No relevant relationships found.")
            
            # Arguments section
            if results['arguments']:
                st.subheader("💬 Relevant Arguments")
                for i, arg in enumerate(results['arguments'][:5]):
                    with st.expander(f"Argument {i+1} ({arg['argument_type']})"):
                        st.write(f"**Sentence:** {arg['sentence']}")
                        st.write(f"**Entities:** {', '.join(arg['entities'])}")
                        st.write(f"**Strength:** {arg['strength']:.2f}")
                        st.write(f"**Document:** {arg['source_document']}")
    
    with tab2:
        st.header("Network Visualization")
        
        # Entity selection for filtering
        all_entities = list(kg.graph.nodes())
        selected_entities = st.multiselect(
            "Filter by specific entities (leave empty to show all):",
            options=all_entities,
            default=[]
        )
        
        # Create and display network
        if st.button("Generate Network Visualization") or selected_entities:
            with st.spinner("Creating network visualization..."):
                fig = create_network_visualization(kg, selected_entities if selected_entities else None)
                st.plotly_chart(fig, use_container_width=True)
        
        # Top connected entities
        st.subheader("🏆 Most Connected Entities")
        node_degrees = dict(kg.graph.degree())
        top_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (entity, degree) in enumerate(top_entities, 1):
            entity_type = kg.graph.nodes[entity].get('type', 'Unknown')
            st.write(f"{i}. **{entity}** ({entity_type}) - {degree} connections")
    
    with tab3:
        st.header("Knowledge Graph Analytics")
        
        # Entity analysis
        fig_pie, fig_flow = create_entity_analysis(kg)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_flow, use_container_width=True)
        
        # Document analysis
        st.subheader("📄 Document Analysis")
        doc_data = []
        for doc_name, entities in kg.entities.items():
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            doc_data.append({
                'Document': doc_name,
                'Total Entities': total_entities,
                'People': len(entities.get('PERSON', [])),
                'Organizations': len(entities.get('ORG', [])),
                'Legal Terms': len(entities.get('LEGAL_TERM', [])),
                'Monetary Values': len(entities.get('MONEY', []))
            })
        
        df_docs = pd.DataFrame(doc_data)
        st.dataframe(df_docs, use_container_width=True)
    
    with tab4:
        st.header("Temporal Analysis")
        
        fig_temporal = create_temporal_analysis(kg)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
        else:
            st.info("No temporal data available for visualization.")
        
        # Show temporal events table
        if kg.temporal_events:
            st.subheader("📅 Temporal Events")
            temporal_data = []
            for event in kg.temporal_events:
                temporal_data.append({
                    'Date': event.get('date_string', 'Unknown'),
                    'Document': event.get('source_document', 'Unknown'),
                    'Context': event.get('context', '')[:100] + '...'
                })
            
            df_temporal = pd.DataFrame(temporal_data)
            st.dataframe(df_temporal, use_container_width=True)
    
    with tab5:
        st.header("Argument Analysis")
        
        if kg.arguments:
            # Argument type distribution
            fig_args = create_argument_analysis(kg)
            if fig_args:
                st.plotly_chart(fig_args, use_container_width=True)
            
            # Argument details
            st.subheader("💬 Argument Details")
            
            # Filter by argument type
            arg_types = list(set(arg['argument_type'] for arg in kg.arguments))
            selected_arg_type = st.selectbox("Filter by argument type:", ['All'] + arg_types)
            
            filtered_args = kg.arguments
            if selected_arg_type != 'All':
                filtered_args = [arg for arg in kg.arguments if arg['argument_type'] == selected_arg_type]
            
            for i, arg in enumerate(filtered_args[:20]):  # Show first 20
                with st.expander(f"Argument {i+1}: {arg['argument_type']} (Strength: {arg['strength']:.2f})"):
                    st.write(f"**Sentence:** {arg['sentence']}")
                    st.write(f"**Entities:** {', '.join(arg['entities']) if arg['entities'] else 'None'}")
                    st.write(f"**Document:** {arg['source_document']}")
        else:
            st.info("No arguments found in the processed documents.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dynamic Knowledge Graph System for Legal Expert Reports Analysis*")

if __name__ == "__main__":
    main() 