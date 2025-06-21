"""
Dynamic Knowledge Graph Demonstration
====================================

This script demonstrates the key analysis capabilities of the Dynamic Knowledge Graph
for the Simmons and Simmons Challenge.
"""

import json
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx

def load_knowledge_graph_data():
    """Load the processed knowledge graph data."""
    print("🔄 Loading knowledge graph data...")
    
    # Load entities
    with open('knowledge_graph_output/entities.json', 'r') as f:
        entities = json.load(f)
    
    # Load relationships
    with open('knowledge_graph_output/relationships.json', 'r') as f:
        relationships = json.load(f)
    
    # Load arguments
    with open('knowledge_graph_output/arguments.json', 'r') as f:
        arguments = json.load(f)
    
    # Load temporal events
    with open('knowledge_graph_output/temporal_events.json', 'r') as f:
        temporal_events = json.load(f)
    
    print("✅ Knowledge graph data loaded successfully!")
    return entities, relationships, arguments, temporal_events

def analyze_expert_opinions(entities, relationships, arguments):
    """Analyze expert opinions and their positions."""
    print("\n🧑‍⚖️ EXPERT OPINION ANALYSIS")
    print("=" * 50)
    
    # Extract all identified experts
    all_experts = set()
    for doc_entities in entities.values():
        all_experts.update(doc_entities.get('EXPERT', []))
        all_experts.update(doc_entities.get('PERSON', []))
    
    # Filter to known expert names
    known_experts = [expert for expert in all_experts if any(name in expert.lower() for name in ['cohen', 'kaczmarek', 'sequeira'])]
    
    print(f"📋 Identified Key Experts:")
    for expert in sorted(known_experts)[:10]:
        print(f"   • {expert}")
    
    # Analyze arguments by expert
    expert_arguments = defaultdict(list)
    for arg in arguments:
        for entity in arg.get('entities', []):
            if any(name in entity.lower() for name in ['cohen', 'kaczmarek', 'sequeira']):
                expert_arguments[entity].append(arg)
    
    print(f"\n💭 Expert Arguments Found:")
    for expert, args in expert_arguments.items():
        if len(args) > 0:
            print(f"   • {expert}: {len(args)} arguments")
            for arg in args[:2]:  # Show first 2 arguments
                print(f"     - {arg['argument_type']}: {arg['sentence'][:100]}...")

def analyze_financial_relationships(relationships, entities):
    """Analyze financial flows and monetary relationships."""
    print("\n💰 FINANCIAL FLOW ANALYSIS")
    print("=" * 50)
    
    # Filter financial relationships
    financial_rels = [rel for rel in relationships if rel['flow_category'] == 'financial_flow']
    
    print(f"📊 Found {len(financial_rels)} financial relationships")
    
    # Analyze financial relationship types
    fin_types = Counter(rel['relationship_type'] for rel in financial_rels)
    print(f"\n💸 Financial Relationship Types:")
    for rel_type, count in fin_types.most_common(5):
        print(f"   • {rel_type}: {count}")
    
    # Extract monetary values mentioned
    monetary_entities = set()
    for doc_entities in entities.values():
        monetary_entities.update(doc_entities.get('MONEY', []))
    
    print(f"\n💵 Key Monetary Values Mentioned:")
    for money in sorted(monetary_entities)[:10]:
        if '$' in money or 'million' in money.lower() or 'billion' in money.lower():
            print(f"   • {money}")

def analyze_temporal_patterns(temporal_events):
    """Analyze temporal patterns in the documents."""
    print("\n📅 TEMPORAL ANALYSIS")
    print("=" * 50)
    
    print(f"📊 Found {len(temporal_events)} temporal events")
    
    # Extract years from temporal events
    years = []
    for event in temporal_events:
        date_str = event.get('date_string', '')
        try:
            if len(date_str) == 4 and date_str.isdigit():
                years.append(int(date_str))
            elif any(year in date_str for year in ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']):
                for year in range(2010, 2021):
                    if str(year) in date_str:
                        years.append(year)
                        break
        except:
            continue
    
    if years:
        year_counts = Counter(years)
        print(f"\n📈 Most Mentioned Years:")
        for year, count in year_counts.most_common(10):
            print(f"   • {year}: {count} mentions")

def analyze_key_debates(relationships, arguments):
    """Analyze key debates and disagreements."""
    print("\n⚔️ KEY DEBATES ANALYSIS")
    print("=" * 50)
    
    # Find opposing relationships
    opposing_rels = [rel for rel in relationships if rel['argument_stance'] == 'oppose']
    supporting_rels = [rel for rel in relationships if rel['argument_stance'] == 'support']
    
    print(f"📊 Found {len(opposing_rels)} opposing relationships")
    print(f"📊 Found {len(supporting_rels)} supporting relationships")
    
    # Analyze argument types
    arg_types = Counter(arg['argument_type'] for arg in arguments)
    print(f"\n💬 Argument Type Distribution:")
    for arg_type, count in arg_types.items():
        print(f"   • {arg_type}: {count}")
    
    # Find key debate topics
    debate_topics = defaultdict(int)
    for rel in opposing_rels:
        for word in rel['context'].lower().split():
            if word in ['damages', 'valuation', 'calculation', 'methodology', 'expert', 'opinion', 'analysis']:
                debate_topics[word] += 1
    
    print(f"\n🔥 Key Debate Topics:")
    for topic, count in Counter(debate_topics).most_common(5):
        print(f"   • {topic}: {count} mentions in opposing contexts")

def generate_query_examples(entities, relationships):
    """Generate example queries and their results."""
    print("\n🔍 QUERY EXAMPLES")
    print("=" * 50)
    
    # Example query function
    def query_graph(query_terms):
        results = {
            'entities': [],
            'relationships': []
        }
        
        query_lower = [term.lower() for term in query_terms]
        
        # Find relevant entities
        for doc_entities in entities.values():
            for entity_type, entity_list in doc_entities.items():
                for entity in entity_list:
                    if any(term in entity.lower() for term in query_lower):
                        results['entities'].append((entity, entity_type))
        
        # Find relevant relationships
        for rel in relationships:
            if any(term in rel['context'].lower() for term in query_lower):
                results['relationships'].append(rel)
        
        return results
    
    # Example queries
    queries = [
        ["damages", "calculation"],
        ["Jeffrey", "Cohen", "expert"],
        ["Uruguay", "arbitration"],
        ["valuation", "methodology"],
        ["Kaczmarek", "analysis"]
    ]
    
    for query_terms in queries:
        query_str = " ".join(query_terms)
        results = query_graph(query_terms)
        
        print(f"\n🔎 Query: '{query_str}'")
        print(f"   📋 Found {len(set(results['entities']))} unique entities")
        print(f"   🔗 Found {len(results['relationships'])} relationships")
        
        # Show sample entities
        unique_entities = list(set(results['entities']))[:5]
        if unique_entities:
            print(f"   🏷️  Sample entities:")
            for entity, entity_type in unique_entities:
                print(f"      - {entity} ({entity_type})")

def create_summary_statistics(entities, relationships, arguments, temporal_events):
    """Create comprehensive summary statistics."""
    print("\n📊 COMPREHENSIVE SUMMARY")
    print("=" * 70)
    
    # Document statistics
    print(f"📄 Documents Processed: {len(entities)}")
    for doc_name in entities.keys():
        print(f"   • {doc_name}")
    
    # Entity statistics
    total_entities = sum(len(entity_list) for doc_entities in entities.values() for entity_list in doc_entities.values())
    print(f"\n🏷️  Total Entities Extracted: {total_entities:,}")
    
    entity_types = defaultdict(int)
    for doc_entities in entities.values():
        for entity_type, entity_list in doc_entities.items():
            entity_types[entity_type] += len(entity_list)
    
    print(f"   Entity Type Breakdown:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {entity_type}: {count:,}")
    
    # Relationship statistics
    print(f"\n🔗 Total Relationships: {len(relationships):,}")
    flow_categories = Counter(rel['flow_category'] for rel in relationships)
    print(f"   Flow Category Breakdown:")
    for category, count in flow_categories.most_common():
        print(f"   • {category}: {count:,}")
    
    # Argument statistics
    print(f"\n💬 Total Arguments: {len(arguments)}")
    
    # Temporal statistics
    print(f"\n📅 Total Temporal Events: {len(temporal_events):,}")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"🎯 This knowledge graph demonstrates:")
    print(f"   • Systematic entity extraction from legal documents")
    print(f"   • Dynamic relationship mapping with flow categorization")
    print(f"   • Temporal analysis and timeline construction")
    print(f"   • Argument chain analysis and debate mapping")
    print(f"   • Interactive querying capabilities")

def main():
    """Run the comprehensive demonstration."""
    print("🕸️  DYNAMIC KNOWLEDGE GRAPH DEMONSTRATION")
    print("   For Simmons and Simmons Challenge")
    print("=" * 70)
    
    # Load data
    entities, relationships, arguments, temporal_events = load_knowledge_graph_data()
    
    # Run analyses
    analyze_expert_opinions(entities, relationships, arguments)
    analyze_financial_relationships(relationships, entities)
    analyze_temporal_patterns(temporal_events)
    analyze_key_debates(relationships, arguments)
    generate_query_examples(entities, relationships)
    create_summary_statistics(entities, relationships, arguments, temporal_events)

if __name__ == "__main__":
    main() 