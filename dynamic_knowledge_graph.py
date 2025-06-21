"""
Dynamic Knowledge Graph System for Legal Expert Reports
=====================================================

A comprehensive system for creating and analyzing dynamic knowledge graphs
from legal expert reports, focusing on entities, relationships, and temporal analysis.
"""

import os
import re
import json
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  PDF processing libraries not available. Some features may be limited.")
    PDF_AVAILABLE = False

# NLP processing
try:
    from textblob import TextBlob
    NLP_BASIC = True
except ImportError:
    print("⚠️  Basic NLP library not available. Using regex-based extraction.")
    NLP_BASIC = False

try:
    import spacy
    NLP_ADVANCED = True
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️  Spacy English model not found. Using basic NLP only.")
        NLP_ADVANCED = False
        nlp_model = None
except ImportError:
    print("⚠️  Advanced NLP library not available. Using basic processing.")
    NLP_ADVANCED = False
    nlp_model = None

# Visualization
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from pyvis.network import Network
    VIZ_AVAILABLE = True
except ImportError:
    print("⚠️  Visualization libraries not available. Text-based output only.")
    VIZ_AVAILABLE = False

class EntityExtractor:
    """Extract entities from text using available NLP techniques."""
    
    def __init__(self):
        self.nlp_model = nlp_model if NLP_ADVANCED else None
        
        # Legal domain specific patterns
        self.legal_patterns = {
            'monetary': r'\$[\d,]+(?:\.\d{2})?|\d+\s*(?:million|billion|thousand)\s*(?:dollars?|USD|usd)',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            'legal_terms': r'\b(?:damages|liability|breach|contract|agreement|settlement|arbitration|tribunal|award|compensation|expert|report|opinion|analysis|calculation|methodology|valuation)\b',
            'companies': r'\b[A-Z][a-zA-Z\s&,.-]*(?:Inc|Corp|LLC|Ltd|Company|Co\.)\b',
            'percentages': r'\d+(?:\.\d+)?%',
            'case_citations': r'\b\w+\s+v\.?\s+\w+\b',
            'experts': r'\b(?:Dr\.?|Professor|Mr\.?|Ms\.?|Mrs\.?)\s+[A-Z][a-zA-Z\s]+\b',
            'countries': r'\b(?:Uruguay|Argentina|United States|USA|UK|Canada|Brazil|Spain|France|Germany)\b'
        }
        
        # Common expert/professional names patterns
        self.expert_patterns = [
            r'\b(?:Jeffrey|Jeff)\s+Cohen\b',
            r'\bKaczmarek\b',
            r'\bSequeira\b',
            r'\bBrent\s+(?:C\.?)?\s*Kaczmarek\b',
            r'\bKiran\s+(?:P\.?)?\s*Sequeira\b'
        ]
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract various types of entities from text."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'MONEY': [],
            'DATE': [],
            'LEGAL_TERM': [],
            'PERCENTAGE': [],
            'CASE': [],
            'EXPERT': [],
            'COUNTRY': []
        }
        
        # Use spaCy for standard entity recognition if available
        if self.nlp_model:
            try:
                doc = self.nlp_model(text[:1000000])  # Limit text size
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'MONEY', 'DATE', 'GPE']:
                        label = 'COUNTRY' if ent.label_ == 'GPE' else ent.label_
                        entities[label].append(ent.text.strip())
            except Exception as e:
                print(f"⚠️  SpaCy processing error: {e}")
        
        # Use regex patterns for domain-specific entities
        for pattern_name, pattern in self.legal_patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if pattern_name == 'monetary':
                    entities['MONEY'].extend(matches)
                elif pattern_name == 'dates':
                    entities['DATE'].extend(matches)
                elif pattern_name == 'legal_terms':
                    entities['LEGAL_TERM'].extend(matches)
                elif pattern_name == 'companies':
                    entities['ORG'].extend(matches)
                elif pattern_name == 'percentages':
                    entities['PERCENTAGE'].extend(matches)
                elif pattern_name == 'case_citations':
                    entities['CASE'].extend(matches)
                elif pattern_name == 'experts':
                    entities['EXPERT'].extend(matches)
                elif pattern_name == 'countries':
                    entities['COUNTRY'].extend(matches)
            except Exception as e:
                print(f"⚠️  Pattern matching error for {pattern_name}: {e}")
        
        # Extract known expert names
        for expert_pattern in self.expert_patterns:
            try:
                matches = re.findall(expert_pattern, text, re.IGNORECASE)
                entities['EXPERT'].extend(matches)
            except Exception as e:
                print(f"⚠️  Expert pattern error: {e}")
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([e.strip() for e in entities[key] if e.strip() and len(e.strip()) > 2]))
        
        return entities

class RelationshipExtractor:
    """Extract relationships between entities based on context and co-occurrence."""
    
    def __init__(self):
        self.relationship_types = {
            'financial_flow': ['paid', 'owes', 'invested', 'transferred', 'received', 'costs', 'damages', 'compensation', 'loss', 'value'],
            'information_flow': ['reported', 'disclosed', 'informed', 'communicated', 'stated', 'testified', 'opined', 'concluded', 'found'],
            'physical_flow': ['delivered', 'shipped', 'transported', 'provided', 'supplied']
        }
        
        # Argument indicators
        self.argument_indicators = {
            'support': ['supports', 'agrees with', 'confirms', 'validates', 'corroborates', 'consistent with'],
            'oppose': ['contradicts', 'disputes', 'challenges', 'refutes', 'disagrees with', 'inconsistent with'],
            'neutral': ['mentions', 'discusses', 'references', 'notes', 'observes', 'considers']
        }
    
    def extract_relationships(self, text: str, entities: Dict[str, List[str]], max_sentences: int = 1000) -> List[Dict]:
        """Extract relationships between entities from text."""
        relationships = []
        sentences = self._split_sentences(text)[:max_sentences]  # Limit processing
        
        for sentence in sentences:
            try:
                # Find entities in this sentence
                sentence_entities = self._find_entities_in_sentence(sentence, entities)
                
                if len(sentence_entities) >= 2:
                    # Extract relationships between pairs of entities
                    for i, entity1 in enumerate(sentence_entities):
                        for entity2 in sentence_entities[i+1:]:
                            rel_type = self._classify_relationship(sentence, entity1, entity2)
                            if rel_type:
                                relationships.append({
                                    'source': entity1,
                                    'target': entity2,
                                    'relationship_type': rel_type,
                                    'context': sentence,
                                    'flow_category': self._get_flow_category(sentence),
                                    'argument_stance': self._get_argument_stance(sentence)
                                })
            except Exception as e:
                print(f"⚠️  Relationship extraction error: {e}")
                continue
        
        return relationships
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    def _find_entities_in_sentence(self, sentence: str, entities: Dict[str, List[str]]) -> List[str]:
        """Find entities present in a sentence."""
        found_entities = []
        sentence_lower = sentence.lower()
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity.lower() in sentence_lower:
                    found_entities.append(entity)
        
        return list(set(found_entities))  # Remove duplicates
    
    def _classify_relationship(self, sentence: str, entity1: str, entity2: str) -> str:
        """Classify the type of relationship between two entities."""
        sentence_lower = sentence.lower()
        
        # Look for explicit relationship indicators
        for rel_type, indicators in self.relationship_types.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    return rel_type
        
        # Default relationship based on proximity
        return 'related_to'
    
    def _get_flow_category(self, sentence: str) -> str:
        """Determine the flow category of a relationship."""
        sentence_lower = sentence.lower()
        
        for flow_type, indicators in self.relationship_types.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    return flow_type
        
        return 'information_flow'  # Default
    
    def _get_argument_stance(self, sentence: str) -> str:
        """Determine the argument stance in a sentence."""
        sentence_lower = sentence.lower()
        
        for stance, indicators in self.argument_indicators.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    return stance
        
        return 'neutral'

class TemporalAnalyzer:
    """Analyze temporal aspects of the knowledge graph."""
    
    def __init__(self):
        self.date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}\b'
        ]
    
    def extract_temporal_info(self, text: str) -> List[Dict]:
        """Extract temporal information from text."""
        temporal_events = []
        
        for pattern in self.date_patterns:
            try:
                matches = re.finditer(pattern, text)
                for match in matches:
                    date_str = match.group()
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        # Get surrounding context
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end]
                        
                        temporal_events.append({
                            'date': parsed_date,
                            'date_string': date_str,
                            'context': context.strip(),
                            'position': match.start()
                        })
            except Exception as e:
                print(f"⚠️  Temporal extraction error: {e}")
                continue
        
        return sorted(temporal_events, key=lambda x: x['date'])
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        try:
            # Try various date formats
            formats = [
                '%B %d, %Y',
                '%B %d %Y',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Try to extract year at least
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return datetime(int(year_match.group()), 1, 1)
                
        except Exception as e:
            print(f"⚠️  Date parsing error: {e}")
            
        return None

class ArgumentChainAnalyzer:
    """Analyze argument chains and debates in the documents."""
    
    def __init__(self):
        self.argument_keywords = [
            'argues', 'contends', 'claims', 'asserts', 'maintains', 'alleges',
            'disputes', 'challenges', 'refutes', 'contradicts', 'disagrees',
            'supports', 'confirms', 'validates', 'corroborates', 'agrees',
            'concludes', 'finds', 'determines', 'opines', 'believes'
        ]
    
    def extract_arguments(self, text: str, entities: Dict[str, List[str]], max_sentences: int = 500) -> List[Dict]:
        """Extract argument chains from text."""
        arguments = []
        sentences = re.split(r'[.!?]+', text)[:max_sentences]  # Limit processing
        
        for i, sentence in enumerate(sentences):
            try:
                if any(keyword in sentence.lower() for keyword in self.argument_keywords):
                    # This sentence contains an argument
                    argument_entities = self._find_entities_in_sentence(sentence, entities)
                    
                    if argument_entities:
                        arguments.append({
                            'sentence': sentence.strip(),
                            'entities': argument_entities,
                            'argument_type': self._classify_argument_type(sentence),
                            'position': i,
                            'strength': self._assess_argument_strength(sentence)
                        })
            except Exception as e:
                print(f"⚠️  Argument extraction error: {e}")
                continue
        
        return arguments
    
    def _find_entities_in_sentence(self, sentence: str, entities: Dict[str, List[str]]) -> List[str]:
        """Find entities present in a sentence."""
        found_entities = []
        sentence_lower = sentence.lower()
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity.lower() in sentence_lower:
                    found_entities.append(entity)
        
        return list(set(found_entities))
    
    def _classify_argument_type(self, sentence: str) -> str:
        """Classify the type of argument."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['disputes', 'challenges', 'refutes', 'contradicts']):
            return 'counter_argument'
        elif any(word in sentence_lower for word in ['supports', 'confirms', 'validates', 'corroborates']):
            return 'supporting_argument'
        else:
            return 'primary_argument'
    
    def _assess_argument_strength(self, sentence: str) -> float:
        """Assess the strength of an argument based on linguistic cues."""
        strength_indicators = {
            'strong': ['clearly', 'definitively', 'conclusively', 'undoubtedly', 'certainly'],
            'moderate': ['likely', 'probably', 'suggests', 'indicates', 'appears'],
            'weak': ['possibly', 'might', 'could', 'may', 'perhaps']
        }
        
        sentence_lower = sentence.lower()
        
        for level, indicators in strength_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                if level == 'strong':
                    return 0.9
                elif level == 'moderate':
                    return 0.6
                else:
                    return 0.3
        
        return 0.5  # Default moderate strength

class DynamicKnowledgeGraph:
    """Main class for creating and managing the dynamic knowledge graph."""
    
    def __init__(self, pdf_folder_path: str):
        self.pdf_folder_path = pdf_folder_path
        self.graph = nx.MultiDiGraph()
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.temporal_analyzer = TemporalAnalyzer()
        self.argument_analyzer = ArgumentChainAnalyzer()
        
        # Data storage
        self.documents = {}
        self.entities = {}
        self.relationships = []
        self.temporal_events = []
        self.arguments = []
        
    def process_all_pdfs(self):
        """Process all PDF files in the folder."""
        if not PDF_AVAILABLE:
            print("❌ PDF processing libraries not available. Cannot process PDF files.")
            return
            
        pdf_files = [f for f in os.listdir(self.pdf_folder_path) if f.endswith('.pdf')]
        
        print(f"📄 Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            print(f"🔄 Processing {pdf_file}...")
            file_path = os.path.join(self.pdf_folder_path, pdf_file)
            
            try:
                text_content = self._extract_text_from_pdf(file_path)
                if not text_content.strip():
                    print(f"⚠️  No text extracted from {pdf_file}")
                    continue
                    
                self.documents[pdf_file] = {
                    'content': text_content,
                    'file_path': file_path,
                    'processed_date': datetime.now(),
                    'word_count': len(text_content.split())
                }
                
                print(f"   📝 Extracted {len(text_content.split())} words")
                
                # Extract entities
                entities = self.entity_extractor.extract_entities(text_content)
                self.entities[pdf_file] = entities
                
                print(f"   🏷️  Found {sum(len(v) for v in entities.values())} entities")
                
                # Extract relationships
                relationships = self.relationship_extractor.extract_relationships(text_content, entities)
                for rel in relationships:
                    rel['source_document'] = pdf_file
                self.relationships.extend(relationships)
                
                print(f"   🔗 Found {len(relationships)} relationships")
                
                # Extract temporal information
                temporal_events = self.temporal_analyzer.extract_temporal_info(text_content)
                for event in temporal_events:
                    event['source_document'] = pdf_file
                self.temporal_events.extend(temporal_events)
                
                print(f"   📅 Found {len(temporal_events)} temporal events")
                
                # Extract arguments
                arguments = self.argument_analyzer.extract_arguments(text_content, entities)
                for arg in arguments:
                    arg['source_document'] = pdf_file
                self.arguments.extend(arguments)
                
                print(f"   💬 Found {len(arguments)} arguments")
                print(f"✅ Completed {pdf_file}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {str(e)}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""
        
        try:
            # Try with pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"   ⚠️  pdfplumber failed: {e}")
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                print(f"   ❌ PyPDF2 also failed: {e2}")
                return ""
        
        return text.strip()
    
    def build_graph(self):
        """Build the NetworkX graph from extracted data."""
        print("🕸️  Building knowledge graph...")
        
        # Add entity nodes
        for doc_name, entities in self.entities.items():
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if not self.graph.has_node(entity):
                        self.graph.add_node(entity, 
                                          type=entity_type, 
                                          documents=[doc_name],
                                          first_mentioned=doc_name)
                    else:
                        # Update existing node
                        if doc_name not in self.graph.nodes[entity]['documents']:
                            self.graph.nodes[entity]['documents'].append(doc_name)
        
        # Add relationship edges
        for rel in self.relationships:
            source = rel['source']
            target = rel['target']
            
            if self.graph.has_node(source) and self.graph.has_node(target):
                self.graph.add_edge(source, target,
                                  relationship_type=rel['relationship_type'],
                                  flow_category=rel['flow_category'],
                                  argument_stance=rel['argument_stance'],
                                  context=rel['context'][:200] + '...',  # Truncate for storage
                                  source_document=rel['source_document'])
        
        print(f"✅ Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def query_graph(self, query: str) -> Dict[str, Any]:
        """Query the knowledge graph with natural language."""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = {
            'entities': [],
            'relationships': [],
            'arguments': [],
            'temporal_events': []
        }
        
        # Search for relevant entities
        for node in self.graph.nodes():
            if any(word in node.lower() for word in query_words):
                node_data = dict(self.graph.nodes[node])
                node_data['name'] = node
                results['entities'].append(node_data)
        
        # Search for relevant relationships
        for rel in self.relationships:
            if any(word in rel['context'].lower() for word in query_words):
                results['relationships'].append(rel)
        
        # Search for relevant arguments
        for arg in self.arguments:
            if any(word in arg['sentence'].lower() for word in query_words):
                results['arguments'].append(arg)
        
        # Search for relevant temporal events
        for event in self.temporal_events:
            if any(word in event['context'].lower() for word in query_words):
                results['temporal_events'].append(event)
        
        return results
    
    def create_visualization(self, output_file: str = "knowledge_graph.html"):
        """Create an interactive visualization of the knowledge graph."""
        if not VIZ_AVAILABLE:
            print("⚠️  Visualization libraries not available. Skipping visualization.")
            return
            
        try:
            net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Color mapping for different entity types
            color_map = {
                'PERSON': '#ff6b6b',
                'ORG': '#4ecdc4',
                'MONEY': '#45b7d1',
                'DATE': '#96ceb4',
                'LEGAL_TERM': '#feca57',
                'PERCENTAGE': '#ff9ff3',
                'CASE': '#54a0ff',
                'EXPERT': '#ff9999',
                'COUNTRY': '#99ff99'
            }
            
            # Add nodes
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                color = color_map.get(node_data.get('type', 'OTHER'), '#ddd')
                
                net.add_node(node, 
                           label=node, 
                           color=color,
                           title=f"Type: {node_data.get('type', 'Unknown')}\nDocuments: {', '.join(node_data.get('documents', []))}")
            
            # Add edges
            for edge in self.graph.edges(data=True):
                source, target, data = edge
                net.add_edge(source, target, 
                           title=f"Relationship: {data.get('relationship_type', 'Unknown')}\nFlow: {data.get('flow_category', 'Unknown')}")
            
            net.save_graph(output_file)
            print(f"📊 Interactive visualization saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = f"""
# 🕸️ Dynamic Knowledge Graph Analysis Report

## 📊 Summary Statistics
- **Total Documents Processed**: {len(self.documents)}
- **Total Entities Extracted**: {sum(len(entities) for entities_dict in self.entities.values() for entities in entities_dict.values())}
- **Total Relationships**: {len(self.relationships)}
- **Total Arguments**: {len(self.arguments)}
- **Total Temporal Events**: {len(self.temporal_events)}
- **Graph Nodes**: {self.graph.number_of_nodes()}
- **Graph Edges**: {self.graph.number_of_edges()}

## 📄 Document Analysis
"""
        
        for doc_name, doc_data in self.documents.items():
            word_count = doc_data.get('word_count', 0)
            report += f"- **{doc_name}**: {word_count:,} words\n"
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for entities_dict in self.entities.values():
            for entity_type, entity_list in entities_dict.items():
                entity_types[entity_type] += len(entity_list)
        
        report += "\n## 🏷️ Entity Type Distribution\n"
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{entity_type}**: {count}\n"
        
        # Relationship analysis
        report += "\n## 🔗 Relationship Analysis\n"
        flow_categories = Counter(rel['flow_category'] for rel in self.relationships)
        report += "\n### Flow Categories:\n"
        for category, count in flow_categories.most_common():
            report += f"- **{category}**: {count}\n"
        
        # Argument analysis
        report += "\n## 💬 Argument Analysis\n"
        argument_types = Counter(arg['argument_type'] for arg in self.arguments)
        report += "\n### Argument Types:\n"
        for arg_type, count in argument_types.most_common():
            report += f"- **{arg_type}**: {count}\n"
        
        # Top entities by connections
        if self.graph.number_of_nodes() > 0:
            report += "\n## 🏆 Key Entities (Most Connected)\n"
            node_degrees = dict(self.graph.degree())
            top_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for entity, degree in top_entities:
                entity_type = self.graph.nodes[entity].get('type', 'Unknown')
                report += f"- **{entity}** ({entity_type}): {degree} connections\n"
        
        # Expert analysis
        experts = set()
        for entities_dict in self.entities.values():
            experts.update(entities_dict.get('EXPERT', []))
        
        if experts:
            report += f"\n## 👨‍💼 Identified Experts\n"
            for expert in sorted(experts):
                report += f"- {expert}\n"
        
        return report
    
    def save_data(self, output_dir: str = "knowledge_graph_output"):
        """Save all extracted data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save entities
            with open(os.path.join(output_dir, "entities.json"), 'w') as f:
                json.dump(self.entities, f, indent=2, default=str)
            
            # Save relationships
            with open(os.path.join(output_dir, "relationships.json"), 'w') as f:
                json.dump(self.relationships, f, indent=2, default=str)
            
            # Save arguments
            with open(os.path.join(output_dir, "arguments.json"), 'w') as f:
                json.dump(self.arguments, f, indent=2, default=str)
            
            # Save temporal events
            with open(os.path.join(output_dir, "temporal_events.json"), 'w') as f:
                json.dump(self.temporal_events, f, indent=2, default=str)
            
            # Save graph as GraphML
            nx.write_graphml(self.graph, os.path.join(output_dir, "knowledge_graph.graphml"))
            
            # Save report
            report = self.generate_report()
            with open(os.path.join(output_dir, "analysis_report.md"), 'w') as f:
                f.write(report)
            
            print(f"💾 All data saved to {output_dir}/")
            
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")

def main():
    """Main function to run the dynamic knowledge graph system."""
    
    print("=" * 70)
    print("🕸️  DYNAMIC KNOWLEDGE GRAPH SYSTEM")
    print("   For Legal Expert Reports Analysis")
    print("=" * 70)
    
    # Initialize the system
    pdf_folder = "/Users/top1/Downloads/Hackathon documents"
    kg = DynamicKnowledgeGraph(pdf_folder)
    
    print("🚀 Processing legal expert reports...")
    
    # Process all PDFs
    kg.process_all_pdfs()
    
    # Build the graph
    kg.build_graph()
    
    # Create visualization
    kg.create_visualization("dynamic_knowledge_graph.html")
    
    # Save all data
    kg.save_data()
    
    # Generate and display report
    report = kg.generate_report()
    print("\n" + "="*70)
    print(report)
    
    # Interactive query example
    print("\n" + "="*70)
    print("🔍 SAMPLE QUERY DEMONSTRATIONS")
    print("="*70)
    
    sample_queries = [
        "Jeffrey Cohen expert opinion",
        "damages calculation methodology", 
        "Uruguay arbitration case",
        "financial losses compensation",
        "Kaczmarek analysis"
    ]
    
    for query in sample_queries:
        print(f"\n🔎 Query: '{query}'")
        try:
            results = kg.query_graph(query)
            print(f"   📋 Found {len(results['entities'])} entities, {len(results['relationships'])} relationships")
            
            # Show top entities
            if results['entities']:
                print("   🏷️  Top entities:")
                for entity in results['entities'][:3]:
                    print(f"      - {entity['name']} ({entity.get('type', 'Unknown')})")
                    
            # Show top relationships
            if results['relationships']:
                print("   🔗 Top relationships:")
                for rel in results['relationships'][:2]:
                    print(f"      - {rel['source']} → {rel['target']} ({rel['relationship_type']})")
                    
        except Exception as e:
            print(f"   ❌ Query error: {e}")
    
    print("\n" + "="*70)
    print("✅ KNOWLEDGE GRAPH ANALYSIS COMPLETE!")
    print("   📊 Check 'knowledge_graph_output/' for detailed results")
    print("   🌐 Open 'dynamic_knowledge_graph.html' for interactive visualization")
    print("="*70)

if __name__ == "__main__":
    main()