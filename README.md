# 🕸️ Dynamic Knowledge Graph for Legal Expert Reports

A comprehensive **systematic dynamic knowledge graph system** built in Python to analyze legal expert reports, extract entities, relationships, and arguments, and provide interactive exploration capabilities.

## 🎯 Challenge Overview

This system addresses the **Simmons and Simmons Challenge** for building an evidence mapping solution that connects key entities, concepts, and positions from across multiple expert reports in litigation. It enables legal teams to quickly understand what each expert says about specific topics and how their views relate, support, or contradict each other.

## ✨ Key Features

### 🔍 **Systematic Entity Extraction**
- **People (PERSON)**: Expert names, legal professionals, witnesses
- **Organizations (ORG)**: Companies, institutions, legal entities
- **Legal Terms**: Damages, liability, breach, contracts, settlements
- **Financial Information (MONEY)**: Monetary values, compensation amounts
- **Temporal Data (DATE)**: Important dates and timelines
- **Legal Cases**: Case citations and references
- **Percentages**: Statistical data and percentages

### 🔗 **Dynamic Relationship Mapping**
- **Financial Flow**: Payment, investment, transfer relationships
- **Information Flow**: Communication, reporting, disclosure relationships
- **Physical Flow**: Delivery, transportation, supply relationships
- **Argument Stances**: Support, opposition, neutral positions

### ⏰ **Temporal Analysis**
- Timeline visualization of events
- Chronological tracking of entity mentions
- Temporal relationship mapping

### 💬 **Argument Chain Analysis**
- **Primary Arguments**: Main claims and positions
- **Supporting Arguments**: Evidence and corroborations
- **Counter Arguments**: Disputes and contradictions
- **Argument Strength Assessment**: Linguistic confidence analysis

### 📊 **Interactive Exploration**
- **Natural Language Queries**: "Who discusses damages?" "Which experts disagree on valuation?"
- **Network Visualizations**: Interactive graph representations
- **Debate Mapping**: Dynamic back-and-forth argument analysis
- **Analytical Dashboards**: Comprehensive statistics and insights

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Spacy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Additional Dependencies (Optional)

For enhanced date parsing:
```bash
pip install python-dateutil
```

## 📁 File Structure

```
Hackathon documents/
├── dynamic_knowledge_graph.py    # Main knowledge graph system
├── streamlit_app.py              # Interactive web interface
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── knowledge_graph_output/       # Generated output directory
│   ├── entities.json
│   ├── relationships.json
│   ├── arguments.json
│   ├── temporal_events.json
│   ├── knowledge_graph.graphml
│   └── analysis_report.md
└── PDF files to be analyzed...
```

## 🏃‍♂️ Usage

### Command Line Interface

```bash
python dynamic_knowledge_graph.py
```

This will:
1. Process all PDF files in the directory
2. Extract entities, relationships, and arguments
3. Build the dynamic knowledge graph
4. Generate interactive visualization (`dynamic_knowledge_graph.html`)
5. Save all data to `knowledge_graph_output/`
6. Display comprehensive analysis report

### Interactive Web Interface

```bash
streamlit run streamlit_app.py
```

Access the web interface at `http://localhost:8501` for:
- 🔍 **Query Explorer**: Natural language search
- 🕸️ **Network View**: Interactive graph visualization
- 📈 **Analytics**: Statistical analysis and charts
- ⏰ **Temporal Analysis**: Timeline visualizations
- 💬 **Arguments**: Argument mapping and analysis

## 🔍 Query Examples

The system supports sophisticated queries like:

### Expert Opinion Queries
- "What does Jeffrey Cohen say about damages?"
- "Which experts discuss valuation methodology?"
- "Who mentions Uruguay arbitration?"

### Disagreement Analysis
- "Which experts disagree on compensation?"
- "What are the opposing views on liability?"
- "Find contradictory positions on market conduct"

### Financial Analysis
- "Show all financial relationships"
- "What are the monetary damages mentioned?"
- "Find payment flows between entities"

### Temporal Queries
- "What happened in 2020?"
- "Show chronological order of events"
- "Find recent expert opinions"

## 🎯 Core Functionality

### 1. **Entity Extractor**
Uses advanced NLP techniques including:
- Spacy Named Entity Recognition
- Legal domain-specific regex patterns
- Fuzzy matching for entity resolution

### 2. **Relationship Extractor**
Identifies relationships through:
- Co-occurrence analysis in sentences
- Pattern matching for relationship types
- Context-based relationship classification

### 3. **Temporal Analyzer**
Extracts temporal information using:
- Multiple date format recognition
- Context-aware date parsing
- Timeline construction

### 4. **Argument Chain Analyzer**
Analyzes arguments through:
- Argument keyword detection
- Stance classification (support/oppose/neutral)
- Strength assessment based on linguistic cues

### 5. **Dynamic Knowledge Graph**
Builds comprehensive graph with:
- NetworkX multi-directed graph structure
- Node attributes for entities
- Edge attributes for relationships
- Interactive query capabilities

## 📊 Output Data

The system generates multiple output formats:

### JSON Files
- `entities.json`: All extracted entities by document
- `relationships.json`: All relationships with context
- `arguments.json`: All arguments with classifications
- `temporal_events.json`: Timeline events with dates

### Graph Formats
- `knowledge_graph.graphml`: NetworkX graph format
- Interactive HTML visualization

### Reports
- `analysis_report.md`: Comprehensive analysis summary
- Statistical breakdowns and insights

## 🔧 Advanced Features

### Custom Entity Types
Easily extend with domain-specific entities:
```python
self.legal_patterns = {
    'custom_pattern': r'your_regex_pattern',
    # Add more patterns as needed
}
```

### Relationship Categories
The system categorizes relationships into three main flows:
- **Financial Flow**: Money, payments, investments
- **Information Flow**: Communications, reports, disclosures
- **Physical Flow**: Deliveries, transfers, supplies

### Argument Stance Detection
Automatically classifies argument stances:
- **Support**: Agreements, confirmations, validations
- **Oppose**: Contradictions, disputes, challenges
- **Neutral**: Mentions, discussions, references

## 🎨 Visualization Features

### Interactive Network Graph
- **Color-coded nodes** by entity type
- **Hover information** with details
- **Filterable views** by entity selection
- **Dynamic layout** with force-directed positioning

### Temporal Visualizations
- **Timeline scatter plots** showing events over time
- **Document-based timelines** for cross-referencing
- **Chronological event sequences**

### Analytics Dashboard
- **Entity type distribution** pie charts
- **Relationship flow** bar charts
- **Argument type** distributions
- **Document statistics** tables

## 🛠️ Technical Architecture

### Core Components

1. **PDFProcessor**: Extracts text from PDF files using PyPDF2 and pdfplumber
2. **NLPPipeline**: Processes text using spaCy and transformers
3. **GraphBuilder**: Constructs NetworkX graph from extracted data
4. **Visualizer**: Creates interactive visualizations using Plotly and Pyvis
5. **QueryEngine**: Enables natural language querying

### Data Flow

```
PDF Files → Text Extraction → NLP Processing → Entity/Relationship Extraction → Graph Construction → Visualization/Query Interface
```

## 📈 Performance Considerations

- **Caching**: Processed data is cached to avoid reprocessing
- **Batch Processing**: Documents processed in batches for efficiency
- **Memory Management**: Large texts split into manageable chunks
- **Progressive Loading**: Streamlit interface loads data progressively

## 🔮 Future Enhancements

- **Machine Learning**: Enhanced entity linking with ML models
- **Multi-language Support**: Extend to other languages
- **Real-time Updates**: Live document processing
- **Advanced Analytics**: Predictive analysis and trend detection
- **Export Capabilities**: Multiple format exports (PDF, Excel, etc.)

## 🤝 Contributing

This system is designed to be modular and extensible. Key areas for enhancement:

1. **Custom Entity Types**: Add domain-specific entities
2. **Relationship Patterns**: Extend relationship detection
3. **Visualization Themes**: Create custom visualization styles
4. **Query Interfaces**: Build more sophisticated query capabilities

## 📝 License

This project is developed for the Simmons and Simmons Challenge as a demonstration of dynamic knowledge graph capabilities for legal document analysis.

---

## 🚀 Getting Started

1. **Place your PDF files** in the designated folder
2. **Run the main script**: `python dynamic_knowledge_graph.py`
3. **Launch the web interface**: `streamlit run streamlit_app.py`
4. **Explore your data** through interactive queries and visualizations

The system will systematically process all PDFs, extract comprehensive knowledge, and provide you with a powerful tool for legal document analysis and exploration.

**Built with ❤️ for the Simmons and Simmons Challenge** 