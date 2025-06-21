"""
Configuration file for JuriSpark Legal Analytics Platform
Centralized configuration management for the three-pillar system
"""

import os
from pathlib import Path
from typing import Dict, List, Any

class Config:
    """Main configuration class"""
    
    # Application Settings
    APP_NAME = "JuriSpark Legal Analytics Platform"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/jurispark.db")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        "page_title": APP_NAME,
        "page_icon": "⚖️",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }

class KnowledgeGraphConfig:
    """Configuration for Pillar 1: Knowledge Graph Engine"""
    
    # Entity Types
    ENTITY_TYPES = {
        "PERSON": {"color": "#FF6B6B", "size": 20},
        "ORG": {"color": "#4ECDC4", "size": 25},
        "MONEY": {"color": "#45B7D1", "size": 15},
        "DATE": {"color": "#96CEB4", "size": 12},
        "LEGAL_TERM": {"color": "#FECA57", "size": 18},
        "CONTRACT": {"color": "#FF9FF3", "size": 22},
        "CASE": {"color": "#54A0FF", "size": 16}
    }
    
    # Relationship Types
    RELATIONSHIP_TYPES = {
        "financial_flow": {
            "color": "#2E8B57",
            "keywords": ["payment", "transfer", "invoice", "compensation", "damages", "fee"]
        },
        "information_flow": {
            "color": "#4169E1", 
            "keywords": ["report", "communication", "disclosure", "notification", "advice"]
        },
        "physical_flow": {
            "color": "#DC143C",
            "keywords": ["delivery", "shipment", "transfer", "movement", "supply"]
        }
    }
    
    # Graph Layout Settings
    LAYOUT_CONFIG = {
        "algorithm": "spring",
        "k": 1,
        "iterations": 50,
        "node_spacing": 100,
        "edge_length": 200
    }
    
    # Temporal Analysis
    TEMPORAL_CONFIG = {
        "date_formats": [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", 
            "%B %d, %Y", "%d %B %Y", "%Y"
        ],
        "time_windows": ["daily", "weekly", "monthly", "quarterly", "yearly"]
    }

class ArgumentAnalyticsConfig:
    """Configuration for Pillar 2: Argument Analytics"""
    
    # ML Model Configuration
    ML_CONFIG = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "test_size": 0.2
    }
    
    # NLP Configuration
    NLP_CONFIG = {
        "spacy_model": "en_core_web_sm",
        "sentence_transformer": "all-MiniLM-L6-v2",
        "max_features": 1000,
        "stop_words": "english"
    }
    
    # Argument Categories
    ARGUMENT_CATEGORIES = {
        "liability": {
            "keywords": ["breach", "violation", "fault", "negligence", "responsibility"],
            "weight": 1.0
        },
        "damages": {
            "keywords": ["loss", "harm", "injury", "compensation", "restitution"],
            "weight": 0.9
        },
        "causation": {
            "keywords": ["caused", "resulted", "led to", "consequence", "effect"],
            "weight": 0.8
        },
        "defense": {
            "keywords": ["justified", "force majeure", "impossibility", "excuse"],
            "weight": 0.7
        }
    }
    
    # Sentiment Analysis
    SENTIMENT_CONFIG = {
        "positive_indicators": ["clearly", "established", "proven", "demonstrated", "evident"],
        "negative_indicators": ["disputed", "unclear", "questionable", "alleged", "unproven"],
        "strength_modifiers": ["strongly", "clearly", "definitively", "arguably", "possibly"]
    }
    
    # AI Enhancement Settings
    AI_ENHANCEMENT_CONFIG = {
        "max_suggestions": 5,
        "confidence_threshold": 0.7,
        "context_window": 3  # sentences
    }

class DebateSimulationConfig:
    """Configuration for Pillar 3: Debate Simulation"""
    
    # Simulation Parameters
    SIMULATION_CONFIG = {
        "default_rounds": 100,
        "max_rounds": 1000,
        "exchanges_per_round": 3,
        "time_limit_per_exchange": 300,  # seconds
        "confidence_threshold": 0.6
    }
    
    # Agent Configuration
    AGENT_CONFIG = {
        "claimant": {
            "personality": "assertive",
            "strategy": "evidence_focused",
            "response_style": "detailed"
        },
        "respondent": {
            "personality": "defensive",
            "strategy": "counter_attack",
            "response_style": "concise"
        },
        "tribunal": {
            "personality": "neutral",
            "strategy": "balanced",
            "response_style": "analytical"
        }
    }
    
    # Legal Principles
    LEGAL_PRINCIPLES = [
        "burden_of_proof",
        "preponderance_of_evidence", 
        "causation_requirements",
        "foreseeability_standard",
        "mitigation_of_damages",
        "good_faith_obligations"
    ]
    
    # Debate Topics
    DEBATE_TOPICS = [
        "contractual_interpretation",
        "breach_determination",
        "damages_calculation",
        "causation_analysis",
        "defense_validity",
        "remedy_appropriateness"
    ]

class VisualizationConfig:
    """Configuration for visualizations and UI"""
    
    # Color Schemes
    COLOR_SCHEMES = {
        "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "legal": ["#1f4e79", "#2c5282", "#3182ce", "#4299e1", "#63b3ed"],
        "debate": ["#1976d2", "#c2185b", "#7b1fa2", "#388e3c", "#f57c00"]
    }
    
    # Chart Configurations
    CHART_CONFIG = {
        "default_height": 400,
        "default_width": 600,
        "font_family": "Arial, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50}
    }
    
    # Network Visualization
    NETWORK_CONFIG = {
        "node_size_range": (10, 30),
        "edge_width_range": (1, 5),
        "layout_algorithm": "force_directed",
        "animation_duration": 1000,
        "hover_distance": 50
    }

class SecurityConfig:
    """Security and privacy configuration"""
    
    # Data Protection
    DATA_PROTECTION = {
        "encrypt_sensitive_data": True,
        "anonymize_personal_info": True,
        "data_retention_days": 365,
        "backup_frequency": "daily"
    }
    
    # Access Control
    ACCESS_CONTROL = {
        "require_authentication": False,  # Set to True for production
        "session_timeout": 3600,  # seconds
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "allowed_file_types": [".pdf", ".docx", ".txt"]
    }

class PerformanceConfig:
    """Performance optimization configuration"""
    
    # Caching
    CACHE_CONFIG = {
        "enable_caching": True,
        "cache_ttl": 3600,  # seconds
        "max_cache_size": 1000,  # MB
        "cache_backend": "memory"  # or "redis"
    }
    
    # Processing
    PROCESSING_CONFIG = {
        "max_workers": 4,
        "chunk_size": 1000,  # for batch processing
        "timeout": 300,  # seconds
        "memory_limit": 2048  # MB
    }
    
    # Optimization
    OPTIMIZATION_CONFIG = {
        "lazy_loading": True,
        "compress_data": True,
        "use_gpu": False,  # Set to True if GPU available
        "parallel_processing": True
    }

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///:memory:"

# Configuration factory
def get_config(environment: str = None) -> Config:
    """Get configuration based on environment"""
    if environment is None:
        environment = os.getenv("FLASK_ENV", "development")
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    return config_map.get(environment, DevelopmentConfig)

# Export all configurations
__all__ = [
    "Config",
    "KnowledgeGraphConfig", 
    "ArgumentAnalyticsConfig",
    "DebateSimulationConfig",
    "VisualizationConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "get_config"
] 