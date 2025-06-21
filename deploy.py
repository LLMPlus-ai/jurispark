#!/usr/bin/env python3
"""
Deployment Script for JuriSpark Legal Analytics Platform
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_jurispark():
    """Deploy JuriSpark Legal Analytics Platform"""
    logger.info("🚀 Starting JuriSpark deployment...")
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')))
    if python_version < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    logger.info("✅ Python version check passed")
    
    # Create directories
    directories = ["data", "output", "models", "logs", "uploads", "cache"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory created: {directory}")
    
    # Create .env file
    env_content = """# JuriSpark Environment Variables
FLASK_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=sqlite:///data/jurispark.db
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("✅ Environment file created")
    
    # Create startup script
    startup_script = """#!/bin/bash
echo "🚀 Starting JuriSpark Legal Analytics Platform..."
streamlit run legal_analytics_platform.py --server.port 8501
"""
    
    with open("run.sh", "w") as f:
        f.write(startup_script)
    
    if platform.system() != "Windows":
        os.chmod("run.sh", 0o755)
    
    logger.info("✅ Startup script created")
    
    # Success message
    print("""
🎉 JuriSpark Deployment Successful! 🎉

Quick Start:
• Run: streamlit run legal_analytics_platform.py
• Access: http://localhost:8501

Features:
• 🕸️ Dynamic Knowledge Graph Engine
• 🧠 Argument Analytics & AI Enhancement  
• ⚔️ Debate Simulation & Predictive Outcomes

Happy analyzing! ⚖️
""")
    
    return True

if __name__ == "__main__":
    success = deploy_jurispark()
    sys.exit(0 if success else 1) 