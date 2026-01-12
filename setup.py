#!/usr/bin/env python3
"""
Setup script to initialize the project.
"""
import os
import sys
from pathlib import Path

def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'lean/backtests',
        'lean/logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'pandas',
        'numpy',
        'yfinance',
        'sqlalchemy'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All required packages installed.")
    return True

def init_database():
    """Initialize database."""
    try:
        from data.database import init_database
        init_database()
        print("Database initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Make sure your database is configured in configs/config.yaml")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Stock Predictor Setup")
    print("=" * 60)
    
    # Change to project root
    os.chdir(get_project_root())
    
    print("\n[1/4] Creating directories...")
    setup_directories()
    
    print("\n[2/4] Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and run setup again.")
        return
    
    print("\n[3/4] Checking configuration...")
    config_path = Path('configs/config.yaml')
    if not config_path.exists():
        print("Warning: configs/config.yaml not found. Using defaults.")
    else:
        print("Configuration file found.")
    
    print("\n[4/4] Initializing database...")
    init_database()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Edit configs/config.yaml with your database settings")
    print("  2. Edit configs/tickers.yaml with your tickers")
    print("  3. Run: python run_pipeline.py")
    print("     Or use Makefile: make all")

if __name__ == '__main__':
    main()

