#!/usr/bin/env python3
"""
Neural Scan - Video Analysis Deployment
Startup script for the Flask application
"""

import os
import sys
from app import app, init_analyzer

def main():
    """Main entry point for the application"""
    print("🚀 Neural Scan - Body Language & Emotions Analysis")
    print("=" * 50)
    
    # Check if required directories exist
    required_dirs = ['uploads', 'results', 'templates']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
    
    # Check if template exists
    if not os.path.exists('templates/index.html'):
        print("❌ Error: templates/index.html not found!")
        print("Please make sure your HTML frontend is in templates/index.html")
        sys.exit(1)
    
    # Initialize the analyzer
    if init_analyzer():
        print("✅ All systems ready!")
        print("🌐 Starting Flask server on http://localhost:5000")
        print("📱 Access the application in your web browser")
        print("\nPress Ctrl+C to stop the server")
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize the video analyzer")
        sys.exit(1)

if __name__ == '__main__':
    main()