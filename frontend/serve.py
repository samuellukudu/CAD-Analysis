#!/usr/bin/env python3
"""
Simple HTTP server to serve Vite production build.
Handles SPA routing by serving index.html for all routes.
"""

import http.server
import socketserver
import os
from pathlib import Path

# Configuration
PORT = int(os.getenv("PORT", 5173))
DIST_DIR = Path(__file__).parent / "dist"

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves index.html for SPA routing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers if needed
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        # Check if the requested path exists as a file
        if self.path.startswith('/api'):
            # Don't handle API routes - return 404
            self.send_error(404, "API routes not handled by this server")
            return
        
        # Check if it's a file request (has extension)
        path = self.path.split('?')[0]  # Remove query string
        if path != '/' and '.' in os.path.basename(path):
            # It's a file request (e.g., /assets/main.js)
            super().do_GET()
        else:
            # It's a route - serve index.html for SPA routing
            self.path = '/index.html'
            super().do_GET()

def main():
    if not DIST_DIR.exists():
        print(f"Error: Build directory not found at {DIST_DIR}")
        print("Please build the frontend first:")
        print("  npm install")
        print("  npm run build")
        return
    
    with socketserver.TCPServer(("", PORT), SPAHandler) as httpd:
        print(f"Serving frontend at http://localhost:{PORT}")
        print(f"Build directory: {DIST_DIR}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()



