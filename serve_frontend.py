"""
Simple HTTP server for frontend testing
"""
import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = "frontend"

os.chdir(DIRECTORY)

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.js': 'application/javascript',
})

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"✓ Frontend server running at http://localhost:{PORT}")
    print(f"✓ Serving files from: {os.getcwd()}")
    print(f"✓ API should be at: http://localhost:5000")
    httpd.serve_forever()
