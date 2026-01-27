#!/usr/bin/env python
"""Robust Flask server wrapper that handles restarts."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    try:
        print("[*] Starting Flask app...")
        from api.app import app
        
        port = int(os.environ.get('PORT', 5000))
        print(f"[*] Flask app created, starting server on port {port}")
        
        # This will block
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
