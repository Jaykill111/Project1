import subprocess
import os
import sys
import time
import webbrowser
import threading




def run_backend():
    """Starts the Flask backend on port 5001."""
    print("[*] Starting Backend on port 5001...")
    env = os.environ.copy()
    env["PORT"] = "5001"
    
    # Check if .env exists in api/
    api_dir = os.path.join(os.getcwd(), 'api')
    if not os.path.exists(os.path.join(api_dir, '.env')):
        print("[!] WARNING: api/.env file not found. LLM features may not work.")
        print("   Please create api/.env with OPENROUTER_API_KEY=...")

    # Run app.py from the api directory context or file path
    # Using python executable to run api/app.py
    backend_process = subprocess.Popen(
        [sys.executable, "api/app.py"],
        env=env,
        cwd=os.getcwd() # Run from root, but app.py handles imports relative to itself usually, 
                        # but wait, app.py does `from features import ...`
                        # features.py is in api/. So we likely need to set PYTHONPATH or run FROM api dir.
                        # api/app.py line 14: `from features import ...`
                        # if we run `python api/app.py` from root, it might fail to import features if `api` is not a package or in path.
                        # It's better to run it as a module or from the directory.
    )
    return backend_process

def run_backend_correctly():
    """Starts backend insuring imports work."""
    print("[*] Starting Backend on port 5000...")
    env = os.environ.copy()
    env["PORT"] = "5000"
    
    # Check .env
    if not os.path.exists(os.path.join('api', '.env')):
        print("[!] WARNING: api/.env missing. Please create api/.env with OPENROUTER_API_KEY=...")
    
    # Run from inside 'api' directory so imports and .env loading work naturally
    return subprocess.Popen(
        [sys.executable, "app.py"],
        env=env,
        cwd=os.path.join(os.getcwd(), 'api')
    )

def run_frontend():
    """Starts a simple HTTP server for the frontend on port 3000."""
    print("[*] Starting Frontend on port 3000...")
    # python -m http.server 3000 --directory frontend
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", "3000", "--directory", "frontend"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def main():
    print("="*50)
    print("   SCOPE - Full Stack Launcher")
    print("="*50)
    
    backend = run_backend_correctly()
    frontend = run_frontend()
    
    print("\n[OK] Services started!")
    print("   Backend:  http://localhost:5000")
    print("   Frontend: http://localhost:3000")
    print("\n[..] Waiting for services to initialize...")
    time.sleep(3)
    
    print("--> Opening http://localhost:3000 in your browser...")
    webbrowser.open("http://localhost:3000")
    
    print("\n[Press Ctrl+C to stop all services]\n")
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n[x] Stopping services...")
        backend.terminate()
        frontend.terminate()
        print("Bye!")

if __name__ == "__main__":
    main()
