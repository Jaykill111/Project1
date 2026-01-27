import os
import pickle
import glob
import sys

# Add current directory to path so we can import dependencies if needed
sys.path.append(os.getcwd())

model_dir = 'models'
print(f"Checking models in {os.path.abspath(model_dir)}")

if not os.path.exists(model_dir):
    print("Models directory does not exist!")
    sys.exit(1)

files = glob.glob(os.path.join(model_dir, '*.pkl'))
print(f"Found {len(files)} .pkl files")

for f in files:
    print(f"\nLoading {f}...")
    try:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            
        print("  - Load successful")
        print(f"  - Keys: {list(data.keys())}")
        
        if 'threshold' in data:
            print(f"  - Threshold: {data['threshold']} (Type: {type(data['threshold'])})")
        else:
            print("  - ERROR: 'threshold' key missing")
            
        if 'model' in data:
            print(f"  - Model type: {type(data['model'])}")
        else:
            print("  - ERROR: 'model' key missing")

    except Exception as e:
        print(f"  - ERROR loading file: {e}")
        import traceback
        traceback.print_exc()
