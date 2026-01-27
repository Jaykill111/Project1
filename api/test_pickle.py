import pickle
import os

model_path = 'h:/Code/SCOPE/api/models/goals_classifier_external_2.5.pkl'

try:
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Successfully loaded: {os.path.basename(model_path)}")
    print(f"   Keys: {list(data.keys())}")
    print(f"   Features: {len(data.get('features', []))}")
except Exception as e:
    print(f"❌ Error loading: {e}")
    import traceback
    traceback.print_exc()
