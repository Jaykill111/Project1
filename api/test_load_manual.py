import sys
import os
import glob
import pickle
sys.path.insert(0, '.')

model_dir = './models'
goals_files = glob.glob(os.path.join(model_dir, 'goals_classifier_*.pkl'))
goals_latest = {}

# Selection phase
for f in goals_files:
    base = os.path.basename(f).replace('.pkl', '')
    import re
    match = re.search(r'goals_classifier_([a-zA-Z0-9]+)_([0-9.]+)$', base)
    if not match:
        continue
    version = match.group(1)
    threshold = float(match.group(2))
    priority_map = {'external': 3, 'v3': 2, 'v2': 1}
    priority = priority_map.get(version, 0)
    
    if threshold not in goals_latest:
        goals_latest[threshold] = (f, priority, version)
    else:
        prev_f, prev_priority, prev_version = goals_latest[threshold]
        if priority > prev_priority:
            goals_latest[threshold] = (f, priority, version)

print(f"Selected {len(goals_latest)} models:")
for th, (f, priority, version) in sorted(goals_latest.items()):
    print(f"  - {th}: {os.path.basename(f)} (version={version})")

# Loading phase
goals_models = {}
for th, (f, priority, version) in goals_latest.items():
    print(f"\nLoading {th}...")
    try:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            goals_models[th] = data
            print(f"  ✓ Success!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n✅ Total loaded: {len(goals_models)}")
