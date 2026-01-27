import sys
import os
import glob
import pickle
sys.path.insert(0, '.')

model_dir = os.path.join(os.path.dirname(__file__), 'models')
goals_files = glob.glob(os.path.join(model_dir, 'goals_classifier_*.pkl'))

print(f"Found {len(goals_files)} files\n")

goals_latest = {}
for f in goals_files:
    try:
        base = os.path.basename(f).replace('.pkl', '')
        print(f"Processing: {base}")
        
        # Extract threshold and version
        import re
        match = re.search(r'goals_classifier_([a-zA-Z0-9]+)_([0-9.]+)$', base)
        if not match:
            print(f"  ✗ Regex did not match!")
            continue
        
        version = match.group(1)
        threshold = float(match.group(2))
        print(f"  ✓ Version={version}, Threshold={threshold}")
        
        priority_map = {'external': 3, 'v3': 2, 'v2': 1}
        priority = priority_map.get(version, 0)
        
        if threshold not in goals_latest:
            goals_latest[threshold] = (f, priority, version)
            print(f"    → Added to dict (priority={priority})")
        else:
            prev_f, prev_priority, prev_version = goals_latest[threshold]
            if priority > prev_priority:
                goals_latest[threshold] = (f, priority, version)
                print(f"    → Replaced (priority {prev_priority} → {priority})")
            else:
                print(f"    → Skipped (priority {priority} <= {prev_priority})")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✅ Final selection ({len(goals_latest)} thresholds):")
for th, (f, priority, version) in sorted(goals_latest.items()):
    print(f"  {th}: {version} (priority={priority})")
