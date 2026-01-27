import sys
import os
import glob
sys.path.insert(0, '.')

model_dir = os.path.join(os.path.dirname(__file__), 'models')
print(f"Model dir: {model_dir}")
print(f"Exists: {os.path.exists(model_dir)}\n")

goals_files = glob.glob(os.path.join(model_dir, 'goals_classifier_*.pkl'))
print(f"Found {len(goals_files)} goals_classifier_*.pkl files:")
for f in sorted(goals_files):
    print(f"  - {os.path.basename(f)}")

print("\n" + "="*70)
print("Testing regex pattern:")
import re

for f in sorted(goals_files)[:3]:
    base = os.path.basename(f)
    print(f"\nFile: {base}")
    
    # Test pattern
    match = re.search(r'goals_classifier_([a-zA-Z0-9]+)_([0-9.]+)', base)
    if match:
        version = match.group(1)
        threshold = float(match.group(2))
        print(f"  ✓ Version: {version}, Threshold: {threshold}")
    else:
        print(f"  ✗ Pattern did NOT match!")
        
print("\n" + "="*70)
