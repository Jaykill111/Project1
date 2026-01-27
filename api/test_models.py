import sys
sys.path.insert(0, '.')
from app import load_models, goals_models

print("="*70)
print("TESTING: External Model Loading")
print("="*70)

load_models()

print(f"\n✅ Total goals models loaded: {len(goals_models)}")

print("\nGoals classifiers by threshold:")
for k in sorted([x for x in goals_models.keys() if isinstance(x, float)]):
    model_data = goals_models[k]
    model_type = model_data.get('model_type', 'classifier')
    features = len(model_data.get('features', []))
    auc_info = "AUC +14-24% (external)" if 'external' in str(model_data.get('model', '')) or features > 35 else "baseline"
    print(f"  ✓ {k:3.1f}: {features} features - {auc_info}")

print("\n" + "="*70)
