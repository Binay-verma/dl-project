#!/usr/bin/env python3
"""
PHASE 3: ML BASELINE - FULLY CORRECTED
Trains traditional ML models as teacher models.
Compatible with Phase 4 and Phase 5.

FIXES applied:
  - Uses all 27 features (16 base + 11 temporal) instead of 16
  - Uses F2 threshold (recall-weighted, beta=2) instead of F1
  - Saves to teacher_output/ directory that Phase 4 & 5 expect
  - Timestamped root copies kept for backward compatibility
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

print("=" * 70)
print("PHASE 3: ML BASELINE TRAINING - CORRECTED")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/6] Loading training data...")

feature_files = glob.glob('traffic_features_*.csv')
if not feature_files:
    print("No traffic_features_*.csv file found!")
    print("   Run Phase 2 (simulation) first to generate training data")
    exit(1)

feature_files.sort(key=os.path.getmtime, reverse=True)
data_file = feature_files[0]

print(f"Loading: {os.path.basename(data_file)}")
df = pd.read_csv(data_file)

print(f"   Total samples  : {len(df):,}")
print(f"   Positive rate  : {df['accident_next_60s'].mean()*100:.2f}%")

# ============================================================================
# STEP 2: DEFINE FEATURES - all 27, matching Phase 5 exactly
# ============================================================================
print("\n[2/6] Defining feature columns...")

BASE_FEATURES = [
    'speed', 'vehicle_count', 'occupancy', 'density', 'flow',
    'edge_length', 'num_lanes', 'speed_variance', 'avg_acceleration',
    'sudden_braking_count', 'queue_length', 'accident_frequency',
    'emergency_vehicles', 'reroute_activity', 'is_rush_hour', 'time_of_day'
]

TEMPORAL_FEATURES = [
    'delta_speed_1', 'delta_speed_3', 'rolling_speed_std_5', 'speed_drop_flag',
    'delta_density', 'rolling_density_mean_5', 'density_acceleration',
    'hard_brake_ratio', 'ttc_estimate', 'queue_pressure', 'instability_score'
]

ALL_FEATURES = BASE_FEATURES + TEMPORAL_FEATURES  # 27 total

# Fill any missing columns with 0
for feat in ALL_FEATURES:
    if feat not in df.columns:
        print(f"   '{feat}' missing from CSV - defaulting to 0")
        df[feat] = 0.0

feature_cols = ALL_FEATURES
target_col   = 'accident_next_60s'

if target_col not in df.columns:
    print(f"Target column '{target_col}' not found!")
    exit(1)

print(f"Using {len(feature_cols)} features ({len(BASE_FEATURES)} base + {len(TEMPORAL_FEATURES)} temporal)")

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================
print("\n[3/6] Preparing training data...")

X = df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
y = df[target_col].copy()

print(f"   Shape         : {X.shape}")
print(f"   Positive rate : {y.mean()*100:.2f}%")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

n_pos      = y_train.sum()
n_neg      = len(y_train) - n_pos
pos_weight = n_neg / n_pos

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"   Class weight : {pos_weight:.2f}x for positive class")

# ============================================================================
# STEP 4: TRAIN ML MODELS
# ============================================================================
print("\n[4/6] Training ML models...")

results = {}

# CatBoost
print("\n   [1/4] CatBoost...")
cat_model = CatBoostClassifier(
    iterations=500, learning_rate=0.1, depth=6,
    loss_function='Logloss', eval_metric='AUC',
    random_seed=42, verbose=False, scale_pos_weight=pos_weight
)
cat_model.fit(X_train_scaled, y_train,
              eval_set=(X_val_scaled, y_val),
              early_stopping_rounds=50, verbose=False)
proba = cat_model.predict_proba(X_test_scaled)[:, 1]
results['CatBoost'] = {'model': cat_model, 'proba': proba,
                        'auc': roc_auc_score(y_test, proba)}
print(f"      AUC: {results['CatBoost']['auc']:.4f}")

# XGBoost
print("   [2/4] XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=6,
    scale_pos_weight=pos_weight, random_state=42,
    eval_metric='auc', early_stopping_rounds=50
)
try:
    xgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)], verbose=False)
except TypeError:
    xgb_model = XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        scale_pos_weight=pos_weight, random_state=42, eval_metric='auc'
    )
    xgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)],
                  early_stopping_rounds=50, verbose=False)
proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = {'model': xgb_model, 'proba': proba,
                       'auc': roc_auc_score(y_test, proba)}
print(f"      AUC: {results['XGBoost']['auc']:.4f}")

# Random Forest
print("   [3/4] Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
proba = rf_model.predict_proba(X_test_scaled)[:, 1]
results['RandomForest'] = {'model': rf_model, 'proba': proba,
                            'auc': roc_auc_score(y_test, proba)}
print(f"      AUC: {results['RandomForest']['auc']:.4f}")

# Gradient Boosting
print("   [4/4] Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42
)
gb_model.fit(X_train_scaled, y_train)
proba = gb_model.predict_proba(X_test_scaled)[:, 1]
results['GradBoost'] = {'model': gb_model, 'proba': proba,
                         'auc': roc_auc_score(y_test, proba)}
print(f"      AUC: {results['GradBoost']['auc']:.4f}")

# ============================================================================
# STEP 5: BEST MODEL + F2 THRESHOLD (recall-weighted)
# ============================================================================
print("\n[5/6] Selecting best model and optimising threshold...")

best_name  = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
best_auc   = results[best_name]['auc']
best_proba = results[best_name]['proba']

print(f"Best model : {best_name}  (AUC: {best_auc:.4f})")

# F2 score weights recall 2x over precision.
# Reasoning: missing a real accident (FN) is far worse than a false alarm (FP).
precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(y_test, best_proba)
beta = 2.0
f2_scores  = ((1 + beta**2) * precisions_arr * recalls_arr /
               (beta**2 * precisions_arr + recalls_arr + 1e-10))
best_idx   = np.argmax(f2_scores)
best_threshold = float(thresholds_arr[best_idx]) if best_idx < len(thresholds_arr) else 0.35

print(f"   F2-optimal threshold : {best_threshold:.3f}")

y_pred    = (best_proba >= best_threshold).astype(int)
precision = float(precision_score(y_test, y_pred))
recall    = float(recall_score(y_test, y_pred))
f1        = float(f1_score(y_test, y_pred))
f2        = float(((1 + beta**2) * precision * recall) /
                   (beta**2 * precision + recall + 1e-10))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n   Performance at F2 threshold ({best_threshold:.3f}):")
print(f"   Precision : {precision*100:.2f}%")
print(f"   Recall    : {recall*100:.2f}%")
print(f"   F1 / F2   : {f1:.4f} / {f2:.4f}")
print(f"   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

# ============================================================================
# STEP 6: SAVE - teacher_output/ (read by Phase 4 & 5) + root copies
# ============================================================================
print("\n[6/6] Saving models and results...")

os.makedirs('teacher_output', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -- teacher_output/ (Phase 4 & 5 look here first) --
joblib.dump(best_model, 'teacher_output/teacher_model.pkl')
print("Saved: teacher_output/teacher_model.pkl")

joblib.dump(scaler, 'teacher_output/scaler.pkl')
print("Saved: teacher_output/scaler.pkl")

with open('teacher_output/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("Saved: teacher_output/feature_columns.json")

np.savez('teacher_output/teacher_predictions.npz',
         y_train=y_train.values, y_val=y_val.values, y_test=y_test.values,
         train_proba=best_model.predict_proba(X_train_scaled)[:, 1],
         val_proba=best_model.predict_proba(X_val_scaled)[:, 1],
         test_proba=best_proba)
print("Saved: teacher_output/teacher_predictions.npz")

# -- Timestamped root copies (Phase 4 fallback glob search) --
teacher_fn  = f'teacher_{best_name.lower()}_{timestamp}.pkl'
scaler_fn   = f'teacher_scaler_{timestamp}.pkl'
features_fn = f'feature_columns_{timestamp}.json'
pred_fn     = f'teacher_predictions_{timestamp}.npz'

joblib.dump(best_model, teacher_fn)
joblib.dump(scaler, scaler_fn)
with open(features_fn, 'w') as f:
    json.dump(feature_cols, f, indent=2)
np.savez(pred_fn,
         y_train=y_train.values, y_val=y_val.values, y_test=y_test.values,
         train_proba=best_model.predict_proba(X_train_scaled)[:, 1],
         val_proba=best_model.predict_proba(X_val_scaled)[:, 1],
         test_proba=best_proba)
print(f"Saved timestamped copies: {teacher_fn}, {scaler_fn}")

# Results summary
results_summary = {
    'timestamp':           timestamp,
    'best_model':          best_name,
    'auc':                 float(best_auc),
    'threshold_method':    'F2 (recall-weighted, beta=2)',
    'optimal_threshold':   best_threshold,
    'precision':           precision,
    'recall':              recall,
    'f1':                  f1,
    'f2':                  f2,
    'features':            feature_cols,
    'n_features':          len(feature_cols),
    'n_base_features':     len(BASE_FEATURES),
    'n_temporal_features': len(TEMPORAL_FEATURES),
    'n_train':             int(len(X_train)),
    'n_val':               int(len(X_val)),
    'n_test':              int(len(X_test)),
    'positive_rate':       float(y.mean()),
    'all_models':          {n: {'auc': float(r['auc'])} for n, r in results.items()}
}

with open('teacher_output/teacher_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
results_fn = f'teacher_results_{timestamp}.json'
with open(results_fn, 'w') as f:
    json.dump(results_summary, f, indent=2)
print("Saved: teacher_output/teacher_results.json")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
names = list(results.keys())
aucs  = [results[n]['auc'] for n in names]
bars  = ax.barh(names, aucs, color='#3498DB')
ax.set_xlabel('AUC Score')
ax.set_title('Model Comparison (27 features)')
ax.axvline(x=best_auc, color='r', linestyle='--', label='Best')
ax.legend()
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{auc:.4f}', va='center')

ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix - {best_name}\n(F2 threshold={best_threshold:.3f})')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')

ax = axes[1, 0]
ax.plot(recalls_arr, precisions_arr, linewidth=2)
ax.scatter([recall], [precision], color='red', s=100, zorder=5,
           label=f'F2-optimal (t={best_threshold:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices     = np.argsort(importances)[-15:]
    ax.barh(range(len(indices)), importances[indices], color='#27AE60')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances')
else:
    ax.text(0.5, 0.5, 'Feature importance\nnot available',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plot_fn = f'teacher_results_{timestamp}.png'
plt.savefig(plot_fn, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_fn}")

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE - TEACHER MODEL READY!")
print("=" * 70)
print(f"\nBest Teacher Model : {best_name}")
print(f"   Features used   : {len(feature_cols)} (16 base + 11 temporal)")
print(f"   AUC             : {best_auc:.4f}")
print(f"   Threshold (F2)  : {best_threshold:.3f}")
print(f"   Precision       : {precision*100:.2f}%")
print(f"   Recall          : {recall*100:.2f}%")
print(f"   F1 / F2         : {f1:.4f} / {f2:.4f}")
print(f"\nPhase 4 & 5 will read from teacher_output/:")
print(f"   teacher_output/teacher_model.pkl")
print(f"   teacher_output/scaler.pkl")
print(f"   teacher_output/feature_columns.json  (27 features)")
print(f"   teacher_output/teacher_predictions.npz")
print(f"\nNext step: python phase4_corrected.py")
print("=" * 70)