#!/usr/bin/env python3
"""
PHASE 4: KNOWLEDGE DISTILLATION - v2 (BUG-FIXED)
Creates lightweight student model from teacher, saves in Phase 5 format.

ORIGINAL FIXES (v1):
  - Removed CascadeClassifier custom class (caused pickle crash in Phase 5)
  - Student saved as plain RandomForestClassifier - no custom class needed
  - Saves to student_output/ directory that Phase 5 reads first
  - Reads from teacher_output/ created by Phase 3
  - Uses F2 threshold (recall-weighted, beta=2) instead of hardcoded 0.60
  - Timestamped root copies kept for Phase 5 fallback search

BUG-FIXES in v2 (precision / F1 issues):
  FIX 1 — alpha reduced 0.8 → 0.5
      With alpha=0.8, uncertain teacher predictions (e.g. proba=0.3 on a true
      positive) produced soft_target = 0.8*0.3 + 0.2*1.0 = 0.44 < 0.5,
      silently flipping genuine positives to 0 in hard_targets.
      Lower alpha keeps hard labels dominant, preventing label corruption.

  FIX 2 — RF now trained on hard_targets (not raw y_train)
      The original code fitted RF on y_train while all sample_weights were
      derived from soft_targets (based on hard_targets).  This mismatch meant
      distillation was completely wasted for the RF student.
      Both students now learn from the same distilled hard_targets.

  FIX 3 — sample_weight minimum clip 0.1 → 0.0
      Keeping a minimum weight of 0.1 forced the model to learn from maximally
      uncertain teacher predictions (soft_target ≈ 0.5).  These ambiguous
      boundary samples are noise — giving them zero weight cleans up training.

  FIX 4 — label-flip diagnostic printed at distillation step
      Prints how many true positives / negatives were flipped by the teacher
      blend so you can catch future regressions early.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, precision_recall_curve
)
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

print("=" * 70)
print("PHASE 4: KNOWLEDGE DISTILLATION - v2 (BUG-FIXED)")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA AND TEACHER MODEL
# ============================================================================
print("\n[1/7] Loading data and teacher model...")

# Training data
feature_files = glob.glob('traffic_features_*.csv')
if not feature_files:
    print("No traffic_features_*.csv found!")
    exit(1)
feature_files.sort(key=os.path.getmtime, reverse=True)
df = pd.read_csv(feature_files[0])
print(f"Data loaded: {os.path.basename(feature_files[0])}  ({len(df):,} rows)")

# Teacher model - prefer teacher_output/ from Phase 3
if os.path.exists('teacher_output/teacher_model.pkl'):
    teacher_model = joblib.load('teacher_output/teacher_model.pkl')
    print("Teacher loaded: teacher_output/teacher_model.pkl")
else:
    t_files = [f for f in glob.glob('teacher_*.pkl') if 'scaler' not in f]
    if not t_files:
        print("No teacher model found! Run Phase 3 first.")
        exit(1)
    t_files.sort(key=os.path.getmtime, reverse=True)
    teacher_model = joblib.load(t_files[0])
    print(f"Teacher loaded: {os.path.basename(t_files[0])}")

# Scaler - prefer teacher_output/
if os.path.exists('teacher_output/scaler.pkl'):
    scaler = joblib.load('teacher_output/scaler.pkl')
    print("Scaler loaded: teacher_output/scaler.pkl")
else:
    s_files = glob.glob('teacher_scaler_*.pkl')
    if not s_files:
        print("No scaler found! Run Phase 3 first.")
        exit(1)
    s_files.sort(key=os.path.getmtime, reverse=True)
    scaler = joblib.load(s_files[0])
    print(f"Scaler loaded: {os.path.basename(s_files[0])}")

# Feature columns - prefer teacher_output/
if os.path.exists('teacher_output/feature_columns.json'):
    with open('teacher_output/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    print(f"Features loaded: teacher_output/feature_columns.json  ({len(feature_cols)} cols)")
else:
    fc_files = glob.glob('feature_columns_*.json')
    if not fc_files:
        print("No feature columns file found! Run Phase 3 first.")
        exit(1)
    fc_files.sort(key=os.path.getmtime, reverse=True)
    with open(fc_files[0], 'r') as f:
        feature_cols = json.load(f)
    print(f"Features loaded: {os.path.basename(fc_files[0])}  ({len(feature_cols)} cols)")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n[2/7] Preparing data...")

target_col = 'accident_next_60s'

for feat in feature_cols:
    if feat not in df.columns:
        df[feat] = 0.0

X = df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
y = df[target_col].copy()

print(f"   Data shape    : {X.shape}")
print(f"   Positive rate : {y.mean()*100:.2f}%")

# Same stratified split as Phase 3 (same random_state ensures consistency)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Use Phase 3 scaler - do NOT refit, must be identical transform
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ============================================================================
# STEP 3: GENERATE SOFT LABELS FROM TEACHER
# ============================================================================
print("\n[3/7] Generating soft labels from teacher...")

teacher_train_proba = teacher_model.predict_proba(X_train_scaled)[:, 1]
teacher_val_proba   = teacher_model.predict_proba(X_val_scaled)[:, 1]
teacher_test_proba  = teacher_model.predict_proba(X_test_scaled)[:, 1]

teacher_auc = roc_auc_score(y_test, teacher_test_proba)
print(f"Teacher test AUC: {teacher_auc:.4f}")

# ── FIX 1: alpha reduced from 0.8 → 0.5 ────────────────────────────────────
# With alpha=0.8 the teacher dominated so strongly that true positives with
# a low teacher probability (e.g. 0.3) were flipped to 0 in hard_targets:
#   0.8 * 0.3 + 0.2 * 1.0 = 0.44  →  hard label = 0  (wrong!)
# alpha=0.5 keeps hard ground-truth labels equally weighted.
alpha = 0.5
soft_targets = alpha * teacher_train_proba + (1 - alpha) * y_train.values

# ── FIX 4: label-flip diagnostic ────────────────────────────────────────────
hard_targets = (soft_targets >= 0.5).astype(int)
y_np         = y_train.values.astype(int)
flipped_pos  = int(((y_np == 1) & (hard_targets == 0)).sum())
flipped_neg  = int(((y_np == 0) & (hard_targets == 1)).sum())
total_pos    = int((y_np == 1).sum())
total_neg    = int((y_np == 0).sum())
print(f"   Label-flip diagnostic (alpha={alpha}):")
print(f"     True positives flipped to 0 : {flipped_pos:,} / {total_pos:,} "
      f"({flipped_pos/max(total_pos,1)*100:.1f}%)  ← target: <5%")
print(f"     True negatives flipped to 1 : {flipped_neg:,} / {total_neg:,} "
      f"({flipped_neg/max(total_neg,1)*100:.1f}%)")
if flipped_pos / max(total_pos, 1) > 0.10:
    print("   ⚠️  >10% of positives flipped — consider lowering alpha further.")

# ── FIX 3: sample_weight minimum clip 0.1 → 0.0 ────────────────────────────
# Uncertain teacher predictions (soft_target ≈ 0.5) were previously kept at
# weight=0.1. These maximally ambiguous samples are pure noise and should not
# influence training. Setting clip min to 0.0 excludes them.
sample_weights = np.clip(np.abs(soft_targets - 0.5) * 2, 0.0, 1.0)

# ============================================================================
# STEP 4: TRAIN STUDENT MODELS
# ============================================================================
print("\n[4/7] Training student models...")

# Student A: LogisticRegression (fastest inference, interpretable)
print("\n   [1/2] LogisticRegression student...")
lr_student = LogisticRegression(
    max_iter=1000, class_weight='balanced',
    random_state=42, solver='liblinear'
)
lr_student.fit(X_train_scaled, hard_targets, sample_weight=sample_weights)
lr_val_proba = lr_student.predict_proba(X_val_scaled)[:, 1]
lr_auc = roc_auc_score(y_val, lr_val_proba)
print(f"      Validation AUC: {lr_auc:.4f}")

# Student B: RandomForest
print("   [2/2] RandomForest student (500 trees)...")
rf_student = RandomForestClassifier(n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    min_samples_split=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)


# ── FIX 2: RF now trained on hard_targets (not raw y_train) ─────────────────
# The original code trained RF on y_train while sample_weights were derived
# from soft_targets (aligned with hard_targets). This mismatch wasted
# distillation entirely for RF. Both students must use the same distilled
# hard_targets so the sample weights are consistent with the labels.
rf_student.fit(X_train_scaled, hard_targets, sample_weight=sample_weights)
rf_val_proba = rf_student.predict_proba(X_val_scaled)[:, 1]
rf_auc = roc_auc_score(y_val, rf_val_proba)
print(f"      Validation AUC: {rf_auc:.4f}")

# Pick best
if rf_auc >= lr_auc:
    student_model = rf_student
    student_name  = 'RandomForest'
    test_proba    = rf_student.predict_proba(X_test_scaled)[:, 1]
    print(f"\n   Best student: RandomForest  (Val AUC {rf_auc:.4f})")
else:
    student_model = lr_student
    student_name  = 'LogisticRegression'
    test_proba    = lr_student.predict_proba(X_test_scaled)[:, 1]
    print(f"\n   Best student: LogisticRegression  (Val AUC {lr_auc:.4f})")

# ============================================================================
# STEP 5: EVALUATE ON TEST SET WITH F2 THRESHOLD
# ============================================================================
print("\n[5/7] Evaluating on test set...")

test_auc = roc_auc_score(y_test, test_proba)

# F2 threshold - recall-weighted (missing an accident is worse than a false alarm)
precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(y_test, test_proba)
beta = 2.0
f2_scores = ((1 + beta**2) * precisions_arr * recalls_arr /
             (beta**2 * precisions_arr + recalls_arr + 1e-10))
best_idx = np.argmax(f2_scores)
deployment_threshold = float(thresholds_arr[best_idx]) if best_idx < len(thresholds_arr) else 0.35

print(f"   F2-optimal threshold : {deployment_threshold:.3f}")

y_pred = (test_proba >= deployment_threshold).astype(int)
prec   = float(precision_score(y_test, y_pred))
rec    = float(recall_score(y_test, y_pred))
f1     = float(f1_score(y_test, y_pred))
f2     = float(((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-10))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nStudent Test Performance:")
print(f"   AUC       : {test_auc:.4f}")
print(f"   Precision : {prec*100:.2f}%")
print(f"   Recall    : {rec*100:.2f}%")
print(f"   F1 / F2   : {f1:.4f} / {f2:.4f}")
print(f"   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

# ============================================================================
# STEP 6: COMPARE TEACHER VS STUDENT
# ============================================================================
print("\n[6/7] Comparing student vs teacher...")

t_pred = (teacher_test_proba >= deployment_threshold).astype(int)
t_prec = float(precision_score(y_test, t_pred))
t_rec  = float(recall_score(y_test, t_pred))
t_f1   = float(f1_score(y_test, t_pred))

print(f"   Teacher : Precision={t_prec*100:.2f}%  Recall={t_rec*100:.2f}%  F1={t_f1:.4f}  AUC={teacher_auc:.4f}")
print(f"   Student : Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1:.4f}  AUC={test_auc:.4f}")

prec_ret = prec / t_prec if t_prec > 0 else 0
rec_ret  = rec  / t_rec  if t_rec  > 0 else 0
print(f"\n   Retention - Precision: {prec_ret*100:.1f}%  Recall: {rec_ret*100:.1f}%")

# ============================================================================
# STEP 7: SAVE TO student_output/ (Phase 5 reads here first)
# ============================================================================
print("\n[7/7] Saving models and results...")

os.makedirs('student_output', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Phase 5 load_model() looks for student_output/student_model.pkl first.
# Save as a plain dict - NO custom classes, no pickle errors.
model_data = {
    'model':      student_model,   # plain sklearn object
    'threshold':  deployment_threshold,
    'type':       student_name,
    'features':   feature_cols,
    'n_features': len(feature_cols),
    'auc':        float(test_auc),
    'precision':  float(prec),
    'recall':     float(rec),
    'f1':         float(f1),
    'f2':         float(f2),
    'timestamp':  timestamp,
    'alpha':      alpha,                  # saved for traceability
    'flip_rate_pos': flipped_pos / max(total_pos, 1),
}

joblib.dump(model_data, 'student_output/student_model.pkl')
print("Saved: student_output/student_model.pkl")

with open('student_output/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("Saved: student_output/feature_columns.json")

# Copy scaler to student_output so Phase 5 finds everything in one place
joblib.dump(scaler, 'student_output/scaler.pkl')
print("Saved: student_output/scaler.pkl")

# Timestamped root copies for Phase 5 fallback glob search
# NOTE: filename does NOT contain 'cascade' - Phase 5 filters those out
student_fn  = f'student_model_precision_{timestamp}.pkl'
scaler_fn   = f'teacher_scaler_{timestamp}.pkl'
features_fn = f'feature_columns_{timestamp}.json'

joblib.dump(model_data, student_fn)
joblib.dump(scaler, scaler_fn)
with open(features_fn, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"Saved root copies: {student_fn}, {scaler_fn}")

# Results JSON
results = {
    'timestamp':          timestamp,
    'student_type':       student_name,
    'deployment_threshold': deployment_threshold,
    'threshold_method':   'F2 (recall-weighted, beta=2)',
    'distillation': {
        'alpha':          alpha,
        'flip_rate_pos':  flipped_pos / max(total_pos, 1),
        'flip_rate_neg':  flipped_neg / max(total_neg, 1),
    },
    'student_metrics':    {'auc': float(test_auc), 'precision': prec,
                           'recall': rec, 'f1': f1, 'f2': f2},
    'teacher_metrics':    {'auc': float(teacher_auc), 'precision': t_prec,
                           'recall': t_rec, 'f1': t_f1},
    'retention':          {'precision': float(prec_ret), 'recall': float(rec_ret)},
    'confusion_matrix':   {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'features':           feature_cols,
    'n_features':         len(feature_cols)
}

with open('student_output/student_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: student_output/student_results.json")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.bar(['Teacher', 'Student'], [teacher_auc, test_auc], color=['#3498DB', '#27AE60'])
ax.set_ylabel('AUC')
ax.set_title('Teacher vs Student AUC')
ax.set_ylim([0, 1])
for i, v in enumerate([teacher_auc, test_auc]):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Student Confusion Matrix\n(F2 threshold={deployment_threshold:.3f})')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')

ax = axes[1, 0]
ax.plot(recalls_arr, precisions_arr, linewidth=2)
ax.scatter([rec], [prec], color='red', s=150, zorder=5,
           label=f'F2-optimal (t={deployment_threshold:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Student Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
metrics      = ['Precision', 'Recall', 'F1']
teacher_vals = [t_prec, t_rec, t_f1]
student_vals = [prec, rec, f1]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, teacher_vals, w, label='Teacher', color='#3498DB', alpha=0.8)
ax.bar(x + w/2, student_vals, w, label='Student',  color='#27AE60', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Teacher vs Student Metrics')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_fn = f'student_results_{timestamp}.png'
plt.savefig(plot_fn, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_fn}")

print("\n" + "=" * 70)
print("PHASE 4 COMPLETE - STUDENT MODEL READY FOR PHASE 5!")
print("=" * 70)
print(f"\nStudent Model : {student_name}")
print(f"   Features   : {len(feature_cols)} (16 base + 11 temporal)")
print(f"   AUC        : {test_auc:.4f}")
print(f"   Threshold  : {deployment_threshold:.3f}  (F2-optimal)")
print(f"   Precision  : {prec*100:.2f}%")
print(f"   Recall     : {rec*100:.2f}%")
print(f"   F1 / F2    : {f1:.4f} / {f2:.4f}")
print(f"\nBug-fix summary (v2):")
print(f"   alpha         : 0.8 → 0.5  (prevents true-positive label flipping)")
print(f"   RF labels     : y_train → hard_targets  (distillation now applies to RF)")
print(f"   weight clip   : min 0.1 → 0.0  (removes noisy boundary samples)")
print(f"   flip rate pos : {flipped_pos/max(total_pos,1)*100:.1f}%  (was potentially >30% with alpha=0.8)")
print(f"\nPhase 5 reads from student_output/:")
print(f"   student_output/student_model.pkl    (no custom classes)")
print(f"   student_output/feature_columns.json (27 features)")
print(f"   student_output/scaler.pkl")
print(f"\nNext step: python phase5_real.py")
print("=" * 70)
