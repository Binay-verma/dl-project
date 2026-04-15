#!/usr/bin/env python3
"""
phase4_pneuma.py — RF Student KD on pNEUMA Data
================================================
Run AFTER phase3_pneuma.py.

Usage: py phase4_pneuma.py
Outputs: pneuma_student_output/
"""
import pandas as pd, numpy as np, joblib, json, os, glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score,
    recall_score, f1_score, precision_recall_curve, confusion_matrix)
from datetime import datetime

print("="*70)
print("PHASE 4 (pNEUMA): RF STUDENT KD — REAL URBAN DATA")
print("="*70)

if not os.path.exists('pneuma_teacher_output/teacher_model.pkl'):
    print("❌ Run phase3_pneuma.py first."); exit(1)

teacher = joblib.load('pneuma_teacher_output/teacher_model.pkl')
scaler  = joblib.load('pneuma_teacher_output/scaler.pkl')
with open('pneuma_teacher_output/feature_columns.json') as f: feature_cols = json.load(f)
data    = np.load('pneuma_teacher_output/teacher_predictions.npz',allow_pickle=True)
y_train = data['y_train']; y_test = data['y_test']; train_proba = data['train_proba']

feat_files = sorted(glob.glob('pneuma_features_*.csv'),key=os.path.getmtime,reverse=True)
df = pd.read_csv(feat_files[0])
for f in feature_cols:
    if f not in df.columns: df[f] = 0.0
X = df[feature_cols].fillna(0).replace([np.inf,-np.inf],0)
y = df['accident_next_60s']

X_tr,X_te_,y_tr,y_te_ = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
_,X_te,_,y_te          = train_test_split(X_te_,y_te_,test_size=0.50,random_state=42,stratify=y_te_)
X_tr_s = scaler.transform(X_tr); X_te_s = scaler.transform(X_te)

alpha = 0.5
soft  = (alpha*train_proba+(1-alpha)*y_train).round().astype(int).clip(0,1)

student = RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_split=20,
    class_weight='balanced',random_state=42,n_jobs=-1)
student.fit(X_tr_s,soft)

prob = student.predict_proba(X_te_s)[:,1]
auc  = roc_auc_score(y_te,prob)
beta=2.0
precs,recs,thrs = precision_recall_curve(y_te,prob)
f2s  = ((1+beta**2)*precs*recs/(beta**2*precs+recs+1e-10))
bi   = int(np.argmax(f2s))
thr  = float(thrs[bi]) if bi<len(thrs) else 0.35
yp   = (prob>=thr).astype(int)
prec = float(precision_score(y_te,yp,zero_division=0))
rec  = float(recall_score(y_te,yp,zero_division=0))
f1   = float(f1_score(y_te,yp,zero_division=0))
tn,fp,fn,tp = confusion_matrix(y_te,yp).ravel()

os.makedirs('pneuma_student_output',exist_ok=True)
joblib.dump({'model':student,'threshold':thr,'features':feature_cols,'type':'RF_pNEUMA'},
            'pneuma_student_output/student_model.pkl')
joblib.dump(scaler,'pneuma_student_output/scaler.pkl')
with open('pneuma_student_output/feature_columns.json','w') as f: json.dump(feature_cols,f)
np.savez('pneuma_student_output/student_predictions.npz',y_test=y_te.values,test_proba=prob)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({'dataset':'pNEUMA','timestamp':ts,'student_metrics':{
    'auc':round(auc,4),'recall':round(rec,4),'precision':round(prec,4),
    'f1':round(f1,4),'threshold':round(thr,4),'tp':int(tp),'fp':int(fp)}},
    open('pneuma_student_output/student_results.json','w'),indent=2)

print(f"RF Student (pNEUMA): AUC={auc:.4f} Recall={rec*100:.1f}% Prec={prec*100:.1f}%")
print(f"SUMO: AUC=0.9871 Recall=91.60%  |  pNEUMA: AUC={auc:.4f} Recall={rec*100:.1f}%")
print("Next: py phase13_pneuma.py")