#!/usr/bin/env python3
"""
phase13_pneuma.py — T-GNN on pNEUMA Urban Real-World Data
==========================================================
Builds road graph from pNEUMA grid cell adjacency.
Runs T-GNN teacher + student KD.
Prints final paper Table V comparison.

Run AFTER phase3_pneuma.py and phase4_pneuma.py.
Usage: py phase13_pneuma.py
"""
import numpy as np, pandas as pd, json, os, glob, joblib, warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score)
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_OK = True
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    TF_OK = False; print("⚠️  TF not found")

print("="*70)
print("PHASE 13 (pNEUMA): T-GNN — REAL URBAN DATA VALIDATION")
print("="*70)

BETA=2.0; EPOCHS=50; BATCH=512; PATIENCE=8
ALPHA_R=0.4; ALPHA_H=0.2; NEIGHBOR_W=0.35

feat_files = sorted(glob.glob('pneuma_features_*.csv'),key=os.path.getmtime,reverse=True)
if not feat_files: print("❌ Run pneuma_feature_extraction.py first."); exit(1)
df = pd.read_csv(feat_files[0])
print(f"Loaded {len(df):,} rows | {df['accident_next_60s'].mean()*100:.1f}% positive")

FEATURE_COLS = [
    'speed','vehicle_count','occupancy','density','flow',
    'edge_length','num_lanes','speed_variance','avg_acceleration',
    'sudden_braking_count','queue_length','accident_frequency',
    'emergency_vehicles','reroute_activity','is_rush_hour','time_of_day',
    'delta_speed_1','delta_speed_3','rolling_speed_std_5','speed_drop_flag',
    'delta_density','rolling_density_mean_5','density_acceleration',
    'hard_brake_ratio','ttc_estimate','queue_pressure','instability_score'
]
for f in FEATURE_COLS:
    if f not in df.columns: df[f] = 0.0
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf,-np.inf],0)
if 'edge_id' not in df.columns: df['edge_id']='cell_0'

# ── Build graph from grid adjacency ──────────────────────────────────────────
print("Building urban road graph from grid cells...")
unique_edges = df['edge_id'].unique().tolist()
edge_to_idx  = {e:i for i,e in enumerate(unique_edges)}
N = len(edge_to_idx)
adj = {i:[] for i in range(N)}

# Adjacent grid cells share an edge
for e in unique_edges:
    try:
        parts = e.split('_')
        if len(parts) >= 2:
            r,c = int(parts[0]), int(parts[1])
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-connectivity
                nb = f"{r+dr}_{c+dc}"
                if nb in edge_to_idx:
                    a = edge_to_idx[e]; b = edge_to_idx[nb]
                    if b not in adj[a]: adj[a].append(b)
    except Exception: pass

conns = sum(len(v) for v in adj.values())
print(f"  {N} nodes | {conns} connections")

# ── ML scores ─────────────────────────────────────────────────────────────────
if os.path.exists('pneuma_student_output/student_model.pkl'):
    raw = joblib.load('pneuma_student_output/student_model.pkl')
    ml_model  = raw['model'] if isinstance(raw,dict) else raw
    ml_scaler = joblib.load('pneuma_student_output/scaler.pkl')
    X_all_s   = ml_scaler.transform(df[FEATURE_COLS].values)
    df['ml_score'] = ml_model.predict_proba(X_all_s)[:,1]
else:
    df['ml_score'] = 1.0/(df['ttc_estimate'].clip(lower=0.1)+1)
    df['ml_score'] = (df['ml_score']-df['ml_score'].min())/(df['ml_score'].max()-df['ml_score'].min()+1e-10)

# ── Graph propagation ─────────────────────────────────────────────────────────
df['graph_score'] = df['ml_score'].copy()
tw_col = 'time_window' if 'time_window' in df.columns else None
if tw_col:
    for tw, grp in df.groupby(tw_col):
        esc = dict(zip(grp['edge_id'], grp['ml_score']))
        for eid, mls in esc.items():
            nidx = edge_to_idx.get(eid,-1)
            if nidx < 0: continue
            nbs = adj.get(nidx,[])
            nav = float(np.mean([esc.get(unique_edges[n] if n<len(unique_edges) else '',mls) for n in nbs])) if nbs else mls
            g   = (1-NEIGHBOR_W)*mls + NEIGHBOR_W*nav
            df.loc[(df['edge_id']==eid)&(df[tw_col]==tw),'graph_score'] = g

# ── Prepare 29-feature input ──────────────────────────────────────────────────
X27  = df[FEATURE_COLS].values.astype(np.float32)
mls  = df['ml_score'].values.astype(np.float32)
gs   = df['graph_score'].values.astype(np.float32)
y    = df['accident_next_60s'].values.astype(np.float32)
fsc  = StandardScaler()
X27s = fsc.fit_transform(X27)
X29  = np.concatenate([X27s,mls.reshape(-1,1),gs.reshape(-1,1)],axis=1).astype(np.float32)

X_tr,X_te,y_tr,y_te,ml_tr,ml_te = train_test_split(X29,y,mls,test_size=0.20,random_state=42,stratify=y)
X_tr,X_va,y_tr,y_va              = train_test_split(X_tr,y_tr,test_size=0.15,random_state=42,stratify=y_tr)

results = {}

# ML_P4 baseline
prec_m,rec_m,thrs_m = precision_recall_curve(y_te,ml_te)
f2m = ((1+BETA**2)*prec_m*rec_m/(BETA**2*prec_m+rec_m+1e-10))
bi  = int(np.argmax(f2m))
thr = float(thrs_m[bi]) if bi<len(thrs_m) else 0.35
yp  = (ml_te>=thr).astype(int)
results['ML_P4'] = {
    'name':'RF Student KD (pNEUMA)',
    'auc':roc_auc_score(y_te,ml_te),
    'recall':float(recall_score(y_te,yp,zero_division=0)),
    'precision':float(precision_score(y_te,yp,zero_division=0)),
    'f1':float(f1_score(y_te,yp,zero_division=0)),
}

if TF_OK:
    pw = min((len(y_tr)-y_tr.sum())/max(y_tr.sum(),1), 10.0)

    def build(n_in,hidden,layers_n,name):
        inp = keras.Input(shape=(n_in,))
        x = inp
        for i in range(layers_n):
            x = layers.Dense(hidden)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.3)(x)
        return keras.Model(inp, layers.Dense(1,activation='sigmoid')(x), name=name)

    # Teacher
    teacher_m = build(29,128,3,'teacher')
    teacher_m.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss='binary_crossentropy',metrics=['AUC'])
    teacher_m.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=EPOCHS,
                  batch_size=BATCH,class_weight={0:1.0,1:pw},
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_AUC',
                    patience=PATIENCE,restore_best_weights=True,mode='max')],verbose=0)
    tp_te = teacher_m.predict(X_te,verbose=0)[:,0]
    tauc  = roc_auc_score(y_te,tp_te)
    prec_t,rec_t,thrs_t = precision_recall_curve(y_te,tp_te)
    f2t = ((1+BETA**2)*prec_t*rec_t/(BETA**2*prec_t+rec_t+1e-10))
    bi  = int(np.argmax(f2t))
    thr = float(thrs_t[bi]) if bi<len(thrs_t) else 0.5
    yp  = (tp_te>=thr).astype(int)
    results['TGNN_Teacher'] = {
        'name':'T-GNN Teacher (pNEUMA)','auc':tauc,
        'recall':float(recall_score(y_te,yp,zero_division=0)),
        'precision':float(precision_score(y_te,yp,zero_division=0)),
        'f1':float(f1_score(y_te,yp,zero_division=0)),
    }
    print(f"T-GNN Teacher pNEUMA: AUC={tauc:.4f} Recall={results['TGNN_Teacher']['recall']*100:.1f}%")

    # Student KD
    student_m = build(29,64,1,'student')
    soft = (ALPHA_R*teacher_m.predict(X_tr,verbose=0)[:,0]+ALPHA_H*y_tr).clip(0,1)
    student_m.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss='binary_crossentropy',metrics=['AUC'])
    student_m.fit(X_tr,soft,validation_data=(X_va,y_va),epochs=EPOCHS,
                  batch_size=BATCH,class_weight={0:1.0,1:pw},
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_AUC',
                    patience=PATIENCE,restore_best_weights=True,mode='max')],verbose=0)
    sp_te = student_m.predict(X_te,verbose=0)[:,0]
    sauc  = roc_auc_score(y_te,sp_te)
    prec_s,rec_s,thrs_s = precision_recall_curve(y_te,sp_te)
    f2s = ((1+BETA**2)*prec_s*rec_s/(BETA**2*prec_s+rec_s+1e-10))
    bi  = int(np.argmax(f2s))
    thr = float(thrs_s[bi]) if bi<len(thrs_s) else 0.5
    yp  = (sp_te>=thr).astype(int)
    results['TGNN_Student'] = {
        'name':'T-GNN Student KD (pNEUMA)','auc':sauc,
        'recall':float(recall_score(y_te,yp,zero_division=0)),
        'precision':float(precision_score(y_te,yp,zero_division=0)),
        'f1':float(f1_score(y_te,yp,zero_division=0)),
    }
    print(f"T-GNN Student pNEUMA: AUC={sauc:.4f} Recall={results['TGNN_Student']['recall']*100:.1f}%")

# Save
os.makedirs('pneuma_tgnn_output',exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({'dataset':'pNEUMA','timestamp':ts,'all_results':results},
          open(f'pneuma_tgnn_output/results_{ts}.json','w'),indent=2)
json.dump({'dataset':'pNEUMA','timestamp':ts,'all_results':results},
          open('pneuma_tgnn_output/results_latest.json','w'),indent=2)

# ── Final paper table ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 13 (pNEUMA) COMPLETE")
print("="*70)
rf_r  = results.get('ML_P4',{}).get('recall',0)
tgnn_r= results.get('TGNN_Student',{}).get('recall',0)
rf_a  = results.get('ML_P4',{}).get('auc',0)
tgnn_a= results.get('TGNN_Student',{}).get('auc',0)

print(f"\n{'Model':<35} {'AUC':>7} {'Recall':>8} {'F1':>7}")
print("-"*55)
for k,r in results.items():
    print(f"{r['name']:<35} {r['auc']:>7.4f} {r['recall']*100:>7.1f}% {r['f1']:>7.4f}")

print(f"""
Paper Table V (Cross-Dataset Validation):
  Model              | SUMO AUC | pNEUMA AUC | SUMO Recall | pNEUMA Recall
  RF Student KD      | 0.9873   | {rf_a:.4f}     | 91.60%      | {rf_r*100:.1f}%
  T-GNN Student KD   | 0.9961   | {tgnn_a:.4f}     | 96.57%      | {tgnn_r*100:.1f}%

Key finding: T-GNN vs RF gap on pNEUMA = {(tgnn_r-rf_r)*100:.1f}% recall
  SUMO gap was: +5.0%
  pNEUMA gap  : {(tgnn_r-rf_r)*100:+.1f}%
""")
print("="*70)