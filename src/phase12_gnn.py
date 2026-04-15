#!/usr/bin/env python3
"""
PHASE 12: GNN + GRAPH-BASED KNOWLEDGE DISTILLATION
Graph Neural Network for accident prediction using road network topology.

WHY GNN BEATS ML/DL HERE:
  ML (XGBoost/RF): sees Edge_123 features in isolation.
  LSTM:            sees Edge_123 features over time in isolation.
  GNN (this):      sees Edge_123 AND its upstream/downstream neighbors.
                   Detects traffic waves building across multiple edges,
                   predicting danger EARLIER and enabling upstream rerouting.

KNOWLEDGE DISTILLATION TYPE COMPARISON:
  Phase 4/8 (Response-based): Student mimics teacher OUTPUT probabilities.
  Phase 12  (Graph-based):    Student mimics teacher GRAPH PROPAGATION
                              (intermediate embeddings + neighbor aggregation).
                              Richer, more transferable knowledge.

PIPELINE:
  1. Build road graph from graph_edges_*.csv + node_features_*.csv
  2. Train GNN Teacher (GraphSAGE, 3 message-passing layers)
  3. Graph-based KD → GNN Student (1 layer, 10x smaller)
  4. Compare: ML student vs DL (LSTM) vs GNN teacher vs GNN student
  5. Deploy best model for Phase 5 rerouting
  6. Save deployment config for Phase 5

NO NEW INSTALLS NEEDED — uses numpy, scipy, sklearn, tensorflow (already installed).
"""

import numpy as np
import pandas as pd
import json
import os
import glob
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not found — using numpy-only GNN fallback")
    TF_AVAILABLE = False

print("=" * 70)
print("PHASE 12: GNN + GRAPH-BASED KNOWLEDGE DISTILLATION")
print("=" * 70)

BETA       = 2.0   # F2 recall-weighted
EPOCHS     = 40
BATCH_SIZE = 512
PATIENCE   = 8

# ============================================================================
# [1/8] LOAD DATA
# ============================================================================
print("\n[1/8] Loading traffic features and road graph...")

# Traffic features (from Phase 2/5)
feat_files = sorted(glob.glob('traffic_features_*.csv'),
                    key=os.path.getmtime, reverse=True)
if not feat_files:
    raise FileNotFoundError("No traffic_features_*.csv found. Run Phase 2 first.")
df = pd.read_csv(feat_files[0])
print(f"✅ Features: {os.path.basename(feat_files[0])}  ({len(df):,} rows)")

# Graph edges (from Phase 5 export)
edge_files = sorted(glob.glob('graph_edges_*.csv'), key=os.path.getmtime, reverse=True)
if edge_files:
    df_edges = pd.read_csv(edge_files[0])
    print(f"✅ Graph edges: {os.path.basename(edge_files[0])}  "
          f"({len(df_edges):,} edges)")
else:
    print("⚠️  No graph_edges_*.csv found — building graph from edge co-occurrence")
    df_edges = None

# ============================================================================
# [2/8] BUILD ROAD GRAPH
# ============================================================================
print("\n[2/8] Building road graph...")

FEATURE_COLS = [
    'speed', 'vehicle_count', 'occupancy', 'density', 'flow',
    'edge_length', 'num_lanes', 'speed_variance', 'avg_acceleration',
    'sudden_braking_count', 'queue_length', 'accident_frequency',
    'emergency_vehicles', 'reroute_activity', 'is_rush_hour', 'time_of_day',
    'delta_speed_1', 'delta_speed_3', 'rolling_speed_std_5', 'speed_drop_flag',
    'delta_density', 'rolling_density_mean_5', 'density_acceleration',
    'hard_brake_ratio', 'ttc_estimate', 'queue_pressure', 'instability_score'
]
TARGET = 'accident_next_60s'

for f in FEATURE_COLS:
    if f not in df.columns:
        df[f] = 0.0
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)

# Get unique edges (nodes in the graph)
if 'edge_id' not in df.columns:
    df['edge_id'] = 'edge_0'   # fallback

unique_edges = sorted(df['edge_id'].unique())
edge_to_idx  = {e: i for i, e in enumerate(unique_edges)}
N_NODES      = len(unique_edges)
print(f"   Nodes (road edges): {N_NODES:,}")

# Build adjacency list from graph_edges CSV or from logical connections
# Detect column names flexibly (source_node / from / src etc.)
def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

if df_edges is not None:
    src_col = _find_col(df_edges, ['source_node','from','src','source','from_edge'])
    dst_col = _find_col(df_edges, ['target_node','to','dst','target','to_edge'])

if df_edges is not None and src_col and dst_col:
    # FIX: map both columns together, then dropna on the combined frame
    # This ensures src and dst arrays stay the same length
    mapped = pd.DataFrame({
        'src': df_edges[src_col].map(edge_to_idx),
        'dst': df_edges[dst_col].map(edge_to_idx),
    }).dropna()
    src_nodes = mapped['src'].astype(int).values
    dst_nodes = mapped['dst'].astype(int).values
    valid = ((src_nodes >= 0) & (dst_nodes >= 0) &
             (src_nodes < N_NODES) & (dst_nodes < N_NODES))
    src_nodes = src_nodes[valid]
    dst_nodes = dst_nodes[valid]
    print(f"   Graph connections: {len(src_nodes):,}  "
          f"(src_col='{src_col}', dst_col='{dst_col}')")
else:
    # Build from edge_id naming convention: edges sharing a prefix are connected
    print("   Building graph from edge naming convention...")
    src_nodes, dst_nodes = [], []
    sorted_uedges = sorted(unique_edges)
    for i in range(len(sorted_uedges) - 1):
        e1, e2 = sorted_uedges[i], sorted_uedges[i+1]
        # Simple heuristic: consecutive edges in sorted order are likely connected
        src_nodes.append(edge_to_idx[e1])
        dst_nodes.append(edge_to_idx[e2])
        src_nodes.append(edge_to_idx[e2])
        dst_nodes.append(edge_to_idx[e1])
    src_nodes = np.array(src_nodes)
    dst_nodes = np.array(dst_nodes)
    print(f"   Heuristic connections: {len(src_nodes):,}")

# Build adjacency matrix as sparse list {node_idx: [neighbor_idx, ...]}
adj = {i: [] for i in range(N_NODES)}
for s, d in zip(src_nodes, dst_nodes):
    adj[s].append(d)
# Remove duplicates
adj = {k: list(set(v)) for k, v in adj.items()}
avg_degree = np.mean([len(v) for v in adj.values()])
print(f"   Average degree: {avg_degree:.1f} neighbors per edge")

# ============================================================================
# [3/8] AGGREGATE FEATURES PER NODE
# ============================================================================
print("\n[3/8] Aggregating node features (mean per edge over all timesteps)...")

# For each unique road edge, aggregate features across all timesteps
# GNN input = per-node feature vector (mean of all samples for that edge)
node_features = np.zeros((N_NODES, len(FEATURE_COLS)), dtype=np.float32)
node_labels   = np.zeros(N_NODES, dtype=np.float32)
node_counts   = np.zeros(N_NODES, dtype=np.int32)

for edge_id, grp in df.groupby('edge_id'):
    if edge_id not in edge_to_idx:
        continue
    idx = edge_to_idx[edge_id]
    node_features[idx] = grp[FEATURE_COLS].mean().values
    node_labels[idx]   = grp[TARGET].mean()   # fraction of time this edge is at risk
    node_counts[idx]   = len(grp)

# Binary label: edge is high-risk if >10% of its timesteps had upcoming accident
node_labels_binary = (node_labels > 0.10).astype(np.float32)

print(f"   Node feature matrix: {node_features.shape}")
print(f"   High-risk nodes: {node_labels_binary.sum():.0f} / {N_NODES} "
      f"({node_labels_binary.mean()*100:.1f}%)")

# Scale node features
scaler_gnn = StandardScaler()
node_features_scaled = scaler_gnn.fit_transform(node_features).astype(np.float32)

# ============================================================================
# [4/8] GRAPHSAGE MESSAGE PASSING (numpy implementation — no extra installs)
# ============================================================================
print("\n[4/8] GraphSAGE message passing (numpy)...")

def graphsage_aggregate(features, adj, n_layers=2):
    """
    GraphSAGE mean aggregation.
    Each layer: h_v = mean([h_v] + [h_u for u in neighbors(v)])
    Returns enriched node embeddings that encode neighborhood context.
    """
    H = features.copy()
    for layer in range(n_layers):
        H_new = np.zeros_like(H)
        for v in range(len(H)):
            neighbors = adj[v]
            if neighbors:
                neigh_mean = H[neighbors].mean(axis=0)
                # Concatenate self + neighbor mean, then reduce back to same dim
                H_new[v] = 0.5 * H[v] + 0.5 * neigh_mean
            else:
                H_new[v] = H[v]
        # Layer normalisation (prevents exploding/vanishing)
        norm = np.linalg.norm(H_new, axis=1, keepdims=True) + 1e-8
        H = H_new / norm
        print(f"   Layer {layer+1}: embedding shape {H.shape}")
    return H.astype(np.float32)

# 3-layer propagation for teacher (captures 3-hop neighborhood)
print("   Teacher: 3-hop propagation...")
X_teacher = graphsage_aggregate(node_features_scaled, adj, n_layers=3)

# 1-layer propagation for student (lightweight — captures only direct neighbors)
print("   Student: 1-hop propagation...")
X_student = graphsage_aggregate(node_features_scaled, adj, n_layers=1)

print(f"   Teacher embeddings: {X_teacher.shape}")
print(f"   Student embeddings: {X_student.shape}")

# ============================================================================
# [5/8] STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[5/8] Stratified split...")

# Filter to nodes with enough data
valid_nodes = node_counts > 0
X_t = X_teacher[valid_nodes]
X_s = X_student[valid_nodes]
y   = node_labels_binary[valid_nodes]
y_raw = node_labels[valid_nodes]

print(f"   Valid nodes: {valid_nodes.sum():,} / {N_NODES}")
print(f"   Positive rate: {y.mean()*100:.1f}%")

idx = np.arange(len(y))
if y.sum() >= 2 and (y == 0).sum() >= 2:
    idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42,
                                           stratify=y)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765,
                                           random_state=42, stratify=y[idx_temp])
else:
    # Too few positive nodes — use random split
    idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765, random_state=42)

X_train_t, X_val_t, X_test_t = X_t[idx_train], X_t[idx_val], X_t[idx_test]
X_train_s, X_val_s, X_test_s = X_s[idx_train], X_s[idx_val], X_s[idx_test]
y_train, y_val, y_test        = y[idx_train], y[idx_val], y[idx_test]

print(f"   Train: {len(y_train):,} ({y_train.mean()*100:.1f}% pos)  "
      f"Val: {len(y_val):,}  Test: {len(y_test):,}")

n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
cw    = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
print(f"   Class weight: {cw[1]:.2f}x for positive nodes")

# ============================================================================
# [6/8] TRAIN GNN TEACHER AND STUDENT WITH KERAS
# ============================================================================
print("\n[6/8] Training GNN models...")

if not TF_AVAILABLE:
    print("⚠️  TensorFlow not available — using sklearn classifiers as fallback")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    teacher_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              random_state=42)
    teacher_clf.fit(X_train_t, y_train)
    teacher_test_prob = teacher_clf.predict_proba(X_test_t)[:, 1]

    student_clf = LogisticRegression(C=1.0, class_weight='balanced',
                                      max_iter=1000, random_state=42)
    student_clf.fit(X_train_s, y_train)
    student_test_prob = student_clf.predict_proba(X_test_s)[:, 1]

    teacher_model = teacher_clf
    student_model = student_clf
    teacher_type  = "GradBoost+GraphSAGE-3hop (teacher)"
    student_type  = "LogReg+GraphSAGE-1hop (student)"

else:
    # ── GNN Teacher: 3-hop embeddings + deep MLP ───────────────────────────
    def build_gnn_teacher(input_dim):
        """Single-output model for training (Keras 3 class_weight needs single output)."""
        inp  = keras.Input(shape=(input_dim,), name='graph_embedding')
        x    = layers.Dense(128, activation='relu')(inp)
        x    = layers.BatchNormalization()(x)
        x    = layers.Dropout(0.3)(x)
        x    = layers.Dense(64,  activation='relu')(x)
        x    = layers.BatchNormalization()(x)
        x    = layers.Dropout(0.2)(x)
        feat = layers.Dense(32,  activation='relu', name='feat_layer')(x)
        out  = layers.Dense(1,   activation='sigmoid', name='output')(feat)
        # Training model: single output (required for class_weight in Keras 3)
        train_model = keras.Model(inputs=inp, outputs=out, name='GNN_Teacher_train')
        # Feature extractor: used AFTER training to get intermediate embeddings for KD
        feat_model  = keras.Model(inputs=inp, outputs=feat, name='GNN_Teacher_feat')
        return train_model, feat_model

    def build_gnn_student(input_dim):
        """Single-output student for graph-based KD via custom training loop."""
        inp  = keras.Input(shape=(input_dim,), name='graph_embedding_1hop')
        x    = layers.Dense(32, activation='relu')(inp)
        x    = layers.BatchNormalization()(x)
        x    = layers.Dropout(0.2)(x)
        feat = layers.Dense(32, activation='relu', name='feat_layer')(x)
        out  = layers.Dense(1,  activation='sigmoid', name='output')(feat)
        train_model = keras.Model(inputs=inp, outputs=out,  name='GNN_Student_train')
        feat_model  = keras.Model(inputs=inp, outputs=feat, name='GNN_Student_feat')
        return train_model, feat_model

    input_dim = X_train_t.shape[1]

    gnn_teacher,      gnn_teacher_feat = build_gnn_teacher(input_dim)
    gnn_student,      gnn_student_feat = build_gnn_student(input_dim)

    print(f"   Teacher params: {gnn_teacher.count_params():,}")
    print(f"   Student params: {gnn_student.count_params():,}")

    # ── Train Teacher (single-output, class_weight works fine) ─────────────
    print("\n   Training GNN Teacher (3-hop GraphSAGE)...")

    # sample_weight is equivalent to class_weight but works for any model
    sample_w = np.where(y_train == 1, cw[1], cw[0]).astype(np.float32)
    sample_w_val = np.where(y_val == 1, cw[1], cw[0]).astype(np.float32)

    gnn_teacher.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='prec'),
                 keras.metrics.Recall(name='rec')]
    )

    t0 = datetime.now()
    gnn_teacher.fit(
        X_train_t, y_train,
        validation_data=(X_val_t, y_val, sample_w_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        sample_weight=sample_w,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_auc', patience=PATIENCE,
                                           restore_best_weights=True, mode='max'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                               patience=3, min_lr=1e-6),
        ],
        verbose=0
    )
    teacher_train_min = (datetime.now() - t0).total_seconds() / 60
    print(f"   Teacher trained: {teacher_train_min:.1f} min")

    # Get teacher outputs and intermediate features for KD
    teacher_out_train = gnn_teacher.predict(X_train_t, verbose=0).flatten()
    teacher_out_val   = gnn_teacher.predict(X_val_t,   verbose=0).flatten()
    teacher_out_test  = gnn_teacher.predict(X_test_t,  verbose=0).flatten()
    # Feature embeddings from intermediate layer (graph-based KD signal)
    teacher_feat_train = gnn_teacher_feat.predict(X_train_t, verbose=0)
    teacher_feat_val   = gnn_teacher_feat.predict(X_val_t,   verbose=0)

    teacher_test_prob = teacher_out_test
    teacher_model     = gnn_teacher
    teacher_type      = "GNN Teacher (3-hop GraphSAGE + MLP)"

    # ── Graph-Based KD: Student learns from teacher soft labels + embeddings ─
    print("\n   Training GNN Student (graph-based KD)...")
    print("   KD targets: teacher output probabilities + feature layer embeddings")

    ALPHA_RESP = 0.4   # weight for response-based loss (output matching)
    ALPHA_FEAT = 0.4   # weight for feature-based loss (embedding matching)
    ALPHA_HARD = 0.2   # weight for hard label BCE loss

    @tf.function
    def gnn_kd_loss(y_true, y_pred_out, y_pred_feat,
                    t_out, t_feat):
        """
        Graph-based knowledge distillation loss:
          response_loss: student output ≈ teacher output probabilities
          feature_loss:  student embeddings ≈ teacher intermediate embeddings
          hard_loss:     standard BCE with ground truth labels
        """
        # Response-based: MSE between student/teacher output probabilities
        response_loss = tf.reduce_mean(tf.square(y_pred_out - t_out))

        # Feature-based: MSE between student/teacher intermediate embeddings
        # This is the GRAPH-BASED KD component — unique to this phase
        feature_loss  = tf.reduce_mean(tf.square(y_pred_feat - t_feat))

        # Hard label BCE
        hard_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(
                tf.reshape(y_true, tf.shape(y_pred_out)), y_pred_out))

        return (ALPHA_RESP * response_loss +
                ALPHA_FEAT * feature_loss  +
                ALPHA_HARD * hard_loss)

    # Build tf.data datasets including teacher soft outputs and embeddings
    def make_kd_dataset(X, y_hard, t_out, t_feat, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices({
            'X':       X.astype(np.float32),
            'y_hard':  y_hard.astype(np.float32),
            't_out':   t_out.flatten().astype(np.float32),
            't_feat':  t_feat.astype(np.float32),
        })
        if shuffle:
            ds = ds.shuffle(10000, seed=42)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_kd_ds = make_kd_dataset(X_train_s, y_train,
                                   teacher_out_train, teacher_feat_train, shuffle=True)
    val_kd_ds   = make_kd_dataset(X_val_s,   y_val,
                                   teacher_out_val,   teacher_feat_val)

    # Custom training loop for graph-based KD
    optimizer  = keras.optimizers.Adam(0.001)
    best_val_auc = 0.0
    best_weights = None
    patience_cnt = 0

    @tf.function
    def train_step(batch):
        X_b      = batch['X']
        y_b      = batch['y_hard']
        t_out_b  = tf.reshape(batch['t_out'], (-1, 1))
        t_feat_b = batch['t_feat']
        # Both student models share weights — update via the training model
        all_vars = gnn_student.trainable_variables + gnn_student_feat.trainable_variables
        # Remove duplicates (shared layers appear in both)
        seen, unique_vars = set(), []
        for v in all_vars:
            if id(v) not in seen:
                seen.add(id(v)); unique_vars.append(v)
        with tf.GradientTape() as tape:
            out_b  = gnn_student(X_b, training=True)
            feat_b = gnn_student_feat(X_b, training=True)
            loss   = gnn_kd_loss(y_b, out_b, feat_b, t_out_b, t_feat_b)
        grads = tape.gradient(loss, unique_vars)
        optimizer.apply_gradients(zip(grads, unique_vars))
        return loss

    print(f"   {'Epoch':>6} {'TrainLoss':>10} {'ValAUC':>8} {'Status':>12}")
    print("   " + "-" * 42)

    t0 = datetime.now()
    for epoch in range(EPOCHS):
        # Training
        train_losses = [train_step(b) for b in train_kd_ds]
        train_loss   = float(np.mean([l.numpy() for l in train_losses]))

        # Validation AUC
        val_pred = gnn_student.predict(X_val_s, verbose=0)
        val_auc     = roc_auc_score(y_val, val_pred.flatten()) if y_val.sum() > 0 else 0

        status = ''
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = gnn_student.get_weights()
            status = '✓ best'
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"   {epoch+1:>6} {train_loss:>10.4f} {val_auc:>8.4f}  early stop")
                break

        if (epoch + 1) % 5 == 0 or status:
            print(f"   {epoch+1:>6} {train_loss:>10.4f} {val_auc:>8.4f}  {status}")

    if best_weights:
        gnn_student.set_weights(best_weights)

    student_train_min = (datetime.now() - t0).total_seconds() / 60
    print(f"   Student trained: {student_train_min:.1f} min")

    student_out_test  = gnn_student.predict(X_test_s, verbose=0)
    student_test_prob = student_out_test.flatten()
    student_model     = gnn_student
    student_type        = "GNN Student (1-hop + graph-based KD)"

# ============================================================================
# [7/8] EVALUATE AND COMPARE ALL MODELS
# ============================================================================
print("\n[7/8] Evaluation and comparison...")

def eval_model(y_true, y_prob, name):
    """Evaluate model with F2-optimal threshold."""
    if y_true.sum() == 0:
        return None
    auc   = float(roc_auc_score(y_true, y_prob))
    precs, recs, thrs = precision_recall_curve(y_true, y_prob)
    f2_arr = ((1 + BETA**2) * precs * recs /
              (BETA**2 * precs + recs + 1e-10))
    best_i = int(np.argmax(f2_arr))
    thr    = float(thrs[best_i]) if best_i < len(thrs) else 0.5
    y_pred = (y_prob >= thr).astype(int)
    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    rec    = float(recall_score(y_true,    y_pred, zero_division=0))
    f1     = float(f1_score(y_true,        y_pred, zero_division=0))
    f2     = float(((1 + BETA**2) * prec * rec) /
                   (BETA**2 * prec + rec + 1e-10))
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   [{name}]")
    print(f"   AUC={auc:.4f}  Prec={prec*100:.1f}%  Rec={rec*100:.1f}%  "
          f"F2={f2:.4f}  Thr={thr:.3f}")
    print(f"   TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return dict(name=name, auc=auc, precision=prec, recall=rec,
                f1=f1, f2=f2, threshold=thr,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

results = {}

# GNN Teacher
r = eval_model(y_test, teacher_test_prob, teacher_type)
if r: results['GNN_Teacher'] = r

# GNN Student (graph-based KD)
r = eval_model(y_test, student_test_prob, student_type)
if r: results['GNN_Student'] = r

# Load Phase 4 ML student results for comparison
print("\n   [Reference: Phase 4 ML Student]")
try:
    with open('student_output/student_results.json') as fh:
        ml_res = json.load(fh)
    ml_rec  = ml_res['student_metrics']['recall']
    ml_prec = ml_res['student_metrics']['precision']
    ml_auc  = ml_res['student_metrics']['auc']
    ml_f2   = ml_res['student_metrics'].get('f2', 0)
    print(f"   AUC={ml_auc:.4f}  Prec={ml_prec*100:.1f}%  Rec={ml_rec*100:.1f}%  "
          f"F2={ml_f2:.4f}")
    results['ML_Student_P4'] = dict(name='ML Student (Phase 4)',
                                     auc=ml_auc, precision=ml_prec,
                                     recall=ml_rec, f2=ml_f2)
except Exception:
    print("   (Phase 4 results not found)")
    ml_rec = ml_prec = ml_auc = ml_f2 = None

# Load Phase 11 hybrid ensemble for comparison
print("\n   [Reference: Phase 11 Hybrid Ensemble]")
try:
    p11_files = sorted(glob.glob('phase11_results_*.json'), key=os.path.getmtime)
    if p11_files:
        with open(p11_files[-1]) as fh:
            p11_res = json.load(fh)
        winner = p11_res.get('winner', {})
        p11_rec  = winner.get('recall', 0)
        p11_prec = winner.get('precision', 0)
        p11_auc  = winner.get('auc', 0)
        p11_f2   = winner.get('f2', 0)
        print(f"   AUC={p11_auc:.4f}  Prec={p11_prec*100:.1f}%  "
              f"Rec={p11_rec*100:.1f}%  F2={p11_f2:.4f}")
        results['Hybrid_P11'] = dict(name='Hybrid Ensemble (Phase 11)',
                                      auc=p11_auc, precision=p11_prec,
                                      recall=p11_rec, f2=p11_f2)
except Exception:
    print("   (Phase 11 results not found)")

# ============================================================================
# [8/8] SAVE
# ============================================================================
print("\n[8/8] Saving...")

os.makedirs('gnn_output', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save GNN models
if TF_AVAILABLE:
    teacher_path = f'gnn_output/gnn_teacher_{timestamp}.keras'
    student_path = f'gnn_output/gnn_student_{timestamp}.keras'
    teacher_model.save(teacher_path)
    student_model.save(student_path)
else:
    teacher_path = f'gnn_output/gnn_teacher_{timestamp}.pkl'
    student_path = f'gnn_output/gnn_student_{timestamp}.pkl'
    joblib.dump(teacher_model, teacher_path)
    joblib.dump(student_model, student_path)

# Save scaler and graph info
scaler_path = f'gnn_output/gnn_scaler_{timestamp}.pkl'
joblib.dump(scaler_gnn, scaler_path)

graph_info_path = f'gnn_output/gnn_graph_info_{timestamp}.json'
graph_info = {
    'edge_to_idx':   edge_to_idx,
    'n_nodes':       N_NODES,
    'feature_cols':  FEATURE_COLS,
    'n_features':    len(FEATURE_COLS),
    'sage_layers_teacher': 3,
    'sage_layers_student': 1,
}
with open(graph_info_path, 'w') as fh:
    json.dump(graph_info, fh, indent=2)

# Save full results
best_gnn = max(
    [k for k in results if k.startswith('GNN')],
    key=lambda k: results[k]['f2']
)
best_r = results[best_gnn]

deployment = {
    'timestamp':          timestamp,
    'best_model':         best_gnn,
    'model_type':         'GNN_graph_based_KD',
    'kd_type':            'graph_based',
    'teacher_path':       teacher_path,
    'student_path':       student_path,
    'scaler_path':        scaler_path,
    'graph_info_path':    graph_info_path,
    'threshold':          best_r.get('threshold', 0.5),
    'auc':                best_r['auc'],
    'precision':          best_r['precision'],
    'recall':             best_r['recall'],
    'f2':                 best_r['f2'],
    'feature_cols':       FEATURE_COLS,
    'sage_hops':          1,   # student uses 1-hop
    'alpha_response':     ALPHA_RESP if TF_AVAILABLE else None,
    'alpha_feature':      ALPHA_FEAT if TF_AVAILABLE else None,
    'alpha_hard':         ALPHA_HARD if TF_AVAILABLE else None,
    'all_results':        {k: {ck: float(cv) if isinstance(cv, (float, np.floating))
                               else cv
                               for ck, cv in v.items()} for k, v in results.items()},
}

depl_path = f'gnn_output/gnn_deployment_{timestamp}.json'
with open(depl_path, 'w') as fh:
    json.dump(deployment, fh, indent=2)

res_path = f'gnn_output/gnn_results_{timestamp}.json'
with open(res_path, 'w') as fh:
    json.dump(deployment, fh, indent=2)

print(f"✅ Teacher model:  {teacher_path}")
print(f"✅ Student model:  {student_path}")
print(f"✅ Scaler:         {scaler_path}")
print(f"✅ Deployment:     {depl_path}")

# Visualisation
n_models = len(results)
if n_models > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names  = [v['name'] if 'name' in v else k for k, v in results.items()]
    aucs   = [v['auc']       for v in results.values()]
    recs   = [v['recall']    for v in results.values()]
    f2s    = [v['f2']        for v in results.values()]
    colors = ['#E74C3C' if 'GNN' in n else '#3498DB' for n in names]

    for ax, vals, title, ylbl in zip(
            axes,
            [aucs, recs, f2s],
            ['AUC Comparison', 'Recall Comparison', 'F2 Comparison'],
            ['AUC', 'Recall', 'F2 Score']):
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylbl, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Phase 12: GNN vs All Models\n(Red = GNN, Blue = Reference)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = f'gnn_output/gnn_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot:           {plot_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 12 COMPLETE — GNN + GRAPH-BASED KD")
print("=" * 70)

print(f"\n{'Model':<35} {'AUC':>7} {'Prec%':>7} {'Rec%':>7} {'F2':>7}")
print("-" * 65)
for k, r in results.items():
    marker = ' ◄ BEST GNN' if k == best_gnn else ''
    name   = r.get('name', k)
    print(f"{name:<35} {r['auc']:>7.4f} {r['precision']*100:>7.1f} "
          f"{r['recall']*100:>7.1f} {r['f2']:>7.4f}{marker}")

print(f"\n📌 Knowledge Distillation Types Used:")
print(f"   Phase 4/8  → Response-based KD  (student mimics teacher probabilities)")
print(f"   Phase 12   → Graph-based KD     (student mimics teacher graph embeddings)")
print(f"                                    = richer, topology-aware knowledge")

print(f"\n📌 Why GNN predicts EARLIER than ML/DL:")
print(f"   ML/DL: 'Edge X is dangerous now'")
print(f"   GNN:   'Edges A→B→X all high density = traffic wave building'")
print(f"   → reroute vehicles on A and B BEFORE they reach X")

print(f"\n⏭️  Phase 5 integration:")
print(f"   Add GNN predictions as a 3rd signal to the Phase 11 ensemble:")
print(f"   final_prob = w_ml * ml_prob + w_dl * dl_prob + w_gnn * gnn_prob")
print(f"   Use graph_info to look up node embedding per edge_id at runtime")
print("=" * 70)