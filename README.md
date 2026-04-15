# Real-Time Traffic Accident Prediction using T-GNN with Knowledge Distillation

## Abstract
This project presents a Temporal Graph Neural Network (T-GNN) framework
for real-time traffic accident prediction 60 seconds in advance. We combine
per-road ML danger scores with road-network graph propagation and use
Knowledge Distillation to create a lightweight student model suitable for
deployment. Validated on SUMO simulation and real-world pNEUMA data.

## Results
- T-GNN AUC: 0.9961 | Recall: 96.57%
- Student model retains 98%+ of teacher accuracy at 10x smaller size

## Setup
pip install -r requirements.txt

## Run order
python src/phase3_corrected.py   # Train teacher
python src/phase4_corrected_final.py  # Knowledge distillation
python src/phase12_gnn.py        # GNN
python src/phase13_tgnn_final.py # T-GNN (main)
python src/phase5_demo.py        # Demo

## Dataset
- SUMO simulation: generated locally (see /data for sample)
- pNEUMA: https://open-traffic.epfl.ch/index.php/downloads/
## Run Order
1. python src/phase3_corrected.py # Train teacher
2. python src/phase4_corrected_final.py # Knowledge distillation
3. python src/phase12_gnn.py # GNN
4. python src/phase13_tgnn_final.py # T-GNN (main)
5. python src/phase5_demo.py # Demo
