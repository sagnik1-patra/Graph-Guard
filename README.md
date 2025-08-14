GraphGuard — Real-Time Fraud Detection with Transaction Graphs (Elliptic)

One-liner: Train a baseline fraud detector on the Elliptic Bitcoin dataset, export deployable artifacts, generate evaluation plots, batch predictions, and spin up a FastAPI service with score + explain endpoints (k-hop ego subgraph + feature attributions).

Base folder: C:\Users\sagni\Downloads\GraphGuard
All scripts and artifacts in this README assume this path.

1) Project layout
GraphGuard/
├─ archive (1)/elliptic_bitcoin_dataset/
│  ├─ elliptic_txs_features.csv
│  ├─ elliptic_txs_classes.csv
│  └─ elliptic_txs_edgelist.csv
│
├─ preprocessor.pkl           # scaler, feature list, split steps, label map
├─ model.h5                   # trained Keras model (HDF5 legacy format)
├─ model.keras                # (optional) native Keras format if you saved it
├─ model_config.yaml          # dataset paths, model and train config
├─ metrics.json               # ROC-AUC, PR-AUC, accuracy/F1, Brier
├─ threshold.json             # best validation threshold by F1
│
├─ confusion_matrix.png       # heatmap @ best threshold
├─ accuracy_curve.png         # if training history was available
├─ accuracy_by_time.png       # fallback: accuracy per time step (test window)
│
├─ predictions.csv            # batch predictions (txId, timeStep, prob, pred, …)
├─ predictions.jsonl
├─ predict_cli.py             # CLI for custom runs (dataset or explicit txIds)
│
├─ app.py                     # FastAPI (lazy-loading) score + explain API
├─ index.html                 # Minimal UI for txId scoring and explanation
├─ requirements_api.txt       # API-only dependencies
├─ run_api.bat                # Windows helper for starting the API
└─ README.md

2) Data

This repo uses the Elliptic Bitcoin dataset (3 CSVs):

elliptic_txs_features.csv → columns: txId, timeStep, f1..fN

elliptic_txs_classes.csv → columns: txId, class (values: 1 licit, 2 illicit, or unknown)

elliptic_txs_edgelist.csv → columns: src, dst (directed edges between transactions)

Place them under:
C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\

3) Environment

Use Python 3.10–3.11. For training/evaluation (not API), install:

python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn pyyaml "tensorflow==2.15.0.post1"


GPU is optional. The baseline is a compact MLP (fast on CPU).
If TensorFlow import fails, reinstall TF with the exact version above.

4) Training (baseline)

In your Jupyter notebook, run the “baseline trainer (dtype-safe)” cell you already used.
It:

Loads the 3 Elliptic CSVs by your exact Windows paths

Forces TX IDs to string to avoid merge dtype issues

Adds simple graph features (in/out degree) from the edge list

Splits by timeStep (60% train, 20% val, 20% test) to prevent leakage

Scales features, trains an MLP with early stopping & class weights

Picks a best F1 threshold on the validation window

Saves artifacts:

Artifacts written to C:\Users\sagni\Downloads\GraphGuard:

preprocessor.pkl

model.h5

model_config.yaml

metrics.json

threshold.json

If you want to save native Keras as well, call model.save("model.keras").

5) Evaluation plots

Run the “Accuracy graph + Confusion-matrix heatmap” cell. It will:

Load preprocessor.pkl, model.h5/model.keras, and threshold.json

Rebuild the test window and compute predictions

Save:

confusion_matrix.png (heatmap @ best threshold)

accuracy_curve.png (if hist is in memory) or accuracy_by_time.png (fallback)

classification_report.txt

6) Batch prediction

Run the “Inference & Batch Prediction” cell. It will:

Load all artifacts and rebuild test features

Save:

predictions.csv and predictions.jsonl (test window)

predict_cli.py (a standalone CLI)

CLI usage (optional):

cd "C:\Users\sagni\Downloads\GraphGuard"
# Predict on test split (from your saved time steps)
python predict_cli.py --mode dataset --subset test

# Predict on specific transaction IDs
python predict_cli.py --mode txids --txids 230425980,230425981


Output columns (CSV/JSONL):
txId, timeStep, prob_illicit, pred_label, true_label (if available)

7) FastAPI service (score + explain)

You already wrote app.py and index.html. Use the lazy-loading app.py version (loads artifacts on first request so the server always starts cleanly).

Install API deps
cd "C:\Users\sagni\Downloads\GraphGuard"
pip install -r requirements_api.txt

Start the server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload


Open:

UI: http://127.0.0.1:8000 → redirects to /static/index.html

Health: http://127.0.0.1:8000/health

Self test: http://127.0.0.1:8000/selftest (runs a tiny inference and returns a sample txId)

Endpoints
POST /score

Modes

{"mode":"txid","txId":"230425980"}

{"mode":"payload","features":{... all feature columns ...}}

Response

{
  "txId": "230425980",
  "timeStep": 31,
  "prob_illicit": 0.8421,
  "pred_label": 1,
  "true_label": 1,
  "threshold": 0.53,
  "top_features": [{"feature":"f47","score":0.1223}, ...]
}


Feature attributions use gradient × input saliency on the scaled vector.

POST /explain

Request:

{"txId":"230425980","k":2,"max_nodes":150}


Response: same fields as /score, plus a k-hop ego subgraph as JSON and a base64 PNG for quick visualization:

{
  "subgraph": {"nodes":[{"id":"..."},...], "edges":[{"source":"..","target":".."}, ...]},
  "subgraph_png_base64": "iVBORw0KGgoAAA..."
}

8) Troubleshooting

“This site can’t be reached / ERR_CONNECTION_REFUSED”

Make sure the server is running and listening on the expected host/port:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

Use 127.0.0.1 instead of localhost.

Try a different port: --port 7860.

Check the console for import errors (TensorFlow, missing artifacts).

Windows firewall may prompt you the first time; allow Private networks.

Model not found

Train first to generate model.h5 (and optionally model.keras).

Keep preprocessor.pkl and threshold.json in the same folder.

Merge dtype error (object vs int)

The training/eval/predict code forces TX IDs to strings. If you edit paths, keep that logic.

TensorFlow import/DLL issues

Confirm version:
python -c "import tensorflow as tf; print(tf.__version__)"
Should print 2.15.x. If not, reinstall:
pip install "tensorflow==2.15.0.post1"

Memory

Pandas can handle the Elliptic CSVs on normal RAM. If tight, filter to a time window to test the pipeline.

9) Reproducibility

Seed is fixed to 42 in code for NumPy, Python random, and TensorFlow.

Time-based split prevents look-ahead leakage.

10) How the baseline works

Tabular + simple graph signal: we use the Elliptic features and add in/out degree from the transaction graph.

MLP classifier trained with class weights and early stopping (monitoring PR-AUC).

Threshold is set by maximizing F1 on the validation window, then applied to test.

This gives you a robust starting point. You can later replace the MLP with a GNN (R-GCN/GraphSAGE) using PyTorch Geometric and keep the same pre/post-processing + API.

11) Next steps (optional)

GNN upgrade (PyG): true heterogeneous graph learning, temporal edges, message passing.

Calibration layer: temperature scaling / isotonic regression for well-calibrated scores.

Streaming: consume transactions from Kafka/Redis; score in micro-batches.

Explainability: swap gradient×input for Integrated Gradients or SHAP on the tabular features.

Rules + model: blend simple heuristics (velocity, device fan-out) with model scores via a stacked calibrator.

12) Author
    SAGNIK PATRA
