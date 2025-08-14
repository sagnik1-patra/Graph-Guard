import os, io, csv, json, pickle, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

import tensorflow as tf

# --------------------------
# Paths
# --------------------------
BASE        = r"C:\Users\sagni\Downloads\GraphGuard"
FEATURES_CSV= r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_features.csv"
CLASSES_CSV = r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_classes.csv"
EDGES_CSV   = r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_edgelist.csv"

PREPROC_PKL = os.path.join(BASE, "preprocessor.pkl")
H5_PATH     = os.path.join(BASE, "model.h5")
KERAS_PATH  = os.path.join(BASE, "model.keras")  # optional
THRESH_PATH = os.path.join(BASE, "threshold.json")

# --------------------------
# Utils
# --------------------------
def robust_read_csv(path, expected_min_cols=2):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    delims = [",",";","\t","|"]
    encs   = ["utf-8","utf-8-sig","cp1252","latin1"]
    try:
        with open(path, "rb") as f:
            head = f.read(8192).decode("latin1", errors="ignore")
        sniff = csv.Sniffer().sniff(head)
        if sniff.delimiter in delims:
            delims = [sniff.delimiter] + [d for d in delims if d != sniff.delimiter]
    except Exception:
        pass
    last_err=None
    for enc in encs:
        for sep in delims:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1]>=expected_min_cols:
                    return df
            except Exception as e:
                last_err=e
    raise RuntimeError(f"Could not parse {path}. Last error: {last_err}")

def numpy_to_base64_png(arr, figsize=(6,4), title=None):
    plt.figure(figsize=figsize)
    plt.plot(arr)
    if title: plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def fig_to_base64_png():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------
# Load artifacts
# --------------------------
with open(PREPROC_PKL, "rb") as f:
    preproc = pickle.load(f)

feature_cols: List[str] = list(preproc["feature_columns"])
scaler = preproc["scaler"]
time_col = preproc["time_column"]
txid_col = preproc["txid_column"]
splits = preproc["splits"]

with open(THRESH_PATH, "r", encoding="utf-8") as f:
    best_t = float(json.load(f)["best_threshold"])

# Model
model = None
if os.path.exists(KERAS_PATH):
    try:
        model = tf.keras.models.load_model(KERAS_PATH, safe_mode=False)
    except Exception:
        model=None
if model is None and os.path.exists(H5_PATH):
    model = tf.keras.models.load_model(H5_PATH)
if model is None:
    raise RuntimeError("Model not found (model.keras / model.h5). Train first.")

# --------------------------
# Load dataframes + build graph & lookups
# --------------------------
df_feat = robust_read_csv(FEATURES_CSV, expected_min_cols=3)
df_cls  = robust_read_csv(CLASSES_CSV,  expected_min_cols=2)
df_edge = robust_read_csv(EDGES_CSV,    expected_min_cols=2)

# Column names
feat_cols = list(df_feat.columns)
tx_col_feat, time_col_feat = feat_cols[0], feat_cols[1]
cls_cols = list(df_cls.columns)
tx_col_cls, class_col = cls_cols[0], cls_cols[1]
edge_cols = list(df_edge.columns)
src_col, dst_col = edge_cols[0], edge_cols[1]

# Cast IDs to string
df_feat[tx_col_feat] = df_feat[tx_col_feat].astype(str)
df_cls[tx_col_cls]   = df_cls[tx_col_cls].astype(str)
df_edge[src_col]     = df_edge[src_col].astype(str)
df_edge[dst_col]     = df_edge[dst_col].astype(str)

# Degrees
in_deg  = df_edge.groupby(dst_col).size().rename("in_degree")
out_deg = df_edge.groupby(src_col).size().rename("out_degree")
deg_df  = pd.concat([in_deg, out_deg], axis=1).fillna(0.0).reset_index()
deg_df.rename(columns={deg_df.columns[0]: tx_col_feat}, inplace=True)

# Merge features + degrees + labels (labels optional here)
df_feat[time_col_feat] = pd.to_numeric(df_feat[time_col_feat], errors="coerce")
df = df_feat.merge(deg_df, on=tx_col_feat, how="left")
df[["in_degree","out_degree"]] = df[["in_degree","out_degree"]].fillna(0.0)
# Optional labels
label_map = {"1":0,"2":1,"licit":0,"illicit":1}
df_cls[class_col] = df_cls[class_col].astype(str).str.lower().str.strip()
df_cls["label"] = df_cls[class_col].map(label_map).astype("Int64")
df = df.merge(df_cls[[tx_col_cls,"label"]], left_on=tx_col_feat, right_on=tx_col_cls, how="left")
if tx_col_cls in df.columns and tx_col_cls != tx_col_feat:
    df = df.drop(columns=[tx_col_cls])

# Keep numeric features
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[time_col_feat] + feature_cols).reset_index(drop=True)

# Build networkx graph for explanations (undirected view is fine for ego)
G = nx.from_pandas_edgelist(df_edge, source=src_col, target=dst_col, create_using=nx.Graph())

# Index for quick lookup by txId
df_indexed = df.set_index(tx_col_feat)

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="GraphGuard API", version="1.0.0", description="Fraud scoring + k-hop subgraph explanations")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
# Serve static (index.html)
app.mount("/static", StaticFiles(directory=BASE), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/static/index.html">')

@app.get("/health")
def health():
    return {
        "status": "ok",
        "feature_dim": len(feature_cols),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "threshold": best_t,
        "time_column": time_col,
        "txid_column": txid_col,
        "splits": {k: len(v) for k,v in splits.items()},
    }

def _scale_and_predict(x_row: np.ndarray) -> (float, int):
    """x_row is unscaled features (1, D). Returns prob, pred@best_t"""
    Xs = scaler.transform(x_row)
    prob = float(model.predict(Xs, verbose=0).ravel()[0])
    pred = 1 if prob >= best_t else 0
    return prob, pred

def _grad_input_importance(x_scaled: np.ndarray) -> np.ndarray:
    """Gradient x input attribution on scaled features."""
    x = tf.convert_to_tensor(x_scaled.astype("float32"))
    with tf.GradientTape() as tape:
        tape.watch(x)
        p = model(x, training=False)
        y = p[:, 0]  # prob of illicit (sigmoid unit)
    grads = tape.gradient(y, x).numpy()[0]
    # grad * input magnitude
    contrib = np.abs(grads * x.numpy()[0])
    return contrib

def _get_tx_features(txid: str) -> Dict[str, Any]:
    if txid not in df_indexed.index:
        raise KeyError(f"txId '{txid}' not found.")
    row = df_indexed.loc[txid]
    feat_vals = row[feature_cols].values.reshape(1, -1)
    time_step = int(row[time_col])
    label = None
    if "label" in df_indexed.columns and not pd.isna(row.get("label", pd.NA)):
        label = int(row["label"])
    return {"x": feat_vals, "time": time_step, "label": label}

@app.post("/score")
async def score(payload: Dict[str, Any]):
    """
    Two modes:
      - {"mode":"txid", "txId":"..."}
      - {"mode":"payload", "features": {<feature>:value,...}, "in_degree":..., "out_degree":..., "timeStep": <int>}
    Returns: prob_illicit, pred_label, (optional) true_label, top_features
    """
    mode = payload.get("mode", "txid")
    try:
        if mode == "txid":
            txid = str(payload.get("txId", "")).strip()
            if not txid:
                raise ValueError("txId required for mode='txid'")
            info = _get_tx_features(txid)
            prob, pred = _scale_and_predict(info["x"])
            # feature attribution
            x_scaled = scaler.transform(info["x"])
            contrib = _grad_input_importance(x_scaled)
            top_idx = np.argsort(contrib)[::-1][:10]
            top_feats = [{"feature": feature_cols[i], "score": float(contrib[i])} for i in top_idx]
            return {
                "txId": txid,
                "timeStep": info["time"],
                "prob_illicit": prob,
                "pred_label": pred,
                "true_label": info["label"],
                "threshold": best_t,
                "top_features": top_feats
            }

        elif mode == "payload":
            fdict = payload.get("features", {})
            missing = [c for c in feature_cols if c not in fdict]
            if missing:
                raise ValueError(f"Missing features: {missing[:10]}...")
            x = np.array([[float(fdict[c]) for c in feature_cols]], dtype="float32")
            prob, pred = _scale_and_predict(x)
            x_scaled = scaler.transform(x)
            contrib = _grad_input_importance(x_scaled)
            top_idx = np.argsort(contrib)[::-1][:10]
            top_feats = [{"feature": feature_cols[i], "score": float(contrib[i])} for i in top_idx]
            return {
                "prob_illicit": prob,
                "pred_label": pred,
                "threshold": best_t,
                "top_features": top_feats
            }
        else:
            raise ValueError("mode must be 'txid' or 'payload'")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _ego_subgraph_png(txid: str, k: int = 2, max_nodes: int = 150) -> (Dict[str, Any], str):
    """
    Returns (subgraph_json, base64_png) for k-hop ego network around txid.
    Limits nodes for readability. Colors center red, neighbors blue.
    """
    if txid not in G:
        raise KeyError(f"txId '{txid}' has no edges in graph.")
    nodes = list(nx.ego_graph(G, txid, radius=k).nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    SG = G.subgraph(nodes).copy()

    # JSON
    nodes_json = [{"id": n} for n in SG.nodes()]
    edges_json = [{"source": u, "target": v} for u, v in SG.edges()]

    # PNG rendering
    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(SG, seed=42, k=1/np.sqrt(max(len(SG),1)))
    node_colors = ["red" if n == txid else "steelblue" for n in SG.nodes()]
    nx.draw_networkx_nodes(SG, pos, node_color=node_colors, node_size=80, alpha=0.9, linewidths=0.3, edgecolors="white")
    nx.draw_networkx_edges(SG, pos, alpha=0.4, width=0.8)
    # label a few around center
    nx.draw_networkx_labels(SG, pos, labels={txid: txid}, font_size=8)
    b64 = fig_to_base64_png()
    return {"nodes": nodes_json, "edges": edges_json}, b64

@app.post("/explain")
async def explain(payload: Dict[str, Any]):
    """
    Input: {"txId": "...", "k": 2, "max_nodes": 150}
    Output:
      - prob_illicit, pred_label, (optional) true_label
      - top_features (grad*input)
      - subgraph {nodes, edges}
      - subgraph_png_base64
    """
    txid = str(payload.get("txId", "")).strip()
    k = int(payload.get("k", 2))
    max_nodes = int(payload.get("max_nodes", 150))
    if not txid:
        raise HTTPException(status_code=400, detail="txId required")

    try:
        info = _get_tx_features(txid)
        prob, pred = _scale_and_predict(info["x"])
        x_scaled = scaler.transform(info["x"])
        contrib = _grad_input_importance(x_scaled)
        top_idx = np.argsort(contrib)[::-1][:10]
        top_feats = [{"feature": feature_cols[i], "score": float(contrib[i])} for i in top_idx]

        sgj, b64 = _ego_subgraph_png(txid, k=k, max_nodes=max_nodes)
        return {
            "txId": txid,
            "timeStep": info["time"],
            "prob_illicit": prob,
            "pred_label": pred,
            "true_label": info["label"],
            "threshold": best_t,
            "top_features": top_feats,
            "subgraph": sgj,
            "subgraph_png_base64": b64
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
