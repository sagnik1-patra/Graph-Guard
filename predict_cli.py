#!/usr/bin/env python
import os, csv, json, pickle, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

def robust_read_csv(path, expected_min_cols=2):
    encodings = ["utf-8","utf-8-sig","cp1252","latin1"]
    delims    = [",",";","\t","|"]
    try:
        with open(path, "rb") as f:
            head = f.read(8192).decode("latin1", errors="ignore")
        sniffed = csv.Sniffer().sniff(head)
        if sniffed.delimiter in delims:
            delims = [sniffed.delimiter] + [d for d in delims if d != sniffed.delimiter]
    except Exception:
        pass
    last_err = None
    for enc in encodings:
        for sep in delims:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= expected_min_cols:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not parse {path}. Last error: {last_err}")

def main():
    p = argparse.ArgumentParser(description="GraphGuard prediction CLI (Elliptic)")
    p.add_argument("--base", default=r"C:\Users\sagni\Downloads\GraphGuard")
    p.add_argument("--features_csv", default=r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_features.csv")
    p.add_argument("--classes_csv",  default=r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_classes.csv")
    p.add_argument("--edges_csv",    default=r"C:\Users\sagni\Downloads\GraphGuard\archive (1)\elliptic_bitcoin_dataset\elliptic_txs_edgelist.csv")
    p.add_argument("--mode", choices=["dataset","txids"], default="dataset")
    p.add_argument("--subset", choices=["train","val","test","all"], default="test")
    p.add_argument("--txids", help="Comma-separated txIds (only for --mode txids)")
    args = p.parse_args()

    BASE = args.base
    PREPROC_PKL = os.path.join(BASE, "preprocessor.pkl")
    H5_PATH     = os.path.join(BASE, "model.h5")
    KERAS_PATH  = os.path.join(BASE, "model.keras")
    THRESH_PATH = os.path.join(BASE, "threshold.json")

    with open(PREPROC_PKL, "rb") as f:
        preproc = pickle.load(f)
    feature_cols = list(preproc["feature_columns"])
    scaler       = preproc["scaler"]
    time_col     = preproc["time_column"]
    txid_col     = preproc["txid_column"]
    train_steps  = set(preproc["splits"]["train_steps"])
    val_steps    = set(preproc["splits"]["val_steps"])
    test_steps   = set(preproc["splits"]["test_steps"])

    with open(THRESH_PATH, "r", encoding="utf-8") as f:
        best_t = float(json.load(f)["best_threshold"])

    # Load data
    df_feat = robust_read_csv(args.features_csv, expected_min_cols=3)
    df_cls  = robust_read_csv(args.classes_csv,  expected_min_cols=2)
    df_edge = robust_read_csv(args.edges_csv,    expected_min_cols=2)

    feat_cols = list(df_feat.columns)
    tx_col_feat   = feat_cols[0]
    time_col_feat = feat_cols[1]
    assert tx_col_feat == txid_col, f"TX ID column mismatch: {tx_col_feat} vs {txid_col}"
    assert time_col_feat == time_col, f"Time column mismatch: {time_col_feat} vs {time_col}"

    cls_cols  = list(df_cls.columns)
    tx_col_cls = cls_cols[0]
    class_col  = cls_cols[1]

    edge_cols = list(df_edge.columns)
    src_col, dst_col = edge_cols[0], edge_cols[1]

    # Force TXID to string
    df_feat[tx_col_feat] = df_feat[tx_col_feat].astype(str)
    df_cls[tx_col_cls]   = df_cls[tx_col_cls].astype(str)
    df_edge[src_col]     = df_edge[src_col].astype(str)
    df_edge[dst_col]     = df_edge[dst_col].astype(str)

    # Labels (optional)
    df_cls[class_col] = df_cls[class_col].astype(str).str.lower().str.strip()
    label_map = {"1":0, "2":1, "licit":0, "illicit":1}
    df_cls["label"] = df_cls[class_col].map(label_map)
    df_cls["label"] = df_cls["label"].astype("Int64")  # allow NA

    # Degrees
    in_deg  = df_edge.groupby(dst_col).size().rename("in_degree")
    out_deg = df_edge.groupby(src_col).size().rename("out_degree")
    deg_df  = pd.concat([in_deg, out_deg], axis=1).fillna(0.0).reset_index()
    deg_df.rename(columns={deg_df.columns[0]: tx_col_feat}, inplace=True)

    # Merge
    df_feat[time_col_feat] = pd.to_numeric(df_feat[time_col_feat], errors="coerce")
    df = df_feat.merge(deg_df, on=tx_col_feat, how="left")
    df[["in_degree","out_degree"]] = df[["in_degree","out_degree"]].fillna(0.0)
    df = df.merge(df_cls[[tx_col_cls,"label"]], left_on=tx_col_feat, right_on=tx_col_cls, how="left")
    if tx_col_cls in df.columns and tx_col_cls != tx_col_feat:
        df = df.drop(columns=[tx_col_cls])

    # Keep numeric features
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + [time_col_feat]).reset_index(drop=True)

    # Subset
    if args.mode == "dataset":
        if args.subset == "train":
            mask = df[time_col].isin(train_steps)
        elif args.subset == "val":
            mask = df[time_col].isin(val_steps)
        elif args.subset == "test":
            mask = df[time_col].isin(test_steps)
        else:
            mask = np.ones(len(df), dtype=bool)
        df_slice = df[mask].copy()
    else:
        # txids mode
        if not args.txids:
            raise SystemExit("--txids is required for --mode txids")
        want = set([s.strip() for s in args.txids.split(",") if s.strip()])
        df_slice = df[df[txid_col].isin(want)].copy()
        if df_slice.empty:
            raise SystemExit("None of the requested txIds were found in features CSV.")

    # Load model
    model = None
    if os.path.exists(KERAS_PATH):
        try:
            model = tf.keras.models.load_model(KERAS_PATH, safe_mode=False)
        except Exception:
            model = None
    if model is None and os.path.exists(H5_PATH):
        model = tf.keras.models.load_model(H5_PATH)
    if model is None:
        raise SystemExit("No model found (model.keras / model.h5).")

    # Predict
    X = scaler.transform(df_slice[feature_cols].values)
    prob = model.predict(X, batch_size=4096, verbose=0).ravel()
    pred = (prob >= best_t).astype(int)

    out = pd.DataFrame({
        "txId": df_slice[txid_col].values,
        "timeStep": df_slice[time_col].values,
        "prob_illicit": prob,
        "pred_label": pred
    })
    if "label" in df_slice.columns:
        out["true_label"] = df_slice["label"].astype("Int64").values

    # Save
    name = f"predictions_{args.mode}_{args.subset}.csv" if args.mode=="dataset" else "predictions_txids.csv"
    csv_path = os.path.join(BASE, name)
    out.to_csv(csv_path, index=False)
    jsonl_path = csv_path.replace(".csv",".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in out.iterrows():
            rec = row.to_dict()
            rec["pred_label_name"] = "illicit" if rec["pred_label"]==1 else "licit"
            f.write(json.dumps(rec)+"\n")

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {jsonl_path}")
    print(out.head())

if __name__ == "__main__":
    main()
