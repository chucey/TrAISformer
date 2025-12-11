#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

def km_to_rad(km: float) -> float:
    return km / EARTH_RADIUS_KM

def nm_to_km(nm: float) -> float:
    return nm * 1.852

def load_positions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"mmsi": "int64", "t_unix": "int64"})
    need = {"mmsi", "t_unix", "lat_deg", "lon_deg"}
    missing = need.difference(df.columns)
    if missing:
        print(f"{csv_path} missing columns: {sorted(missing)}")
    return df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)


def align_window(df: pd.DataFrame, center_ts: int, tol: int) -> pd.DataFrame:
    if tol <= 0:
        return df[df["t_unix"] == center_ts]
    lo, hi = center_ts - tol, center_ts + tol
    window = df[(df["t_unix"] >= lo) & (df["t_unix"] <= hi)]
    if window.empty:
        return window
    window = window.assign(_dt=(window["t_unix"] - center_ts).abs())
    return (window.sort_values(["mmsi", "_dt"]) \
                  .drop_duplicates("mmsi") \
                  .drop(columns=["_dt"]))


def build_edges(df: pd.DataFrame, radius_km: float, time_tolerance_sec: int) -> pd.DataFrame:
    r = km_to_rad(radius_km)
    edges = []
    times = df["t_unix"].unique()
    for t in times:
        g = align_window(df, t, time_tolerance_sec)
        if len(g) <= 1:
            continue
        latlon = np.deg2rad(g[["lat_deg", "lon_deg"]].to_numpy())
        ids = g["mmsi"].to_numpy()
        tree = BallTree(latlon, metric="haversine", leaf_size=50)
        neighborhoods = tree.query_radius(latlon, r=r, return_distance=False)
        for i, inds in enumerate(neighborhoods):
            src = ids[i]
            for idx in inds:
                if idx == i:
                    continue
                dst = ids[idx]
                a, b = (src, dst) if src < dst else (dst, src)
                edges.append((int(t), int(a), int(b)))
    if not edges:
        return pd.DataFrame(columns=["t_unix", "src_mmsi", "dst_mmsi"])
    return (pd.DataFrame(edges, columns=["t_unix", "src_mmsi", "dst_mmsi"]) \
              .drop_duplicates()
              .sort_values(["t_unix", "src_mmsi", "dst_mmsi"]) \
              .reset_index(drop=True))


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def evaluate(pred_edges: pd.DataFrame, gt_edges: pd.DataFrame) -> dict:
    pred_set = set(map(tuple, pred_edges[["t_unix", "src_mmsi", "dst_mmsi"]].to_numpy()))
    gt_set = set(map(tuple, gt_edges[["t_unix", "src_mmsi", "dst_mmsi"]].to_numpy()))
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    metrics = {"tp": tp, "fp": fp, "fn": fn, "recall": recall(tp, fn)}

    rows_t = []
    for t in sorted(set(pred_edges["t_unix"]).union(gt_edges["t_unix"])):
        P = {e for e in pred_set if e[0] == t}
        G = {e for e in gt_set if e[0] == t}
        tp_t = len(P & G)
        fp_t = len(P - G)
        fn_t = len(G - P)
        rows_t.append((t, tp_t, fp_t, fn_t, recall(tp_t, fn_t)))
    per_timestamp = pd.DataFrame(rows_t, columns=["t_unix", "tp", "fp", "fn", "recall"])

    def explode(edges: pd.DataFrame) -> pd.DataFrame:
        a = edges.rename(columns={"src_mmsi": "mmsi", "dst_mmsi": "nbr"})
        b = edges.rename(columns={"dst_mmsi": "mmsi", "src_mmsi": "nbr"})
        return pd.concat([a[["t_unix", "mmsi", "nbr"]], b[["t_unix", "mmsi", "nbr"]]], ignore_index=True)

    inc_pred = explode(pred_edges)
    inc_gt = explode(gt_edges)
    inc_pred["key"] = list(zip(inc_pred["t_unix"], inc_pred["mmsi"], inc_pred["nbr"]))
    inc_gt["key"] = list(zip(inc_gt["t_unix"], inc_gt["mmsi"], inc_gt["nbr"]))

    rows_m = []
    for m in sorted(set(inc_pred["mmsi"]).union(inc_gt["mmsi"])):
        P = set(inc_pred[inc_pred["mmsi"] == m]["key"])
        G = set(inc_gt[inc_gt["mmsi"] == m]["key"])
        tp_m = len(P & G)
        fp_m = len(P - G)
        fn_m = len(G - P)
        rows_m.append((m, tp_m, fp_m, fn_m, recall(tp_m, fn_m)))
    per_mmsi = pd.DataFrame(rows_m, columns=["mmsi", "tp", "fp", "fn", "recall"])

    return {"overall": metrics, "per_timestamp": per_timestamp, "per_mmsi": per_mmsi}


def main():
    ap = argparse.ArgumentParser(description="Evaluate neighbor predictions at scale")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--radius_nm", type=float, default=1.0)
    ap.add_argument("--time_tolerance_sec", type=int, default=0)
    ap.add_argument("--out_dir", default="neighbors_eval_out")
    args = ap.parse_args()

    radius_km = nm_to_km(args.radius_nm)
    df_pred = load_positions(args.pred_csv)
    df_gt = load_positions(args.gt_csv)
    pred_edges = build_edges(df_pred, radius_km, args.time_tolerance_sec)
    gt_edges = build_edges(df_gt, radius_km, args.time_tolerance_sec)
    metrics = evaluate(pred_edges, gt_edges)

    o = metrics["overall"]
    print(f"Overall recall={o['recall']:.3f} (tp={o['tp']}, fp={o['fp']}, fn={o['fn']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_edges.to_csv(out_dir / "pred_edges.csv", index=False)
    gt_edges.to_csv(out_dir / "gt_edges.csv", index=False)
    metrics["per_timestamp"].to_csv(out_dir / "per_timestamp.csv", index=False)
    metrics["per_mmsi"].to_csv(out_dir / "per_mmsi.csv", index=False)
    print(f"Detailed results saved to {out_dir}")


if __name__ == "__main__":
    main()

