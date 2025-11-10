#!/usr/bin/env python3
"""
Fast neighbor evaluation — focuses only on a single target vessel MMSI.
"""

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
    for c in ["mmsi", "t_unix", "lat_deg", "lon_deg"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")
    return df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)

def get_neighbors_for_target(df, target_mmsi, radius_km, time_tolerance_sec=0):
    """Return dict: t_unix -> set(neighbor_mmsi) for target vessel."""
    r = km_to_rad(radius_km)
    df_target = df[df["mmsi"] == target_mmsi]
    if df_target.empty:
        raise ValueError(f"No entries for MMSI {target_mmsi}")

    neighbor_dict = {}
    for t, tgt_grp in df_target.groupby("t_unix"):
        # filter others at same or nearby timestamp
        if time_tolerance_sec > 0:
            lo, hi = t - time_tolerance_sec, t + time_tolerance_sec
            df_oth = df[(df["t_unix"] >= lo) & (df["t_unix"] <= hi) & (df["mmsi"] != target_mmsi)]
        else:
            df_oth = df[(df["t_unix"] == t) & (df["mmsi"] != target_mmsi)]
        if df_oth.empty:
            neighbor_dict[t] = set()
            continue

        tree = BallTree(np.deg2rad(df_oth[["lat_deg", "lon_deg"]]), metric="haversine")
        pts_target = np.deg2rad(tgt_grp[["lat_deg", "lon_deg"]])
        inds_list = tree.query_radius(pts_target, r=r)
        # flatten since usually only one point per timestamp
        inds = inds_list[0] if len(inds_list) > 0 else []
        nbrs = set(df_oth.iloc[inds]["mmsi"].astype(int))
        neighbor_dict[t] = nbrs
    return neighbor_dict

def evaluate_target(pred_csv, gt_csv, target_mmsi, radius_km, time_tolerance_sec=0):
    df_pred = load_positions(pred_csv)
    df_gt = load_positions(gt_csv)

    pred_nbrs = get_neighbors_for_target(df_pred, target_mmsi, radius_km, time_tolerance_sec)
    gt_nbrs   = get_neighbors_for_target(df_gt,   target_mmsi, radius_km, time_tolerance_sec)

    common_times = sorted(set(pred_nbrs) & set(gt_nbrs))
    tp = fp = fn = 0
    per_t_rows = []
    for t in common_times:
        P, G = pred_nbrs[t], gt_nbrs[t]
        tp_t = len(P & G)
        fp_t = len(P - G)
        fn_t = len(G - P)
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t   = 2*prec_t*rec_t/(prec_t+rec_t) if (prec_t+rec_t)>0 else 0
        tp += tp_t; fp += fp_t; fn += fn_t
        per_t_rows.append((t, tp_t, fp_t, fn_t, prec_t, rec_t, f1_t))

    overall_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_f1   = 2*overall_prec*overall_rec/(overall_prec+overall_rec) if (overall_prec+overall_rec)>0 else 0

    per_t = pd.DataFrame(per_t_rows, columns=["t_unix","tp","fp","fn","precision","recall","f1"])
    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": overall_prec, "recall": overall_rec, "f1": overall_f1},
        "per_t": per_t
    }

def main():
    ap = argparse.ArgumentParser(description="Fast target-vessel neighbor evaluation")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--target_mmsi", type=int, required=True)
    ap.add_argument("--radius_nm", type=float, default=None)
    ap.add_argument("--radius_km", type=float, default=None)
    ap.add_argument("--time_tolerance_sec", type=int, default=0)
    ap.add_argument("--out_dir", default="neighbors_eval_fast")
    args = ap.parse_args()

    radius_km = nm_to_km(args.radius_nm) if args.radius_nm else (args.radius_km or 1.852)
    metrics = evaluate_target(args.pred_csv, args.gt_csv, args.target_mmsi, radius_km, args.time_tolerance_sec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics["per_t"].to_csv(out_dir / f"per_t_{args.target_mmsi}.csv", index=False)
    o = metrics["overall"]
    print(f"[{args.target_mmsi}] Precision={o['precision']:.3f} Recall={o['recall']:.3f} F1={o['f1']:.3f}")
    print(f"TP={o['tp']} FP={o['fp']} FN={o['fn']} | Saved results to {out_dir}")

if __name__ == "__main__":
    main()