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

def get_neighbors_for_target(df, target_mmsi, radius_km_value, time_tolerance_sec=0):
    r = km_to_rad(radius_km_value)
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

def evaluate_target_frames(df_pred: pd.DataFrame,
                           df_gt: pd.DataFrame,
                           target_mmsi: int,
                           radius_distance_km: float,
                           time_tolerance_sec: int = 0):
    pred_nbrs = get_neighbors_for_target(df_pred, target_mmsi, radius_distance_km, time_tolerance_sec)
    gt_nbrs   = get_neighbors_for_target(df_gt,   target_mmsi, radius_distance_km, time_tolerance_sec)

    common_times = sorted(set(pred_nbrs) & set(gt_nbrs))
    tp = fp = fn = 0
    per_t_rows = []
    for t in common_times:
        P, G = pred_nbrs[t], gt_nbrs[t]
        tp_t = len(P & G)
        fp_t = len(P - G)
        fn_t = len(G - P)
        rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        tp += tp_t; fp += fp_t; fn += fn_t
        per_t_rows.append((t, tp_t, fp_t, fn_t, rec_t))

    overall_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

    per_t = pd.DataFrame(per_t_rows, columns=["t_unix","tp","fp","fn","recall"])
    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "recall": overall_rec},
        "per_t": per_t
    }

def evaluate_target(pred_csv, gt_csv, target_mmsi, radius_distance_km, time_tolerance_sec=0):
    df_pred = load_positions(pred_csv)
    df_gt = load_positions(gt_csv)
    return evaluate_target_frames(df_pred, df_gt, target_mmsi, radius_distance_km, time_tolerance_sec)

def main():
    ap = argparse.ArgumentParser(description="Fast target-vessel neighbor evaluation")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--target_mmsi", type=int, default=None)
    ap.add_argument("--neighbor_csv", type=Path, default=Path("neighbors_out/neighbor_list.csv"))
    ap.add_argument("--max_targets", type=int, default=10)
    ap.add_argument("--radius_nm", type=float, default=5.0)
    ap.add_argument("--time_tolerance_sec", type=int, default=300)
    ap.add_argument("--out_dir", default="neighbors_eval_fast")
    args = ap.parse_args()

    radius_distance_km = nm_to_km(args.radius_nm)
    df_pred = load_positions(args.pred_csv)
    df_gt = load_positions(args.gt_csv)

    if args.target_mmsi is not None:
        target_mmsis = [args.target_mmsi]
    else:
        if not args.neighbor_csv.exists():
            raise SystemExit(f"Neighbor CSV not found: {args.neighbor_csv}")
        df_neighbors = pd.read_csv(args.neighbor_csv)
        if "target_mmsi" not in df_neighbors.columns:
            raise SystemExit(f"Neighbor CSV {args.neighbor_csv} missing 'target_mmsi' column")
        target_mmsis = []
        for val in df_neighbors["target_mmsi"].dropna().astype(int).tolist():
            if val not in target_mmsis:
                target_mmsis.append(val)
            if len(target_mmsis) >= args.max_targets:
                break
        if not target_mmsis:
            raise SystemExit(f"Neighbor CSV {args.neighbor_csv} did not provide any MMSIs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for mmsi in target_mmsis:
        metrics = evaluate_target_frames(df_pred, df_gt, mmsi, radius_distance_km, args.time_tolerance_sec)
        metrics["per_t"].to_csv(out_dir / f"per_t_{mmsi}.csv", index=False)
        o = metrics["overall"]
        summary_rows.append({
            "mmsi": mmsi,
            "tp": o["tp"],
            "fp": o["fp"],
            "fn": o["fn"],
            "recall": o["recall"]
        })
        print(f"[{mmsi}] Recall={o['recall']:.3f} | TP={o['tp']} FP={o['fp']} FN={o['fn']}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved per-target summaries to {out_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()
