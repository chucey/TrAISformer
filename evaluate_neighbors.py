#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

# ---------- helpers ----------
def km_to_rad(km: float) -> float:
    return km / EARTH_RADIUS_KM

def nm_to_km(nm: float) -> float:
    return nm * 1.852

def load_positions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"mmsi": "int64", "t_unix": "int64"})
    need = {"mmsi", "t_unix", "lat_deg", "lon_deg"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)

def positions_to_edges(df: pd.DataFrame, radius_km: float,
                       time_tolerance_sec: int = 0,
                       undirected: bool = True) -> pd.DataFrame:
    """
    Build neighbor edges at each timestamp (or ± tol sec).
    Returns DataFrame with cols: t_unix, src_mmsi, dst_mmsi
    If undirected=True, edges are canonicalized (min, max) and deduped.
    """
    r = km_to_rad(radius_km)
    out_rows = []

    # If tolerance==0, group by exact t. Otherwise, we’ll align per t by nearest-in-time per MMSI.
    if time_tolerance_sec <= 0:
        grouped = df.groupby("t_unix", sort=True)
        for t, g in grouped:
            if len(g) <= 1:
                continue
            latlon_rad = np.deg2rad(g[["lat_deg", "lon_deg"]].to_numpy())
            mmsis = g["mmsi"].to_numpy()
            tree = BallTree(latlon_rad, metric="haversine")
            inds_list, _ = tree.query_radius(latlon_rad, r=r, return_distance=True, sort_results=False)
            for i, inds in enumerate(inds_list):
                src = mmsis[i]
                for j in inds:
                    if j == i:
                        continue
                    dst = mmsis[j]
                    if undirected:
                        a, b = (src, dst) if src < dst else (dst, src)
                        out_rows.append((int(t), int(a), int(b)))
                    else:
                        out_rows.append((int(t), int(src), int(dst)))
    else:
        # tolerance path: for each target time t, pick one nearest-time row per MMSI within ±tol
        times = df["t_unix"].unique()
        df = df.sort_values(["t_unix", "mmsi"])
        for t in times:
            lo, hi = t - time_tolerance_sec, t + time_tolerance_sec
            window = df[(df["t_unix"] >= lo) & (df["t_unix"] <= hi)]
            if window.empty:
                continue
            window = window.assign(_dt=(window["t_unix"] - t).abs())
            # one row per MMSI: the closest-in-time sample to t
            g = (window.sort_values(["mmsi", "_dt"])
                        .drop_duplicates(subset=["mmsi"], keep="first")
                        .drop(columns=["_dt"]))
            if len(g) <= 1:
                continue
            latlon_rad = np.deg2rad(g[["lat_deg", "lon_deg"]].to_numpy())
            mmsis = g["mmsi"].to_numpy()
            tree = BallTree(latlon_rad, metric="haversine")
            inds_list, _ = tree.query_radius(latlon_rad, r=r, return_distance=True, sort_results=False)
            for i, inds in enumerate(inds_list):
                src = mmsis[i]
                for j in inds:
                    if j == i: continue
                    dst = mmsis[j]
                    if undirected:
                        a, b = (src, dst) if src < dst else (dst, src)
                        out_rows.append((int(t), int(a), int(b)))
                    else:
                        out_rows.append((int(t), int(src), int(dst)))

    edges = pd.DataFrame(out_rows, columns=["t_unix", "src_mmsi", "dst_mmsi"])
    if edges.empty:
        return edges
    edges = edges.drop_duplicates().sort_values(["t_unix", "src_mmsi", "dst_mmsi"]).reset_index(drop=True)
    return edges

def f1(prec, rec):
    return 0.0 if (prec == 0.0 and rec == 0.0) else 2*prec*rec/(prec+rec)

# ---------- evaluation ----------
def evaluate_neighbor_sets(pred_edges: pd.DataFrame,
                           gt_edges: pd.DataFrame,
                           restrict_mmsi: int | None = None) -> dict:
    """
    Compute precision/recall/F1 overall, and per-timestamp summaries.
    If restrict_mmsi is given, only keep edges that involve that MMSI.
    """
    if restrict_mmsi is not None:
        mask_pred = (pred_edges["src_mmsi"] == restrict_mmsi) | (pred_edges["dst_mmsi"] == restrict_mmsi)
        mask_gt   = (gt_edges["src_mmsi"]   == restrict_mmsi) | (gt_edges["dst_mmsi"]   == restrict_mmsi)
        pred_edges = pred_edges[mask_pred].copy()
        gt_edges   = gt_edges[mask_gt].copy()

    # key = (t, a, b)
    def keyify(df):
        return set(map(tuple, df[["t_unix", "src_mmsi", "dst_mmsi"]].to_numpy()))

    P = keyify(pred_edges)
    G = keyify(gt_edges)

    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    prec = 0.0 if (tp+fp)==0 else tp/(tp+fp)
    rec  = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    f1_  = f1(prec, rec)

    # per-timestamp breakdown
    all_t = sorted(set(pred_edges["t_unix"]).union(set(gt_edges["t_unix"])))
    rows_t = []
    for t in all_t:
        Pt = {e for e in P if e[0]==t}
        Gt = {e for e in G if e[0]==t}
        tp_t = len(Pt & Gt)
        fp_t = len(Pt - Gt)
        fn_t = len(Gt - Pt)
        pr_t = 0.0 if (tp_t+fp_t)==0 else tp_t/(tp_t+fp_t)
        rc_t = 0.0 if (tp_t+fn_t)==0 else tp_t/(tp_t+fn_t)
        f1_t = f1(pr_t, rc_t)
        rows_t.append((t, tp_t, fp_t, fn_t, pr_t, rc_t, f1_t))
    per_t = pd.DataFrame(rows_t, columns=["t_unix","tp","fp","fn","precision","recall","f1"]).sort_values("t_unix")

    # per-MMSI degree-based precision/recall: expand edges into directed incidents for both endpoints
    def explode_incidents(edges):
        a = edges.rename(columns={"src_mmsi":"mmsi","dst_mmsi":"nbr"}).copy()
        b = edges.rename(columns={"dst_mmsi":"mmsi","src_mmsi":"nbr"}).copy()
        return pd.concat([a[["t_unix","mmsi","nbr"]], b[["t_unix","mmsi","nbr"]]], ignore_index=True)

    incP = explode_incidents(pred_edges)
    incG = explode_incidents(gt_edges)
    incP["key"] = list(zip(incP["t_unix"], incP["mmsi"], incP["nbr"]))
    incG["key"] = list(zip(incG["t_unix"], incG["mmsi"], incG["nbr"]))
    KP = set(incP["key"]); KG = set(incG["key"])
    # per-mmsi: compute metrics on incidents
    ms = sorted(set(incP["mmsi"]).union(set(incG["mmsi"])))
    rows_m = []
    for m in ms:
        KPm = {k for k in KP if k[1]==m}
        KGm = {k for k in KG if k[1]==m}
        tp_m = len(KPm & KGm)
        fp_m = len(KPm - KGm)
        fn_m = len(KGm - KPm)
        pr_m = 0.0 if (tp_m+fp_m)==0 else tp_m/(tp_m+fp_m)
        rc_m = 0.0 if (tp_m+fn_m)==0 else tp_m/(tp_m+fn_m)
        f1_m = f1(pr_m, rc_m)
        rows_m.append((m, tp_m, fp_m, fn_m, pr_m, rc_m, f1_m))
    per_mmsi = pd.DataFrame(rows_m, columns=["mmsi","tp","fp","fn","precision","recall","f1"]).sort_values("mmsi")

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1_},
        "per_timestamp": per_t,
        "per_mmsi": per_mmsi,
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted neighbors vs ground truth neighbors")
    ap.add_argument("--pred_csv", required=True, help="TrAISformer PREDICTED positions CSV")
    ap.add_argument("--gt_csv", required=True, help="Ground-truth positions CSV")
    ap.add_argument("--radius_nm", type=float, default=None, help="Neighbor radius in nautical miles")
    ap.add_argument("--radius_km", type=float, default=None, help="Neighbor radius in kilometers")
    ap.add_argument("--time_tolerance_sec", type=int, default=0, help="± seconds to align timestamps (0 = exact)")
    ap.add_argument("--restrict_mmsi", type=int, default=None, help="If set, evaluate only neighbors involving this MMSI")
    ap.add_argument("--out_dir", default="neighbors_eval_out", help="Directory to write detailed outputs")
    args = ap.parse_args()

    radius_km = nm_to_km(args.radius_nm) if args.radius_nm is not None else (args.radius_km or 1.852)

    df_pred = load_positions(args.pred_csv)
    df_gt   = load_positions(args.gt_csv)

    # Build neighbor edges under identical conditions
    pred_edges = positions_to_edges(df_pred, radius_km=radius_km,
                                    time_tolerance_sec=args.time_tolerance_sec, undirected=True)
    gt_edges   = positions_to_edges(df_gt, radius_km=radius_km,
                                    time_tolerance_sec=args.time_tolerance_sec, undirected=True)

    metrics = evaluate_neighbor_sets(pred_edges, gt_edges, restrict_mmsi=args.restrict_mmsi)

    # Print overall
    o = metrics["overall"]
    print(f"Overall:  precision={o['precision']:.3f}  recall={o['recall']:.3f}  f1={o['f1']:.3f}  (tp={o['tp']}, fp={o['fp']}, fn={o['fn']})")

    # Save detailed outputs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics["per_timestamp"].to_csv(out_dir / "per_timestamp.csv", index=False)
    metrics["per_mmsi"].to_csv(out_dir / "per_mmsi.csv", index=False)
    pred_edges.to_csv(out_dir / "pred_edges.csv", index=False)
    gt_edges.to_csv(out_dir / "gt_edges.csv", index=False)
    print(f"Wrote details to: {out_dir}/")

if __name__ == "__main__":
    main()