#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import sys

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
    print(f"Loading {csv_path}...")
    sys.stdout.flush()  # Ensure output appears in SLURM logs immediately
    df = pd.read_csv(csv_path, dtype={"mmsi": "int64", "t_unix": "int64"})
    need = {"mmsi", "t_unix", "lat_deg", "lon_deg"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    df = df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} position records")
    sys.stdout.flush()
    return df

def positions_to_edges_optimized(df: pd.DataFrame, radius_km: float,
                                time_tolerance_sec: int = 0,
                                undirected: bool = True) -> pd.DataFrame:
    """
    OPTIMIZED version: Build neighbor edges at each timestamp with performance improvements.
    """
    print(f"Building neighbor edges with radius {radius_km:.3f} km...")
    sys.stdout.flush()
    start_time = time.time()
    
    r = km_to_rad(radius_km)
    
    # Pre-allocate for better performance
    edge_batches = []
    
    # If tolerance==0, group by exact t. Otherwise, we'll align per t by nearest-in-time per MMSI.
    if time_tolerance_sec <= 0:
        grouped = df.groupby("t_unix", sort=True)
        total_timestamps = len(grouped)
        print(f"Processing {total_timestamps:,} timestamps...")
        sys.stdout.flush()
        
        processed = 0
        for t, g in grouped:
            if len(g) <= 1:
                continue
                
            # Progress reporting
            processed += 1
            if processed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (total_timestamps - processed) / rate if rate > 0 else 0
                print(f"  Progress: {processed:,}/{total_timestamps:,} timestamps "
                      f"({100*processed/total_timestamps:.1f}%) - "
                      f"Est. remaining: {remaining/60:.1f}m")
                sys.stdout.flush()
            
            # Optimized BallTree usage
            latlon_rad = np.deg2rad(g[["lat_deg", "lon_deg"]].to_numpy())
            mmsis = g["mmsi"].to_numpy()
            
            # OPTIMIZATION: larger leaf_size, no distance calculation
            tree = BallTree(latlon_rad, metric="haversine", leaf_size=50)
            inds_list = tree.query_radius(latlon_rad, r=r, return_distance=False, sort_results=False)
            
            # OPTIMIZATION: vectorized edge creation
            timestamp_edges = []
            for i, inds in enumerate(inds_list):
                src = mmsis[i]
                # Filter out self-connections
                neighbor_inds = inds[inds != i]
                if len(neighbor_inds) == 0:
                    continue
                    
                dst_vessels = mmsis[neighbor_inds]
                
                if undirected:
                    # Create canonicalized edges (src < dst)
                    for dst in dst_vessels:
                        a, b = (src, dst) if src < dst else (dst, src)
                        timestamp_edges.append((int(t), int(a), int(b)))
                else:
                    # Create directed edges
                    for dst in dst_vessels:
                        timestamp_edges.append((int(t), int(src), int(dst)))
            
            if timestamp_edges:
                edge_batches.append(timestamp_edges)
    
    else:
        # OPTIMIZATION: Pre-sort and index for tolerance path
        print(f"Using time tolerance ±{time_tolerance_sec} seconds...")
        sys.stdout.flush()
        times = np.unique(df["t_unix"].values)
        df_sorted = df.sort_values(["t_unix", "mmsi"])
        total_times = len(times)
        print(f"Processing {total_times:,} target times...")
        sys.stdout.flush()
        
        processed = 0
        for t in times:
            processed += 1
            if processed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (total_times - processed) / rate if rate > 0 else 0
                print(f"  Progress: {processed:,}/{total_times:,} times "
                      f"({100*processed/total_times:.1f}%) - "
                      f"Est. remaining: {remaining/60:.1f}m")
                sys.stdout.flush()
            
            lo, hi = t - time_tolerance_sec, t + time_tolerance_sec
            window = df_sorted[(df_sorted["t_unix"] >= lo) & (df_sorted["t_unix"] <= hi)]
            if window.empty:
                continue
                
            # OPTIMIZATION: vectorized time difference calculation
            window = window.assign(_dt=np.abs(window["t_unix"].values - t))
            g = (window.sort_values(["mmsi", "_dt"])
                        .drop_duplicates(subset=["mmsi"], keep="first")
                        .drop(columns=["_dt"]))
            
            if len(g) <= 1:
                continue
            
            # Same optimized BallTree processing as above
            latlon_rad = np.deg2rad(g[["lat_deg", "lon_deg"]].to_numpy())
            mmsis = g["mmsi"].to_numpy()
            tree = BallTree(latlon_rad, metric="haversine", leaf_size=50)
            inds_list = tree.query_radius(latlon_rad, r=r, return_distance=False, sort_results=False)
            
            timestamp_edges = []
            for i, inds in enumerate(inds_list):
                src = mmsis[i]
                neighbor_inds = inds[inds != i]
                if len(neighbor_inds) == 0:
                    continue
                    
                dst_vessels = mmsis[neighbor_inds]
                
                if undirected:
                    for dst in dst_vessels:
                        a, b = (src, dst) if src < dst else (dst, src)
                        timestamp_edges.append((int(t), int(a), int(b)))
                else:
                    for dst in dst_vessels:
                        timestamp_edges.append((int(t), int(src), int(dst)))
            
            if timestamp_edges:
                edge_batches.append(timestamp_edges)

    # OPTIMIZATION: Efficient DataFrame creation
    if not edge_batches:
        return pd.DataFrame(columns=["t_unix", "src_mmsi", "dst_mmsi"])
    
    print("Consolidating edges...")
    sys.stdout.flush()
    all_edges = []
    for batch in edge_batches:
        all_edges.extend(batch)
    
    edges = pd.DataFrame(all_edges, columns=["t_unix", "src_mmsi", "dst_mmsi"])
    
    # OPTIMIZATION: More efficient deduplication
    print(f"Deduplicating {len(edges):,} edges...")
    sys.stdout.flush()
    edges = edges.drop_duplicates()
    edges = edges.sort_values(["t_unix", "src_mmsi", "dst_mmsi"]).reset_index(drop=True)
    
    elapsed = time.time() - start_time
    print(f"Built {len(edges):,} unique edges in {elapsed:.1f} seconds")
    sys.stdout.flush()
    return edges

def f1(prec, rec):
    return 0.0 if (prec == 0.0 and rec == 0.0) else 2*prec*rec/(prec+rec)

# ---------- evaluation ----------
def evaluate_neighbor_sets_optimized(pred_edges: pd.DataFrame,
                                    gt_edges: pd.DataFrame,
                                    restrict_mmsi: int | None = None) -> dict:
    """
    HEAVILY OPTIMIZED version: Compute precision/recall/F1 with major performance improvements.
    """
    print("Starting evaluation (optimized version)...")
    print(f"Input data: {len(pred_edges):,} predicted edges, {len(gt_edges):,} GT edges")
    eval_start = time.time()
    sys.stdout.flush()
    
    if restrict_mmsi is not None:
        print(f"Restricting to MMSI {restrict_mmsi}...")
        mask_pred = (pred_edges["src_mmsi"] == restrict_mmsi) | (pred_edges["dst_mmsi"] == restrict_mmsi)
        mask_gt   = (gt_edges["src_mmsi"]   == restrict_mmsi) | (gt_edges["dst_mmsi"]   == restrict_mmsi)
        pred_edges = pred_edges[mask_pred].copy()
        gt_edges   = gt_edges[mask_gt].copy()
        print(f"After restriction: {len(pred_edges):,} predicted edges, {len(gt_edges):,} GT edges")
        sys.stdout.flush()

    # POTENTIAL BOTTLENECK 1: Converting large DataFrames to numpy arrays
    print("Converting DataFrames to numpy arrays...")
    start_convert = time.time()
    sys.stdout.flush()
    
    # Convert to numpy arrays first for vectorized operations
    pred_array = pred_edges[["t_unix", "src_mmsi", "dst_mmsi"]].to_numpy()
    gt_array = gt_edges[["t_unix", "src_mmsi", "dst_mmsi"]].to_numpy()
    
    convert_time = time.time() - start_convert
    print(f"Array conversion completed in {convert_time:.1f} seconds")
    sys.stdout.flush()
    
    # POTENTIAL BOTTLENECK 2: Creating massive frozensets from millions of tuples
    print("Creating edge sets (this may take several minutes for large datasets)...")
    start_sets = time.time()
    sys.stdout.flush()
    
    # For very large datasets, use a more memory-efficient approach
    if len(pred_array) > 10_000_000 or len(gt_array) > 10_000_000:
        print("Large dataset detected - using memory-efficient set creation...")
        sys.stdout.flush()
        
        # Process in chunks to avoid memory issues
        chunk_size = 1_000_000
        P_chunks = []
        for i in range(0, len(pred_array), chunk_size):
            chunk = pred_array[i:i+chunk_size]
            P_chunks.append(set(map(tuple, chunk)))
            if i % (chunk_size * 5) == 0:
                print(f"  Processed {min(i+chunk_size, len(pred_array)):,}/{len(pred_array):,} predicted edges")
                sys.stdout.flush()
        
        G_chunks = []
        for i in range(0, len(gt_array), chunk_size):
            chunk = gt_array[i:i+chunk_size]
            G_chunks.append(set(map(tuple, chunk)))
            if i % (chunk_size * 5) == 0:
                print(f"  Processed {min(i+chunk_size, len(gt_array)):,}/{len(gt_array):,} GT edges")
                sys.stdout.flush()
        
        # Combine chunks
        print("Combining edge set chunks...")
        sys.stdout.flush()
        P = set()
        for chunk in P_chunks:
            P.update(chunk)
        G = set()
        for chunk in G_chunks:
            G.update(chunk)
    else:
        # Original approach for smaller datasets
        P = frozenset(map(tuple, pred_array))
        G = frozenset(map(tuple, gt_array))
    
    sets_time = time.time() - start_sets
    print(f"Edge sets created in {sets_time:.1f} seconds")
    print(f"Predicted edges: {len(P):,}, GT edges: {len(G):,}")
    sys.stdout.flush()

    # POTENTIAL BOTTLENECK 3: Set intersection operations on massive sets
    print("Computing overall metrics (set operations on large sets)...")
    start_metrics = time.time()
    sys.stdout.flush()
    
    tp = len(P & G)
    print(f"True positives computed: {tp:,}")
    sys.stdout.flush()
    
    fp = len(P - G)
    print(f"False positives computed: {fp:,}")
    sys.stdout.flush()
    
    fn = len(G - P)
    print(f"False negatives computed: {fn:,}")
    sys.stdout.flush()
    
    prec = 0.0 if (tp+fp)==0 else tp/(tp+fp)
    rec  = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    f1_  = f1(prec, rec)
    
    metrics_time = time.time() - start_metrics
    print(f"Overall metrics computed in {metrics_time:.1f} seconds")
    sys.stdout.flush()

    # POTENTIAL BOTTLENECK 4: Per-timestamp processing
    print("Computing per-timestamp metrics...")
    start_timestamp = time.time()
    sys.stdout.flush()
    
    # Get unique timestamps more efficiently
    pred_times = set(pred_edges["t_unix"].unique())
    gt_times = set(gt_edges["t_unix"].unique())
    all_t = np.array(sorted(pred_times.union(gt_times)))
    
    print(f"Processing {len(all_t):,} unique timestamps...")
    sys.stdout.flush()
    
    # OPTIMIZATION: Skip per-timestamp if too many timestamps
    if len(all_t) > 50_000:
        print(f"WARNING: {len(all_t):,} timestamps detected. Skipping detailed per-timestamp analysis for performance.")
        print("Consider using --restrict_mmsi or processing smaller time windows.")
        per_t = pd.DataFrame(columns=["t_unix","tp","fp","fn","precision","recall","f1"])
    else:
        # Pre-group edges by timestamp for faster lookup
        pred_by_t = {}
        gt_by_t = {}
        
        print("Grouping edges by timestamp...")
        group_start = time.time()
        sys.stdout.flush()
        
        # Vectorized grouping with progress tracking
        for i, t in enumerate(all_t):
            if i % 5000 == 0:
                print(f"  Grouping timestamp {i:,}/{len(all_t):,}")
                sys.stdout.flush()
            
            pred_mask = pred_array[:, 0] == t
            gt_mask = gt_array[:, 0] == t
            pred_by_t[t] = set(map(tuple, pred_array[pred_mask]))
            gt_by_t[t] = set(map(tuple, gt_array[gt_mask]))
        
        group_time = time.time() - group_start
        print(f"Timestamp grouping completed in {group_time:.1f} seconds")
        sys.stdout.flush()
        
        # Vectorized metric computation
        def compute_metrics_batch(timestamps):
            results = []
            for t in timestamps:
                Pt = pred_by_t.get(t, set())
                Gt = gt_by_t.get(t, set())
                tp_t = len(Pt & Gt)
                fp_t = len(Pt - Gt)
                fn_t = len(Gt - Pt)
                pr_t = 0.0 if (tp_t+fp_t)==0 else tp_t/(tp_t+fp_t)
                rc_t = 0.0 if (tp_t+fn_t)==0 else tp_t/(tp_t+fn_t)
                f1_t = f1(pr_t, rc_t)
                results.append((t, tp_t, fp_t, fn_t, pr_t, rc_t, f1_t))
            return results
        
        # Process timestamps in chunks for better performance
        chunk_size = 1000
        rows_t = []
        for i in range(0, len(all_t), chunk_size):
            chunk = all_t[i:i+chunk_size]
            rows_t.extend(compute_metrics_batch(chunk))
            if i % (chunk_size * 10) == 0:
                print(f"  Processed {min(i+chunk_size, len(all_t)):,}/{len(all_t):,} timestamps")
                sys.stdout.flush()
        
        per_t = pd.DataFrame(rows_t, columns=["t_unix","tp","fp","fn","precision","recall","f1"])
    
    timestamp_time = time.time() - start_timestamp
    print(f"Per-timestamp metrics completed in {timestamp_time:.1f} seconds")
    sys.stdout.flush()

    # POTENTIAL BOTTLENECK 5: Per-MMSI processing (often the biggest bottleneck)
    print("Computing per-MMSI metrics...")
    start_mmsi = time.time()
    sys.stdout.flush()
    
    # Create incident arrays more efficiently using numpy operations
    print("Creating incident arrays...")
    sys.stdout.flush()
    
    # Predicted incidents: both (src->dst) and (dst->src)
    pred_inc1 = np.column_stack([pred_array[:, 0], pred_array[:, 1], pred_array[:, 2]])  # t, src, dst
    pred_inc2 = np.column_stack([pred_array[:, 0], pred_array[:, 2], pred_array[:, 1]])  # t, dst, src
    pred_incidents = np.vstack([pred_inc1, pred_inc2])
    
    # Ground truth incidents
    gt_inc1 = np.column_stack([gt_array[:, 0], gt_array[:, 1], gt_array[:, 2]])
    gt_inc2 = np.column_stack([gt_array[:, 0], gt_array[:, 2], gt_array[:, 1]])
    gt_incidents = np.vstack([gt_inc1, gt_inc2])
    
    print(f"Created incident arrays: {len(pred_incidents):,} predicted, {len(gt_incidents):,} GT")
    sys.stdout.flush()
    
    # Get unique MMSIs
    pred_mmsis = set(pred_incidents[:, 1])
    gt_mmsis = set(gt_incidents[:, 1])
    all_mmsi = np.array(sorted(pred_mmsis.union(gt_mmsis)))
    
    print(f"Processing {len(all_mmsi):,} unique MMSIs...")
    sys.stdout.flush()
    
    # OPTIMIZATION: Skip per-MMSI if too many MMSIs
    if len(all_mmsi) > 100_000:
        print(f"WARNING: {len(all_mmsi):,} MMSIs detected. Skipping detailed per-MMSI analysis for performance.")
        print("Consider using --restrict_mmsi or processing smaller datasets.")
        per_m = pd.DataFrame(columns=["mmsi","tp","fp","fn","precision","recall","f1"])
    else:
        # Group incidents by MMSI using vectorized operations
        pred_by_mmsi = {}
        gt_by_mmsi = {}
        
        print("Grouping incidents by MMSI...")
        group_mmsi_start = time.time()
        sys.stdout.flush()
        
        for i, mmsi in enumerate(all_mmsi):
            if i % 10000 == 0:
                print(f"  Grouping MMSI {i:,}/{len(all_mmsi):,}")
                sys.stdout.flush()
            
            pred_mask = pred_incidents[:, 1] == mmsi
            gt_mask = gt_incidents[:, 1] == mmsi
            pred_by_mmsi[mmsi] = set(map(tuple, pred_incidents[pred_mask]))
            gt_by_mmsi[mmsi] = set(map(tuple, gt_incidents[gt_mask]))
        
        group_mmsi_time = time.time() - group_mmsi_start
        print(f"MMSI grouping completed in {group_mmsi_time:.1f} seconds")
        sys.stdout.flush()
        
        # Compute per-MMSI metrics in batches
        def compute_mmsi_metrics_batch(mmsis):
            results = []
            for mmsi in mmsis:
                Pm = pred_by_mmsi.get(mmsi, set())
                Gm = gt_by_mmsi.get(mmsi, set())
                tp_m = len(Pm & Gm)
                fp_m = len(Pm - Gm)
                fn_m = len(Gm - Pm)
                pr_m = 0.0 if (tp_m+fp_m)==0 else tp_m/(tp_m+fp_m)
                rc_m = 0.0 if (tp_m+fn_m)==0 else tp_m/(tp_m+fn_m)
                f1_m = f1(pr_m, rc_m)
                results.append((mmsi, tp_m, fp_m, fn_m, pr_m, rc_m, f1_m))
            return results
        
        # Process MMSIs in chunks
        chunk_size = 500
        rows_m = []
        for i in range(0, len(all_mmsi), chunk_size):
            chunk = all_mmsi[i:i+chunk_size]
            rows_m.extend(compute_mmsi_metrics_batch(chunk))
            if i % (chunk_size * 20) == 0:
                print(f"  Processed {min(i+chunk_size, len(all_mmsi)):,}/{len(all_mmsi):,} MMSIs")
                sys.stdout.flush()
        
        per_m = pd.DataFrame(rows_m, columns=["mmsi","tp","fp","fn","precision","recall","f1"])
    
    mmsi_time = time.time() - start_mmsi
    print(f"Per-MMSI metrics completed in {mmsi_time:.1f} seconds")
    sys.stdout.flush()

    eval_time = time.time() - eval_start
    print(f"=== EVALUATION BREAKDOWN ===")
    print(f"Array conversion: {convert_time:.1f}s")
    print(f"Set creation: {sets_time:.1f}s")
    print(f"Overall metrics: {metrics_time:.1f}s")
    print(f"Per-timestamp: {timestamp_time:.1f}s")
    print(f"Per-MMSI: {mmsi_time:.1f}s")
    print(f"Total evaluation time: {eval_time:.1f} seconds ({eval_time/60:.1f} minutes)")
    sys.stdout.flush()

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1_},
        "per_timestamp": per_t,
        "per_mmsi": per_m,
    }
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

    # per-MMSI metrics: for each MMSI, what's the precision/recall of its neighbor sets?
    all_mmsi = sorted(set(incP["mmsi"]).union(set(incG["mmsi"])))
    rows_m = []
    for mmsi in all_mmsi:
        Pm = set(incP[incP["mmsi"]==mmsi]["key"])
        Gm = set(incG[incG["mmsi"]==mmsi]["key"])
        tp_m = len(Pm & Gm)
        fp_m = len(Pm - Gm)
        fn_m = len(Gm - Pm)
        pr_m = 0.0 if (tp_m+fp_m)==0 else tp_m/(tp_m+fp_m)
        rc_m = 0.0 if (tp_m+fn_m)==0 else tp_m/(tp_m+fn_m)
        f1_m = f1(pr_m, rc_m)
        rows_m.append((mmsi, tp_m, fp_m, fn_m, pr_m, rc_m, f1_m))
    per_m = pd.DataFrame(rows_m, columns=["mmsi","tp","fp","fn","precision","recall","f1"]).sort_values("mmsi")

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1_},
        "per_timestamp": per_t,
        "per_mmsi": per_m,
    }

def main():
    # SLURM-friendly: Print start time and job info
    start_time = time.time()
    print(f"=== NEIGHBOR EVALUATION SCRIPT STARTED ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(sys.argv)}")
    sys.stdout.flush()
    
    ap = argparse.ArgumentParser(description="Evaluate predicted neighbors vs ground truth neighbors (OPTIMIZED)")
    ap.add_argument("--pred_csv", help="CSV with predicted positions (mmsi, t_unix, lat_deg, lon_deg)")
    ap.add_argument("--gt_csv", help="CSV with ground truth positions")
    ap.add_argument("--radius_km", type=float, help="Neighbor radius in km")
    ap.add_argument("--radius_nm", type=float, help="Neighbor radius in nautical miles (overrides radius_km)")
    ap.add_argument("--time_tolerance_sec", type=int, default=0, help="± seconds to align timestamps (0 = exact)")
    ap.add_argument("--restrict_mmsi", type=int, default=None, help="Legacy option: evaluate only neighbors involving this MMSI")
    ap.add_argument("--target_mmsi", type=int, default=None, help="Explicit MMSI to evaluate when not using a neighbor CSV")
    ap.add_argument("--neighbor_csv", type=Path, default=Path("neighbors_out/neighbor_list.csv"), help="CSV whose target_mmsi column lists vessels to evaluate (default: neighbors_out/neighbor_list.csv)")
    ap.add_argument("--max_targets", type=int, default=10, help="Limit the number of MMSIs read from --neighbor_csv (default: 10)")
    ap.add_argument("--out_dir", default="neighbors_eval_out", help="Directory to write detailed outputs")
    args = ap.parse_args()

    radius_km = nm_to_km(args.radius_nm) if args.radius_nm is not None else (args.radius_km or 1.852)
    print(f"Using neighbor radius: {radius_km:.3f} km")
    sys.stdout.flush()

    # Load data
    df_pred = load_positions(args.pred_csv)
    df_gt   = load_positions(args.gt_csv)

    # Build neighbor edges under identical conditions using optimized version
    pred_edges = positions_to_edges_optimized(df_pred, radius_km=radius_km,
                                             time_tolerance_sec=args.time_tolerance_sec, undirected=True)
    gt_edges   = positions_to_edges_optimized(df_gt, radius_km=radius_km,
                                             time_tolerance_sec=args.time_tolerance_sec, undirected=True)
    print(f"Predicted edges: {len(pred_edges):,}")
    print(f"Ground-truth edges: {len(gt_edges):,}")
    sys.stdout.flush()

    print("Evaluating neighbor sets...")
    sys.stdout.flush()
    restrict_val = args.target_mmsi if args.target_mmsi is not None else args.restrict_mmsi
    metrics = evaluate_neighbor_sets_optimized(pred_edges, gt_edges, restrict_mmsi=restrict_val)

    # Print overall
    o = metrics["overall"]
    print(f"Overall:  precision={o['precision']:.3f}  recall={o['recall']:.3f}  f1={o['f1']:.3f}  (tp={o['tp']}, fp={o['fp']}, fn={o['fn']})")
    sys.stdout.flush()

    # Determine which MMSIs to evaluate individually
    target_mmsis = []
    if args.neighbor_csv and args.neighbor_csv.exists():
        try:
            df_targets = pd.read_csv(args.neighbor_csv)
        except Exception as exc:
            raise SystemExit(f"Failed to read neighbor CSV {args.neighbor_csv}: {exc}")
        if "target_mmsi" not in df_targets.columns:
            raise SystemExit(f"Neighbor CSV {args.neighbor_csv} must contain a 'target_mmsi' column")
        seen = []
        for val in df_targets["target_mmsi"].dropna().astype(int).tolist():
            if val not in seen:
                seen.append(val)
            if len(seen) >= args.max_targets:
                break
        if not seen:
            raise SystemExit(f"Neighbor CSV {args.neighbor_csv} did not yield any target_mmsi values")
        target_mmsis = seen
        print(f"Loaded {len(target_mmsis)} target MMSIs from {args.neighbor_csv}")
    elif args.target_mmsi is not None:
        target_mmsis = [args.target_mmsi]
    elif args.restrict_mmsi is not None:
        target_mmsis = [args.restrict_mmsi]
    else:
        raise SystemExit("No target MMSIs provided. Supply --neighbor_csv or --target_mmsi/--restrict_mmsi.")

    per_target_rows = []
    if target_mmsis:
        print("\nPer-target evaluation summary:")
        for mmsi in target_mmsis:
            metrics_target = evaluate_neighbor_sets(pred_edges, gt_edges, restrict_mmsi=mmsi)
            overall_target = metrics_target["overall"]
            per_target_rows.append({
                "mmsi": mmsi,
                "precision": overall_target["precision"],
                "recall": overall_target["recall"],
                "f1": overall_target["f1"],
                "tp": overall_target["tp"],
                "fp": overall_target["fp"],
                "fn": overall_target["fn"]
            })
            print(f"  MMSI {mmsi}: precision={overall_target['precision']:.3f}, recall={overall_target['recall']:.3f}, f1={overall_target['f1']:.3f}")
        sys.stdout.flush()
    else:
        print("No target MMSIs provided or sampled for per-target evaluation.")

    # Save detailed outputs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics["per_timestamp"].to_csv(out_dir / "per_timestamp.csv", index=False)
    metrics["per_mmsi"].to_csv(out_dir / "per_mmsi.csv", index=False)
    pred_edges.to_csv(out_dir / "pred_edges.csv", index=False)
    gt_edges.to_csv(out_dir / "gt_edges.csv", index=False)
    if per_target_rows:
        per_target_df = pd.DataFrame(per_target_rows)
        per_target_df.to_csv(out_dir / "per_target_summary.csv", index=False)
        print(f"Per-target summary saved to {out_dir / 'per_target_summary.csv'}")
    print(f"Wrote details to: {out_dir}/")
    
    # SLURM-friendly: Print completion info
    total_time = time.time() - start_time
    print(f"=== NEIGHBOR EVALUATION COMPLETED ===")
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
