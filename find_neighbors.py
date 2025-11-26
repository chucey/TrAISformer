# file: find_neighbors.py
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

# -------------------- helpers --------------------

def to_radians(df, lat_col="lat_deg", lon_col="lon_deg"):
    return np.deg2rad(df[[lat_col, lon_col]].to_numpy())

def build_balltree(df_other, lat_col="lat_deg", lon_col="lon_deg"):
    pts_rad = to_radians(df_other, lat_col, lon_col)
    return BallTree(pts_rad, metric="haversine")

def km_to_rad(km: float) -> float:
    return km / EARTH_RADIUS_KM

def nm_to_km(nm: float) -> float:
    return nm * 1.852

def load_predictions(path_csv: str) -> pd.DataFrame:
    # expects columns: mmsi,t_unix,lat_deg,lon_deg (at minimum)
    df = pd.read_csv(path_csv, dtype={"mmsi": "int64", "t_unix": "int64"})
    need = {"mmsi", "t_unix", "lat_deg", "lon_deg"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"CSV {path_csv} is missing columns: {sorted(missing)}")
    return df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)

def match_other_positions_exact(df_target_times: pd.DataFrame, df_other_all: pd.DataFrame) -> pd.DataFrame:
    """Filter others to timestamps present in target (exact match)."""
    tset = set(df_target_times["t_unix"].unique().tolist())
    return df_other_all[df_other_all["t_unix"].isin(tset)].copy()

def match_other_positions_with_tolerance(df_target_times: pd.DataFrame,
                                         df_other_all: pd.DataFrame,
                                         tol_sec: int) -> dict[int, pd.DataFrame]:
    """
    For each target timestamp t, choose ONE row per other MMSI:
      the row from df_other_all with |t_other - t| minimal AND within tol_sec.
    Returns a dict: t_unix -> df_others_aligned_at_t
    """
    out = {}
    times = df_target_times["t_unix"].unique()
    # pre-sort once for fast slicing
    df_other_all = df_other_all.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)
    for t in times:
        if tol_sec <= 0:
            df_slice = df_other_all[df_other_all["t_unix"] == t]
        else:
            lo, hi = t - tol_sec, t + tol_sec
            window = df_other_all[(df_other_all["t_unix"] >= lo) & (df_other_all["t_unix"] <= hi)]
            if window.empty:
                df_slice = window
            else:
                # pick, per MMSI, the row whose time is closest to t
                window = window.assign(_dt=(window["t_unix"] - t).abs())
                df_slice = (window.sort_values(["mmsi", "_dt"])
                                  .drop_duplicates(subset=["mmsi"], keep="first")
                                  .drop(columns=["_dt"]))
        out[int(t)] = df_slice
    return out

# -------------------- core --------------------

def find_neighbors_for_target(
    df_target: pd.DataFrame,    # predictions for ONE target vessel
    df_others: pd.DataFrame,    # predictions for OTHER vessels
    radius_km: float = 1.0,     # default ~0.54 nm
    time_tolerance_sec: int = 0 # 0 = exact timestamp match
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      target_mmsi,target_shiptype, t_unix, lat_deg, lon_deg, neighbor_mmsi, neighbor_shiptype, neighbor_dist_km, n_neighbors
    One row per neighbor edge (and rows with neighbor_mmsi=None when none found).
    """
    out_rows = []
    r_rad = km_to_rad(radius_km)

    # Build per-t timestamp -> df_others_aligned_at_t (handles tolerance)
    if time_tolerance_sec <= 0:
        # exact matching; compute per timestamp on the fly
        df_oth_by_t = None
    else:
        df_oth_by_t = match_other_positions_with_tolerance(df_target_times=df_target,
                                                           df_other_all=df_others,
                                                           tol_sec=time_tolerance_sec)

    for t, df_tgt_t in df_target.groupby("t_unix"):
        if df_oth_by_t is None:
            df_oth_t = df_others[df_others["t_unix"] == t]
        else:
            df_oth_t = df_oth_by_t.get(int(t), pd.DataFrame(columns=df_others.columns))

        if df_oth_t.empty:
            for _, r in df_tgt_t.iterrows():
                out_rows.append({
                    "target_mmsi": int(r.mmsi), 
                    # "target_shiptype": int(r.SHIPTYPE), "SHIPTYPE" may not exist in some datasets
                    "t_unix": int(t),
                    "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                    "neighbor_mmsi": None, 
                    # "neighbor_shiptype": None, "SHIPTYPE" may not exist in some datasets
                    "neighbor_dist_km": None,
                    "n_neighbors": 0
                })
            continue

        tree = build_balltree(df_oth_t)
        pts_target_rad = to_radians(df_tgt_t)
        inds_list, dists_list = tree.query_radius(pts_target_rad, r=r_rad,
                                                  return_distance=True, sort_results=True)

        for row_i, (inds, dists) in enumerate(zip(inds_list, dists_list)):
            r = df_tgt_t.iloc[row_i]
            if len(inds) == 0: 
                out_rows.append({
                    "target_mmsi": int(r.mmsi),
                    # "target_shiptype": int(r.SHIPTYPE), 
                    "t_unix": int(t),
                    "lat_deg": float(r.lat_deg), 
                    "lon_deg": float(r.lon_deg),
                    "neighbor_mmsi": None, 
                    # "neighbor_shiptype": None, 
                    "neighbor_dist_km": None,
                    "n_neighbors": 0
                })
            else:
                # add all neighbors, skipping self if present
                count = 0
                for idx, dist_rad in zip(inds, dists):
                    neighbor = df_oth_t.iloc[idx]
                    if neighbor.mmsi == r.mmsi:
                        continue
                    out_rows.append({
                        "target_mmsi": int(r.mmsi),
                        # "target_shiptype": int(r.SHIPTYPE), 
                        "t_unix": int(t),
                        "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                        "neighbor_mmsi": int(neighbor.mmsi),
                        # "neighbor_shiptype": neighbor.SHIPTYPE,
                        "neighbor_dist_km": float(dist_rad * EARTH_RADIUS_KM),
                        "n_neighbors": None  # will fill later
                    })
                    count += 1
                if count == 0:  # only self was in range
                    out_rows.append({
                        "target_mmsi": int(r.mmsi),
                        # "target_shiptype": int(r.SHIPTYPE), 
                        "t_unix": int(t),
                        "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                        "neighbor_mmsi": None, 
                        # "neighbor_shiptype": None,
                        "neighbor_dist_km": None,
                        "n_neighbors": 0
                    })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # Robust neighbor counts (handles case where no neighbors at all)
    if "neighbor_mmsi" in out.columns:
        counts = (out.dropna(subset=["neighbor_mmsi"])
                    .groupby(["target_mmsi", "t_unix"])
                    .size()
                    .rename("n_neighbors_calc")
                    .reset_index())
        if counts.empty:
            out["n_neighbors"] = 0
        else:
            out = out.merge(counts, on=["target_mmsi", "t_unix"], how="left")
            out["n_neighbors"] = out["n_neighbors_calc"].fillna(0).astype(int)
            out = out.drop(columns=["n_neighbors_calc"])
    else:
        out["n_neighbors"] = 0

    return out.sort_values(["t_unix", "target_mmsi", "neighbor_dist_km"], na_position="last").reset_index(drop=True)

# -------------------- CLI --------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Neighbor vessels on TrAISformer predictions")
    ap.add_argument("--target_mmsi", type=int, required=True,
                    help="MMSI of the vessel of interest")
    ap.add_argument("--preds_csv_target", type=Path, required=True,
                    help="CSV with TrAISformer predictions for ALL vessels (includes the target)")
    ap.add_argument("--preds_csv_others", type=Path, default=None,
                    help="Optional CSV for others; default uses the same file as target.")
    # radius options
    ap.add_argument("--radius_km", type=float, default=None,
                    help="Proximity radius in kilometers (e.g., 1.852 ≈ 1 NM)")
    ap.add_argument("--radius_nm", type=float, default=None,
                    help="Proximity radius in nautical miles; overrides --radius_km if provided")
    # time tolerance
    ap.add_argument("--time_tolerance_sec", type=int, default=0,
                    help="Match others within ±T seconds and use their closest-time position (0 = exact match)")
    ap.add_argument("--out_csv", type=Path, default=Path("neighbors_out/neighbor_list.csv"))
    args = ap.parse_args()

    # pick radius
    if args.radius_nm is not None:
        radius_km = nm_to_km(args.radius_nm)
    elif args.radius_km is not None:
        radius_km = args.radius_km
    else:
        radius_km = 1.0  # default

    df_all = load_predictions(str(args.preds_csv_target))

    # Target's PREDICTED trajectory
    df_target = df_all[df_all["mmsi"] == args.target_mmsi].copy()
    if df_target.empty:
        raise SystemExit(f"No predictions found for MMSI {args.target_mmsi} in {args.preds_csv_target}")

    # Neighbors’ PREDICTED trajectories (everyone else from the SAME predictions file)
    df_others = df_all[df_all["mmsi"] != args.target_mmsi].copy()


    # Align timestamps (exact or tolerance inside the core function)
    out = find_neighbors_for_target(
        df_target=df_target,
        df_others=df_others,
        radius_km=radius_km,
        time_tolerance_sec=args.time_tolerance_sec
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote neighbors to {args.out_csv} ({len(out)} rows)")
