import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

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
    df = pd.read_csv(path_csv, dtype={"mmsi": "int64", "t_unix": "int64"})
    need = {"mmsi", "t_unix", "lat_deg", "lon_deg"}
    missing = need.difference(df.columns)
    if missing:
        print(f"CSV {path_csv} is missing columns: {sorted(missing)}")
    return df.sort_values(["t_unix", "mmsi"]).reset_index(drop=True)

def match_other_positions_with_tolerance(df_target_times: pd.DataFrame,
                                         df_other_all: pd.DataFrame,
                                         tol_sec: int) -> dict[int, pd.DataFrame]:
    out = {}
    times = df_target_times["t_unix"].unique()
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
                window = window.assign(_dt=(window["t_unix"] - t).abs())
                df_slice = (window.sort_values(["mmsi", "_dt"])
                                  .drop_duplicates(subset=["mmsi"], keep="first")
                                  .drop(columns=["_dt"]))
        out[int(t)] = df_slice
    return out

def find_neighbors_for_target(df_target: pd.DataFrame,df_others: pd.DataFrame,radius_km: float,time_tolerance_sec: int = 0) -> pd.DataFrame:
    out_rows = []
    r_rad = km_to_rad(radius_km)

    if time_tolerance_sec <= 0:
        df_oth_by_t = None
    else:
        df_oth_by_t = match_other_positions_with_tolerance(df_target_times=df_target,df_other_all=df_others,tol_sec=time_tolerance_sec)
    
    for t, df_tgt_t in df_target.groupby("t_unix"):
        if df_oth_by_t is None:
            df_oth_t = df_others[df_others["t_unix"] == t]
        else:
            df_oth_t = df_oth_by_t.get(int(t), pd.DataFrame(columns=df_others.columns))

        if df_oth_t.empty:
            for _, r in df_tgt_t.iterrows():
                out_rows.append({
                    "target_mmsi": int(r.mmsi),
                    "t_unix": int(t),
                    "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                    "neighbor_mmsi": None,
                    "neighbor_dist_km": None,
                    "n_neighbors": 0
                })
            continue

        tree = build_balltree(df_oth_t)
        pts_target_rad = to_radians(df_tgt_t)
        inds_list, dists_list = tree.query_radius(pts_target_rad, r=r_rad,return_distance=True, sort_results=True)

        for row_i, (inds, dists) in enumerate(zip(inds_list, dists_list)):
            r = df_tgt_t.iloc[row_i]
            if len(inds) == 0: 
                out_rows.append({
                    "target_mmsi": int(r.mmsi),
                    "t_unix": int(t),
                    "lat_deg": float(r.lat_deg), 
                    "lon_deg": float(r.lon_deg),
                    "neighbor_mmsi": None,
                    "neighbor_dist_km": None,
                    "n_neighbors": 0
                })
            else:
                count = 0
                for idx, dist_rad in zip(inds, dists):
                    neighbor = df_oth_t.iloc[idx]
                    if neighbor.mmsi == r.mmsi:
                        continue
                    out_rows.append({
                        "target_mmsi": int(r.mmsi),
                        "t_unix": int(t),
                        "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                        "neighbor_mmsi": int(neighbor.mmsi),
                        "neighbor_dist_km": float(dist_rad * EARTH_RADIUS_KM),
                        "n_neighbors": None
                    })
                    count += 1
                if count == 0:
                    out_rows.append({
                        "target_mmsi": int(r.mmsi),
                        "t_unix": int(t),
                        "lat_deg": float(r.lat_deg), "lon_deg": float(r.lon_deg),
                        "neighbor_mmsi": None,
                        "neighbor_dist_km": None,
                        "n_neighbors": 0
                    })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Neighbor vessels on TrAISformer predictions")
    ap.add_argument("--target_mmsi", type=int, default=None)
    ap.add_argument("--preds_csv_target", type=Path, required=True)
    ap.add_argument("--preds_csv_others", type=Path, default=None)
    ap.add_argument("--num_random_targets", type=int, default=10)
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--radius_nm", type=float, default=5)
    ap.add_argument("--time_tolerance_sec", type=int, default=300)
    ap.add_argument("--out_csv", type=Path, default=Path("neighbors_out/neighbor_list.csv"))
    args = ap.parse_args()

    radius_km = nm_to_km(args.radius_nm)
    df_all = load_predictions(str(args.preds_csv_target))
    df_others_all = load_predictions(str(args.preds_csv_others)) if args.preds_csv_others else df_all.copy()

    unique_mmsi = sorted(df_all["mmsi"].unique().tolist())
    if not unique_mmsi:
        print("No vessels found in predictions CSV")

    if args.target_mmsi is not None:
        target_mmsis = [args.target_mmsi]
    else:
        num_to_pick = min(args.num_random_targets, len(unique_mmsi))
        rng = np.random.default_rng(args.random_seed)
        target_mmsis = rng.choice(unique_mmsi, size=num_to_pick, replace=False).tolist()

    all_outputs = []
    for target_mmsi in target_mmsis:
        df_target = df_all[df_all["mmsi"] == target_mmsi].copy()
        if df_target.empty:
            continue
        df_others = df_others_all[df_others_all["mmsi"] != target_mmsi].copy()
        out = find_neighbors_for_target(
            df_target=df_target,
            df_others=df_others,
            radius_km=radius_km,
            time_tolerance_sec=args.time_tolerance_sec
        )
        if not out.empty:
            all_outputs.append(out)

    if not all_outputs:
        print("No neighbor rows generated for any target")

    merged = pd.concat(all_outputs, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)