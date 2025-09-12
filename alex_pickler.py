import os, pickle, numpy as np, pandas as pd
def load_points(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype={"MMSI":"Int64"})
        # map common CSV headers -> expected names; edit if yours differ
        rename = {"LAT":"lat","LON":"lon","SOG":"sog","COG":"cog",
                  "BaseDateTime":"time","MMSI":"mmsi"}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        if "unix_ts" not in df.columns:
            ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df["unix_ts"] = (ts.view("int64") // 10**9)
        keep = ["lat","lon","sog","cog","unix_ts","mmsi"]
        df = df[keep].dropna()
        arr = df.to_numpy(dtype=np.float32)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError("Input PKL must be an (N,6) array: [lat,lon,sog,cog,unix_ts,mmsi]")
    # sort by MMSI then time
    order = np.lexsort((arr[:,4], arr[:,5]))
    return arr[order]

# segment points into tracks by MMSI and time gaps
def to_tracks(arr, max_gap_sec=30*60, min_len=20):
    mmsi = arr[:,5].astype(np.int64)
    t    = arr[:,4].astype(np.int64)
    tracks = []
    for m in np.unique(mmsi):
        g = arr[mmsi == m]
        tt = g[:,4].astype(np.int64)
        cut = np.where(np.diff(tt) > max_gap_sec)[0]
        starts = np.r_[0, cut+1]
        ends   = np.r_[cut+1, len(g)]
        for s,e in zip(starts, ends):
            seg = g[s:e]
            if len(seg) < min_len: 
                continue
            traj = seg[:, :4].astype(np.float32)   # [lat, lon, sog, cog]
            tracks.append({"mmsi": int(m), "traj": traj})
    return tracks

# splitting tracks into train/valid/test by MMSI
def split_tracks(tracks, ratios=(0.8,0.1,0.1), seed=42):
    rng = np.random.default_rng(seed)
    vessel_ids = np.array(sorted({t["mmsi"] for t in tracks}), dtype=np.int64)
    rng.shuffle(vessel_ids)
    n = len(vessel_ids)
    n_tr = int(n*ratios[0]); n_va = int(n*(ratios[0]+ratios[1]))
    sets = {
        "train": set(vessel_ids[:n_tr]),
        "valid": set(vessel_ids[n_tr:n_va]),
        "test":  set(vessel_ids[n_va:])
    }
    out = {k: [t for t in tracks if t["mmsi"] in s] for k,s in sets.items()}
    return out

# ---------- save ----------
def save_pkl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---- run it ----
in_path = "/Users/alexmiekisz/Downloads/AIS_2024_01_01 2.csv"   
dataset_name = "ct_dma"
out_dir = f"./data/{dataset_name}/"

points = load_points(in_path)
tracks = to_tracks(points, max_gap_sec=30*60, min_len=20)
splits = split_tracks(tracks, ratios=(0.8,0.1,0.1), seed=42)

save_pkl(os.path.join(out_dir, f"{dataset_name}_train.pkl"), splits["train"])
save_pkl(os.path.join(out_dir, f"{dataset_name}_valid.pkl"), splits["valid"])
save_pkl(os.path.join(out_dir, f"{dataset_name}_test.pkl"),  splits["test"])

print({k: len(v) for k,v in splits.items()}, "tracks")

