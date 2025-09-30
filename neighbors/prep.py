# neighbors/prep.py
# Inputs expected: DataFrame with columns ['mmsi','timestamp','lat','lon','sog','cog']

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import math
import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PrepConfig:
    freq_min: int = 30                  # 30-min grid
    max_gap_min: int = 60               # forward-fill up to 60 min
    K: int = 12                         # keep up to 12 candidates
    r_gate_nm: float = 12.0             # candidate search radius
    r_neighbor_nm: float = 4.0          # label: within 4 nm by/within horizon
    horizons_min: Tuple[int, ...] = (30, 60, 90, 120)  # recommend grid-aligned
    hist_steps: int = 4                 # history steps (e.g., 2 hours @30-min)

    # Tokenizer bins (coarse, four-hot style)
    bins_range: int = 60
    bins_bearing: int = 72
    bins_dSOG: int = 32
    bins_dCOG: int = 72

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def steps_for(H_min: int, freq_min: int) -> int:
    """How many steps forward represent horizon H on a grid of freq_min.
    Ceil rule so 20 min @ 30-min grid -> 1 step; 40 -> 2 steps, etc."""
    return int(math.ceil(H_min / float(freq_min)))

# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

R_EARTH_NM = 3440.065

def haversine_nm(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R_EARTH_NM * c

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlam = lam2 - lam1
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    brg = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brg

def enu_offsets_nm(lat_ref, lon_ref, lat, lon):
    dy = (lat - lat_ref) * 60.0
    mean_lat = np.radians((lat + lat_ref) / 2.0)
    dx = (lon - lon_ref) * 60.0 * np.cos(mean_lat)
    return dx, dy

def sogcog_to_vel_nm_per_min(sog_kn, cog_deg):
    v = sog_kn / 60.0
    vx = v * np.sin(np.radians(cog_deg))
    vy = v * np.cos(np.radians(cog_deg))
    return vx, vy

def wrap180(deg):
    return ((deg + 180.0) % 360.0) - 180.0

# ──────────────────────────────────────────────────────────────────────────────
# Resampling & indexing
# ──────────────────────────────────────────────────────────────────────────────

def resample_ais(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """Resample per MMSI to a fixed-minute grid; forward-fill within max_gap_min."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['mmsi', 'timestamp'])
    out = []
    for mmsi, g in df.groupby('mmsi', sort=False):
        g = g.set_index('timestamp').asfreq(f'{cfg.freq_min}T')
        g['mmsi'] = mmsi
        limit = max(1, cfg.max_gap_min // cfg.freq_min)
        g[['lat','lon','sog','cog']] = g[['lat','lon','sog','cog']].ffill(limit=limit)
        g = g.dropna(subset=['lat','lon','sog','cog'], how='any')
        out.append(g.reset_index())
    if not out:
        return pd.DataFrame(columns=['timestamp','mmsi','lat','lon','sog','cog'])
    res = pd.concat(out, ignore_index=True)
    return res[['timestamp','mmsi','lat','lon','sog','cog']]

def build_time_index(df_resampled: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    snaps: Dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, g in df_resampled.groupby('timestamp', sort=True):
        snaps[ts] = g[['mmsi','lat','lon','sog','cog']].reset_index(drop=True)
    return snaps

# ──────────────────────────────────────────────────────────────────────────────
# Candidate retrieval
# ──────────────────────────────────────────────────────────────────────────────

def candidates_radius(snapshot: pd.DataFrame, target_row: pd.Series, cfg: PrepConfig) -> pd.DataFrame:
    lat_t, lon_t = target_row['lat'], target_row['lon']
    d_nm = haversine_nm(lat_t, lon_t, snapshot['lat'].to_numpy(), snapshot['lon'].to_numpy())
    idx_other = snapshot['mmsi'].to_numpy() != target_row['mmsi']
    d_nm = d_nm[idx_other]
    sub = snapshot[idx_other].copy()
    sub = sub.assign(range_nm=d_nm)
    within = sub[sub['range_nm'] <= cfg.r_gate_nm]
    if len(within) == 0:
        chosen = sub.nsmallest(cfg.K, 'range_nm')
    else:
        chosen = within.nsmallest(cfg.K, 'range_nm')
    return chosen.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Relative history & labels
# ──────────────────────────────────────────────────────────────────────────────

def build_rel_history(df_resampled: pd.DataFrame,
                      ts_now: pd.Timestamp,
                      target_mmsi: int,
                      cand_mmsi: int,
                      cfg: PrepConfig):
    """Return (hist_df_target, rel_seq_dict) for T history steps aligned on timestamp."""
    times = [ts_now - pd.Timedelta(minutes=cfg.freq_min * i) for i in range(cfg.hist_steps)][::-1]
    tg = df_resampled[(df_resampled['mmsi']==target_mmsi) & (df_resampled['timestamp'].isin(times))].copy()
    cg = df_resampled[(df_resampled['mmsi']==cand_mmsi) & (df_resampled['timestamp'].isin(times))].copy()
    if len(tg) != len(times) or len(cg) != len(times):
        return pd.DataFrame(), {}
    tg = tg.sort_values('timestamp'); cg = cg.sort_values('timestamp')
    r = haversine_nm(tg['lat'].to_numpy(), tg['lon'].to_numpy(), cg['lat'].to_numpy(), cg['lon'].to_numpy())
    brg = initial_bearing_deg(tg['lat'].to_numpy(), tg['lon'].to_numpy(), cg['lat'].to_numpy(), cg['lon'].to_numpy())
    dSOG = (cg['sog'] - tg['sog']).to_numpy()
    dCOG = wrap180((cg['cog'] - tg['cog']).to_numpy())
    dr = np.gradient(r, edge_order=1) / cfg.freq_min  # nm/min approx
    rel_seq = {
        'range_nm': r.astype(np.float32),
        'bearing_deg': brg.astype(np.float32),
        'dSOG': dSOG.astype(np.float32),
        'dCOG': dCOG.astype(np.float32),
        'dr_nm_per_min': dr.astype(np.float32),
        'times': np.array([t.value for t in tg['timestamp']], dtype=np.int64)
    }
    return tg[['timestamp','lat','lon','sog','cog']].reset_index(drop=True), rel_seq

def label_neighbor_future(df_resampled: pd.DataFrame,
                          ts_now: pd.Timestamp,
                          target_mmsi: int,
                          cand_mmsi: int,
                          H_min: int,
                          r_neighbor_nm: float,
                          freq_min: int) -> int:
    """1 if min separation within [now, now+H] is < r_neighbor_nm, else 0 (ceil horizon)."""
    n_steps = steps_for(H_min, freq_min)
    times = [ts_now + pd.Timedelta(minutes=i*freq_min) for i in range(0, n_steps+1)]
    tg = df_resampled[(df_resampled['mmsi']==target_mmsi) & (df_resampled['timestamp'].isin(times))]
    cg = df_resampled[(df_resampled['mmsi']==cand_mmsi) & (df_resampled['timestamp'].isin(times))]
    if tg.empty or cg.empty:
        return 0
    mg = pd.merge(tg, cg, on='timestamp', suffixes=('_t','_c'))
    if mg.empty:
        return 0
    d = haversine_nm(mg['lat_t'].to_numpy(), mg['lon_t'].to_numpy(), mg['lat_c'].to_numpy(), mg['lon_c'].to_numpy())
    return int(np.nanmin(d) < r_neighbor_nm)

# ──────────────────────────────────────────────────────────────────────────────
# Tokenization helpers
# ──────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, cfg: PrepConfig):
        self.cfg = cfg
        self.range_edges = np.linspace(0.0, cfg.r_gate_nm, cfg.bins_range, dtype=np.float32)
        self.bear_edges  = np.linspace(0.0, 360.0,     cfg.bins_bearing, dtype=np.float32)
        self.dSOG_edges  = np.linspace(-20.0, 20.0,    cfg.bins_dSOG,    dtype=np.float32)
        self.dCOG_edges  = np.linspace(-180.0, 180.0,  cfg.bins_dCOG,    dtype=np.float32)

    def digitize(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        idx = np.argmin(np.abs(x[..., None] - edges[None, ...]), axis=-1)
        return idx.astype(np.int64)

    def token_indices(self, rel_seq: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            'range_idx':   self.digitize(rel_seq['range_nm'],              self.range_edges),
            'bearing_idx': self.digitize((rel_seq['bearing_deg'] % 360.0), self.bear_edges),
            'dSOG_idx':    self.digitize(rel_seq['dSOG'],                  self.dSOG_edges),
            'dCOG_idx':    self.digitize(rel_seq['dCOG'],                  self.dCOG_edges),
        }

# ──────────────────────────────────────────────────────────────────────────────
# Batch builder
# ──────────────────────────────────────────────────────────────────────────────

def build_one_batch(df_resampled: pd.DataFrame,
                    snaps: Dict[pd.Timestamp, pd.DataFrame],
                    cfg: PrepConfig,
                    ts_now: pd.Timestamp,
                    target_mmsi_list: Iterable[int],
                    H_min: int) -> Dict[str, torch.Tensor]:
    """Build a batch for a single timestamp and list of targets."""
    token = SimpleTokenizer(cfg)

    all_rel_seq: List[List[Dict[str, np.ndarray]]] = []
    all_geom:    List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []
    all_candids: List[np.ndarray] = []
    all_tgtids:  List[int] = []
    all_mask:    List[np.ndarray] = []

    snap = snaps.get(ts_now)
    if snap is None:
        raise ValueError(f"No snapshot found at {ts_now}.")

    for target_mmsi in target_mmsi_list:
        tgt_row = snap[snap['mmsi']==target_mmsi]
        if tgt_row.empty:
            # still push a fully padded row so B aligns with targets length
            all_rel_seq.append([{
                'range_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'bearing_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'dSOG_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'dCOG_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
            }]*cfg.K)
            all_geom.append(np.zeros((cfg.K, 2), dtype=np.float32))
            all_labels.append(np.zeros((cfg.K,), dtype=np.int64))
            all_candids.append(np.zeros((cfg.K,), dtype=np.int64))
            all_tgtids.append(int(target_mmsi))
            all_mask.append(np.zeros((cfg.K,), dtype=np.float32))
            continue

        tgt_row = tgt_row.iloc[0]
        cand_df = candidates_radius(snap, tgt_row, cfg)

        rel_seqs, geoms, labels, cand_ids, mask = [], [], [], [], []

        for _, row in cand_df.iterrows():
            hist_df, rel_seq = build_rel_history(df_resampled, ts_now, int(tgt_row['mmsi']), int(row['mmsi']), cfg)
            if rel_seq == {}:
                continue
            # now geometry
            r_now = float(rel_seq['range_nm'][-1])
            brg_now = float(rel_seq['bearing_deg'][-1] % 360.0)
            # label for H_min (ceil to grid)
            y = label_neighbor_future(df_resampled, ts_now, int(tgt_row['mmsi']), int(row['mmsi']),
                                      H_min, cfg.r_neighbor_nm, cfg.freq_min)
            # tokens
            tok = token.token_indices(rel_seq)

            rel_seqs.append(tok)
            geoms.append(np.array([r_now, brg_now/360.0], dtype=np.float32))
            labels.append(y)
            cand_ids.append(int(row['mmsi']))
            mask.append(1.0)

            if len(labels) == cfg.K:
                break

        # Pad to K
        pad_needed = max(0, cfg.K - len(labels))
        for _ in range(pad_needed):
            rel_seqs.append({
                'range_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'bearing_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'dSOG_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
                'dCOG_idx': np.zeros((cfg.hist_steps,), dtype=np.int64),
            })
            geoms.append(np.zeros((2,), dtype=np.float32))
            labels.append(0)
            cand_ids.append(0)
            mask.append(0.0)

        all_rel_seq.append(rel_seqs[:cfg.K])
        all_geom.append(np.stack(geoms[:cfg.K], axis=0))
        all_labels.append(np.array(labels[:cfg.K], dtype=np.int64))
        all_candids.append(np.array(cand_ids[:cfg.K], dtype=np.int64))
        all_tgtids.append(int(target_mmsi))
        all_mask.append(np.array(mask[:cfg.K], dtype=np.float32))

    if len(all_labels) == 0:
        raise ValueError("No valid samples built for this timestamp/targets.")

    # Stack to tensors
    range_idx = torch.from_numpy(np.stack([[rs['range_idx']   for rs in rel_list] for rel_list in all_rel_seq])).long()   # [B,K,T]
    bearing_idx = torch.from_numpy(np.stack([[rs['bearing_idx'] for rs in rel_list] for rel_list in all_rel_seq])).long() # [B,K,T]
    dSOG_idx = torch.from_numpy(np.stack([[rs['dSOG_idx']     for rs in rel_list] for rel_list in all_rel_seq])).long()   # [B,K,T]
    dCOG_idx = torch.from_numpy(np.stack([[rs['dCOG_idx']     for rs in rel_list] for rel_list in all_rel_seq])).long()   # [B,K,T]
    geom = torch.from_numpy(np.stack(all_geom)).float()                                                                     # [B,K,2]
    y = torch.from_numpy(np.stack(all_labels)).float()                                                                      # [B,K]
    cand_mmsi = torch.from_numpy(np.stack(all_candids)).long()                                                              # [B,K]
    target_mmsi = torch.from_numpy(np.array(all_tgtids)).long()                                                             # [B]
    cand_mask = torch.from_numpy(np.stack(all_mask)).float()                                                                # [B,K]

    return {
        'range_idx': range_idx,      # [B,K,T] (long)
        'bearing_idx': bearing_idx,  # [B,K,T] (long)
        'dSOG_idx': dSOG_idx,        # [B,K,T] (long)
        'dCOG_idx': dCOG_idx,        # [B,K,T] (long)
        'geom': geom,                # [B,K,2]  (float) [range_now_nm, bearing_norm_0to1]
        'y_near': y,                 # [B,K]    (float) {0,1}
        'cand_mmsi': cand_mmsi,      # [B,K]    (long)  MMSI per candidate (0 = pad)
        'target_mmsi': target_mmsi,  # [B]      (long)  MMSI per target row
        'cand_mask': cand_mask,      # [B,K]    (float) 1=real candidate, 0=pad
    }

def pick_targets_for_timestamp(snaps: Dict[pd.Timestamp, pd.DataFrame],
                               ts_now: pd.Timestamp,
                               max_targets: int = 8) -> List[int]:
    snap = snaps.get(ts_now)
    if snap is None or snap.empty:
        return []
    mmsis = snap['mmsi'].drop_duplicates().astype(int).tolist()
    return mmsis[:max_targets]


if __name__ == "__main__":
    import argparse, pickle, pandas as pd

    ap = argparse.ArgumentParser(description="Resample AIS and build timestamp snapshots")
    ap.add_argument("--csv", required=True, help="Input AIS CSV with columns: mmsi,timestamp,lat,lon,sog,cog")
    ap.add_argument("--out-resampled", default="data/resampled.parquet", help="Parquet output for resampled data")
    ap.add_argument("--out-snaps", default="data/snaps.pkl", help="Pickle dict {Timestamp: DataFrame}")
    ap.add_argument("--freq-min", type=int, default=30)
    ap.add_argument("--hist-steps", type=int, default=4)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--r-gate-nm", type=float, default=12.0)
    ap.add_argument("--r-neighbor-nm", type=float, default=4.0)
    args = ap.parse_args()

    cfg = PrepConfig(freq_min=args.freq_min,
                     hist_steps=args.hist_steps,
                     K=args.K,
                     r_gate_nm=args.r_gate_nm,
                     r_neighbor_nm=args.r_neighbor_nm)

    print(f"[prep] reading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[prep] resampling (freq={cfg.freq_min} min)")
    df_r = resample_ais(df, cfg)
    df_r.to_parquet(args.out_resampled)
    print(f"[prep] wrote {args.out_resampled} ({len(df_r)} rows)")

    snaps = build_time_index(df_r)
    with open(args.out_snaps, "wb") as f:
        pickle.dump(snaps, f)
    print(f"[prep] wrote {args.out_snaps} ({len(snaps)} timestamps)")
    print("[prep] done")