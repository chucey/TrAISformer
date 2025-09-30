# neighbors/infer_neighbor.py
import numpy as np
import pandas as pd
import torch

from neighbors.head import NeighborPredictor
from neighbors.prep import PrepConfig, build_one_batch

# -------- Helpers -------------------------------------------------------------

@torch.no_grad()
def rollout_to_horizon(model, x_hist, H_steps):
    """
    Autoregress H_steps into the future using the model's discrete heads.
    x_hist: [B, T_in, 4] real-valued tokens in [0,1) (lat, lon, sog, cog)
    Returns h_H: [B, D] hidden at (T_in + H_steps).
    """
    x = x_hist.clone()
    for _ in range(H_steps):
        logits, _, fea = model(x, with_targets=False, return_hidden=True)   # fea: [B, T_cur, D]
        last = logits[:, -1, :]                                             # [B, full_size]
        # split last-step logits using model's vocab sizes
        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
            last,
            (model.lat_size, model.lon_size, model.sog_size, model.cog_size),
            dim=-1
        )
        lat_idx = lat_logits.argmax(-1).float()
        lon_idx = lon_logits.argmax(-1).float()
        sog_idx = sog_logits.argmax(-1).float()
        cog_idx = cog_logits.argmax(-1).float()

        # inverse of to_indexes() (uniform): real = idx / att_sizes
        att = model.att_sizes.to(x.device).float()  # [4]
        next_real = torch.stack([
            lat_idx / att[0],
            lon_idx / att[1],
            sog_idx / att[2],
            cog_idx / att[3],
        ], dim=-1)                                   # [B, 4]
        next_real = torch.clamp(next_real, 0.0, 1.0 - 1e-6)

        x = torch.cat([x, next_real.unsqueeze(1)], dim=1)  # append new step

    _, _, fea = model(x, with_targets=False, return_hidden=True)
    h_H = fea[:, -1, :]
    return h_H


def build_target_x_like_repo(df_resampled, ts_now, targets, cfg: PrepConfig, model, T_in=None):
    """
    Build TrAISformer input x in the repo's expected real format: [0,1) per channel.
    Returns x_hist: [B, T_in, 4] with (lat, lon, sog, cog).
    """
    if T_in is None:
        T_in = cfg.hist_steps

    # times: last T_in steps ending at ts_now
    times = [ts_now - pd.Timedelta(minutes=cfg.freq_min * i) for i in range(T_in-1, -1, -1)]

    # Use ROI & ranges from the trained model if available
    has_roi = hasattr(model, "lat_min") and hasattr(model, "lat_max") \
              and hasattr(model, "lon_min") and hasattr(model, "lon_max")
    lat_min = getattr(model, "lat_min", None)
    lat_max = getattr(model, "lat_max", None)
    lon_min = getattr(model, "lon_min", None)
    lon_max = getattr(model, "lon_max", None)
    sog_range = getattr(model, "sog_range", 30.0)

    xs = []
    for mmsi in targets:
        g = df_resampled[(df_resampled["mmsi"] == mmsi) &
                         (df_resampled["timestamp"].isin(times))].sort_values("timestamp")
        if len(g) != len(times):
            raise ValueError(f"Missing history for MMSI {mmsi} at {ts_now} (need {len(times)} steps).")

        lat = g["lat"].to_numpy()
        lon = g["lon"].to_numpy()
        sog = g["sog"].to_numpy()
        cog = g["cog"].to_numpy()

        # Normalize to [0,1) as the repo expects before to_indexes()
        if has_roi:
            x_lat = (lat - lat_min) / (lat_max - lat_min + 1e-8)
            x_lon = (lon - lon_min) / (lon_max - lon_min + 1e-8)
        else:
            # Fallback: local min/max over this window (not ideal; prefer ROI from the model)
            x_lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
            x_lon = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)

        x_sog = sog / sog_range
        x_cog = (cog % 360.0) / 360.0

        x = np.stack([x_lat, x_lon, x_sog, x_cog], axis=-1)
        x = np.clip(x, 0.0, 1.0 - 1e-6)
        xs.append(x)

    x_hist = torch.from_numpy(np.stack(xs, axis=0)).float()  # [B, T_in, 4]
    return x_hist


# -------- Main API ------------------------------------------------------------

@torch.no_grad()
def predict_neighbors(model,
                      nbr: NeighborPredictor,
                      df_resampled,
                      snaps,
                      ts_now,
                      targets,
                      H_min: int = 60,
                      cfg: PrepConfig = PrepConfig(),
                      top_k: int = 5,
                      latent_mode: str = "now",   # "now" or "rollout"
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Returns a list of {target_index, horizon_min, neighbors:[...]}, one per target.
    """
    model.eval()
    nbr.eval()

    step_idx = H_min // cfg.freq_min  # used only for "rollout"

    # 1) Build candidate batch (relative-history indices) on device
    batch = build_one_batch(df_resampled, snaps, cfg, ts_now, targets, H_min)
    for k in ["range_idx", "bearing_idx", "dSOG_idx", "dCOG_idx", "geom"]:
        batch[k] = batch[k].to(device)

    # 2) Build TrAISformer inputs x in the repo's real-token format
    x_hist = build_target_x_like_repo(df_resampled, ts_now, targets, cfg, model).to(device)

    # 3) Get h_H from TrAISformer
    if latent_mode == "rollout":
        H_steps = step_idx
        h_H = rollout_to_horizon(model, x_hist, H_steps)          # [B, D]
    else:
        # "now" latent: single forward, take last input step
        _, _, fea = model(x_hist, with_targets=False, return_hidden=True)
        h_H = fea[:, -1, :]                                       # [B, D]

    # 4) Neighbor probabilities
    logits_near, p_near = nbr(h_H,
                              batch["range_idx"], batch["bearing_idx"],
                              batch["dSOG_idx"],  batch["dCOG_idx"])          # [B,K]

    # 5) Package top-K
    B, K = p_near.shape
    rng = batch["geom"][..., 0].detach().cpu().numpy()                    # range_now (nm if your prep is in nm)
    brg = (batch["geom"][..., 1].detach().cpu().numpy() * 360.0)          # denorm to degrees
    p_np = p_near.detach().cpu().numpy()

    # Optional: include MMSIs if your prep returns them
    cand_ids = batch.get("cand_mmsi", None)
    tgt_ids  = batch.get("target_mmsi", None)

    out = []
    for b in range(B):
        order = np.argsort(-p_np[b])[:top_k]
        items = []
        for j in order:
            rec = {
                "candidate_index": int(j),
                "range_nm_now": float(rng[b, j]),
                "bearing_deg_now": float(brg[b, j] % 360.0),
                "p_near": float(p_np[b, j]),
            }
            if cand_ids is not None:
                rec["mmsi"] = int(cand_ids[b, j])
            items.append(rec)

        header = {"target_index": int(b), "horizon_min": int(H_min), "neighbors": items}
        if tgt_ids is not None:
            header["target_mmsi"] = int(tgt_ids[b])
        out.append(header)

    return out


if __name__ == "__main__":
    import argparse, pickle, pandas as pd, json, torch
    from types import SimpleNamespace
    from neighbors.prep import PrepConfig, pick_targets_for_timestamp
    from neighbors.head import NeighborPredictor

    def load_or_make_model(ckpt_path, device):
        """Load TrAISformer w/ config from checkpoint, or create a tiny dummy model for smoke runs."""
        from models import TrAISformer
        if ckpt_path:
            obj = torch.load(ckpt_path, map_location=device)
            if isinstance(obj, dict) and "config" in obj:
                cfg = obj["config"]
                if not isinstance(cfg, SimpleNamespace):
                    cfg = SimpleNamespace(**cfg)
                model = TrAISformer(cfg).to(device).eval()
                sd = obj.get("state_dict") or obj.get("model_state_dict") or obj
                model.load_state_dict(sd, strict=False)
            else:
                raise RuntimeError(
                    "Unsupported TrAISformer checkpoint format. "
                    "Expected a dict with 'config' and 'state_dict'."
                )
        else:
            cfg = SimpleNamespace(
                lat_size=256, lon_size=256, sog_size=64, cog_size=360,
                full_size=256+256+64+360,
                n_lat_embd=64, n_lon_embd=64, n_sog_embd=32, n_cog_embd=32,
                n_embd=64+64+32+32, n_layer=4, n_head=8, max_seqlen=128,
                embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                partition_mode="uniform", mode="pos", blur=False
            )
            model = TrAISformer(cfg).to(device).eval()
        return model

    ap = argparse.ArgumentParser(description="Infer Top-K neighbor probabilities for selected targets")
    ap.add_argument("--resampled", default="data/resampled.parquet")
    ap.add_argument("--snaps", default="data/snaps.pkl")
    ap.add_argument("--traisformer-ckpt", default=None, help="Path to trained TrAISformer checkpoint")
    ap.add_argument("--neighbor-ckpt", required=True, help="Path to trained neighbor head .pt")
    ap.add_argument("--ts", default=None, help="ISO8601 UTC timestamp; if omitted, middle of data")
    ap.add_argument("--targets", nargs="*", type=int, default=None, help="Explicit MMSIs; if omitted, auto-pick")
    ap.add_argument("--max-targets", type=int, default=4)
    ap.add_argument("--H-min", type=int, default=60)
    ap.add_argument("--freq-min", type=int, default=30)
    ap.add_argument("--hist-steps", type=int, default=4)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--latent-mode", default="now", choices=["now","rollout"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    args = ap.parse_args()

    device = args.device
    df_r = pd.read_parquet(args.resampled)
    with open(args.snaps, "rb") as f:
        snaps = pickle.load(f)
    ts_list = sorted(snaps.keys())

    ts_now = (pd.Timestamp(args.ts, tz="UTC") if args.ts else ts_list[len(ts_list)//2])
    targets = (args.targets if args.targets
               else pick_targets_for_timestamp(snaps, ts_now, max_targets=args.max_targets))

    # Models
    model = load_or_make_model(args.traisformer_ckpt, device)
    cfg = PrepConfig(freq_min=args.freq_min, hist_steps=args.hist_steps, K=args.K)
    d_model = model.ln_f.normalized_shape[0]
    bins = dict(bins_range=cfg.bins_range, bins_bearing=cfg.bins_bearing,
                bins_dSOG=cfg.bins_dSOG, bins_dCOG=cfg.bins_dCOG)
    nbr = NeighborPredictor(bins=bins, d_tok=64, d_model=d_model, fusion="dot").to(device).eval()
    nbr.load_state_dict(torch.load(args.neighbor_ckpt, map_location=device))

    # Predict
    out = predict_neighbors(model, nbr, df_r, snaps, ts_now, targets,
                            H_min=args.H_min, cfg=cfg, top_k=args.top_k,
                            latent_mode=args.latent_mode, device=device)

    txt = json.dumps(out, indent=2)
    if args.out == "-" or args.out == "/dev/stdout":
        print(txt)
    else:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(txt)
        print(f"[infer] wrote {args.out}")
