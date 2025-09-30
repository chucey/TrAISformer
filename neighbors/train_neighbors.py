# neighbors/train_neighbor.py
import torch
import torch.nn.functional as F

from neighbors.head import NeighborPredictor
from neighbors.prep import PrepConfig, steps_for, build_one_batch
from neighbors.infer_neighbor import build_target_x_like_repo, rollout_to_horizon  # reuse helpers

def init_neighbor_head(model, cfg: PrepConfig, fusion="dot", d_tok=64):
    """Create the neighbor head with bins from cfg and d_model from TrAISformer."""
    d_model = model.ln_f.normalized_shape[0]  # TrAISformer hidden size
    bins = dict(bins_range=cfg.bins_range, bins_bearing=cfg.bins_bearing,
                bins_dSOG=cfg.bins_dSOG, bins_dCOG=cfg.bins_dCOG)
    return NeighborPredictor(bins=bins, d_tok=d_tok, d_model=d_model, fusion=fusion)

def make_optimizers(model, nbr, lr_model=2e-5, lr_head=8e-4, weight_decay=1e-2):
    """Two-optimizer setup using AdamW"""
    opt_model = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr_model, weight_decay=weight_decay)
    opt_nbr   = torch.optim.AdamW(nbr.parameters(), lr=lr_head, weight_decay=weight_decay)
    return opt_model, opt_nbr

def train_step(model,
               nbr,
               df_resampled,
               snaps,
               ts_now,
               targets,
               H_min: int,
               cfg: PrepConfig,
               opt_model,
               opt_nbr,
               device: str = "cuda" if torch.cuda.is_available() else "cpu",
               latent_mode: str = "now"):
    model.train()
    nbr.train()

    # 1) Build candidate batch (relative-history token indices)
    batch = build_one_batch(df_resampled, snaps, cfg, ts_now, targets, H_min)
    for k in ["range_idx", "bearing_idx", "dSOG_idx", "dCOG_idx", "geom", "y_near", "cand_mask"]:
        batch[k] = batch[k].to(device)

    # 2) Build TrAISformer inputs (repo real-token format in [0,1))
    x_hist = build_target_x_like_repo(df_resampled, ts_now, targets, cfg, model).to(device)

    # 3) Get horizon latent h_H
    if latent_mode == "rollout":
        H_steps = steps_for(H_min, cfg.freq_min)  # ceil to grid
        h_H = rollout_to_horizon(model, x_hist, H_steps)           # [B, D]
    else:
        _, _, fea = model(x_hist, with_targets=False, return_hidden=True)
        h_H = fea[:, -1, :]                                        # [B, D]

    # 4) Neighbor head forward
    logits_near, _ = nbr(h_H,
                         batch["range_idx"], batch["bearing_idx"],
                         batch["dSOG_idx"],  batch["dCOG_idx"])     # [B, K]

    # 5) Loss with padding mask + robust pos_weight
    y = batch["y_near"]               # [B, K] float {0,1}
    w = batch["cand_mask"]            # [B, K] 1 for real candidate, 0 for pad

    # count positives/negatives only over valid candidates
    pos = (y * w).sum().clamp_min(1.0)
    neg = ((1.0 - y) * w).sum().clamp_min(1.0)
    pos_w = neg / pos                 # scalar pos_weight

    loss = F.binary_cross_entropy_with_logits(
        logits_near, y, weight=w, pos_weight=pos_w
    )

    # 6) Backprop
    opt_model.zero_grad(set_to_none=True)
    opt_nbr.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(nbr.parameters()), 1.0
    )
    opt_model.step()
    opt_nbr.step()

    return float(loss.item())


if __name__ == "__main__":
    import argparse, pickle, pandas as pd, torch
    from types import SimpleNamespace
    from neighbors.prep import PrepConfig, pick_targets_for_timestamp

    def load_or_make_model(ckpt_path, device):
        """Try to load a trained TrAISformer; otherwise make a small dummy config (for smoke runs)."""
        from models import TrAISformer
        if ckpt_path:
            obj = torch.load(ckpt_path, map_location=device)
            # Common patterns: {"config": ..., "state_dict": ...} or raw state_dict
            if isinstance(obj, dict) and "config" in obj:
                cfg = obj["config"]
                if not isinstance(cfg, SimpleNamespace):
                    cfg = SimpleNamespace(**cfg)
                model = TrAISformer(cfg).to(device)
                sd = obj.get("state_dict") or obj.get("model_state_dict") or obj
                model.load_state_dict(sd, strict=False)
            else:
                raise RuntimeError(
                    "Unsupported TrAISformer checkpoint format. "
                    "Expected a dict with 'config' and 'state_dict'."
                )
        else:
            # Dummy model (replace with real checkpoint for real training)
            cfg = SimpleNamespace(
                lat_size=256, lon_size=256, sog_size=64, cog_size=360,
                full_size=256+256+64+360,
                n_lat_embd=64, n_lon_embd=64, n_sog_embd=32, n_cog_embd=32,
                n_embd=64+64+32+32, n_layer=4, n_head=8, max_seqlen=128,
                embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                partition_mode="uniform", mode="pos", blur=False
            )
            model = TrAISformer(cfg).to(device)
        return model

    ap = argparse.ArgumentParser(description="Train neighbor head on top of TrAISformer")
    ap.add_argument("--resampled", default="data/resampled.parquet")
    ap.add_argument("--snaps", default="data/snaps.pkl")
    ap.add_argument("--traisformer-ckpt", default=None, help="Path to trained TrAISformer checkpoint")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--H-min", type=int, default=60)
    ap.add_argument("--freq-min", type=int, default=30)
    ap.add_argument("--hist-steps", type=int, default=4)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fusion", default="dot", choices=["dot","bilinear","cosine"])
    ap.add_argument("--latent-mode", default="now", choices=["now","rollout"])
    ap.add_argument("--lr-model", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--freeze-traisformer", action="store_true")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--save", default="checkpoints/neighbor_head.pt")
    args = ap.parse_args()

    device = args.device
    df_r = pd.read_parquet(args.resampled)
    with open(args.snaps, "rb") as f:
        snaps = pickle.load(f)
    ts_list = sorted(snaps.keys())

    # TrAISformer backbone
    model = load_or_make_model(args.traisformer_ckpt, device)
    if args.freeze_traisformer:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    # Neighbor head + optimizers
    cfg = PrepConfig(freq_min=args.freq_min, hist_steps=args.hist_steps, K=args.K)
    nbr = init_neighbor_head(model, cfg, fusion=args.fusion).to(device)
    opt_model, opt_nbr = make_optimizers(
        model, nbr,
        lr_model=(0.0 if args.freeze_traisformer else args.lr_model),
        lr_head=args.lr_head, weight_decay=args.weight_decay,
    )

    steps = 0
    for epoch in range(args.epochs):
        for ts_now in ts_list[10:-10]:
            targets = pick_targets_for_timestamp(snaps, ts_now, max_targets=8)
            if not targets:
                continue
            loss = train_step(
                model, nbr, df_r, snaps, ts_now, targets,
                H_min=args.H_min, cfg=cfg,
                opt_model=opt_model, opt_nbr=opt_nbr,
                device=device, latent_mode=args.latent_mode
            )
            steps += 1
            if steps % args.print_every == 0:
                print(f"[train] epoch {epoch} step {steps} loss {loss:.4f}")
            if steps >= args.max_steps:
                break
        if steps >= args.max_steps:
            break

    import os
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(nbr.state_dict(), args.save)
    print(f"[train] saved neighbor head → {args.save}")
