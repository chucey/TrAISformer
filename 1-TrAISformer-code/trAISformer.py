#!/usr/bin/env python
# coding: utf-8
# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config
from export_hooks import PredictionWriter


cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle,
                                   )
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([(model.lat_max - model.lat_min), (model.lon_max - model.lon_min), 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4
    STEP_SECONDS = getattr(cf, "step_seconds", getattr(cf, "dt_seconds", 600))
    pred_writer = PredictionWriter(
    out_dir=os.path.join(cf.savedir, "predictions_out")
)

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            # seqs: (B, S, 4) in normalized space (first 2 dims are lat/lon)
            # masks: (B, max_seqlen)
            # mmsis: (B,)
            # time_starts: (B,) unix seconds for seq index 0

            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            T_pred = max_seqlen - init_seqlen

            # For metrics
            error_ens = torch.zeros((batchsize, T_pred, cf.n_samples)).to(cf.device)

            # For exporting predictions (collect per-sample, then mean)
            pred_samples_latlon_deg = []  # list of (B, T_pred, 2) in degrees

            for i_sample in range(cf.n_samples):
                preds = trainers.sample(
                    model,
                    seqs_init,
                    T_pred,
                    temperature=1.0,
                    sample=True,
                    sample_mode=cf.sample_mode,
                    r_vicinity=cf.r_vicinity,
                    top_k=cf.top_k
                )
                # Convert inputs/preds back to degrees for metrics (and to radians for haversine)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                # (lat/lon only span; sog/cog are 0 in v_ranges)
                v_ranges = torch.tensor([(model.lat_max - model.lat_min), (model.lon_max - model.lon_min), 0, 0], device=cf.device)
                v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0], device=cf.device)

                # Degrees (for export)
                preds_deg = (preds * v_ranges + v_roi_min)[..., :2]  # (B, T_pred, 2) degrees
                pred_samples_latlon_deg.append(preds_deg.detach().cpu())

                # Radians (for distance metrics)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180.0
                pred_coords  = (preds  * v_ranges + v_roi_min) * torch.pi / 180.0
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            # === metrics aggregation (unchanged) ===
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

            # === export averaged predictions to CSV ===
            # Average across samples in DEG space (safer numerically here than averaging radians)
            preds_deg_mean = torch.stack(pred_samples_latlon_deg, dim=-1).mean(dim=-1).numpy()  # (B, T_pred, 2)

            # Build timestamps for future steps:
            # time index j in [0..T_pred-1] corresponds to absolute sequence index (init_seqlen + j)
            # Timestamp = time_start + (init_seqlen + j) * STEP_SECONDS
            time_starts_np = np.asarray(time_starts, dtype=np.int64)  # (B,)
            idx_offsets = init_seqlen + np.arange(T_pred, dtype=np.int64)  # (T_pred,)
            t_unix_batch = time_starts_np[:, None] + idx_offsets[None, :] * int(STEP_SECONDS)  # (B, T_pred)

            # Write batch
            mmsis_np = np.asarray(mmsis, dtype=np.int64)  # (B,)
            pred_writer.write_batch("test", mmsis_np, t_unix_batch, preds_deg_mean)



    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    # plt.xlim([0, 12])
    # plt.ylim([0, 5])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")
    plt.savefig("prediction_error.png")
    # Close CSV writer
    pred_writer.close()
    print("Wrote prediction CSV to:", os.path.join(cf.savedir, "predictions_out", "traisformer_preds_test.csv"))


    # Yeah, done!!!
