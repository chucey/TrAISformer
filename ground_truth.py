#!/usr/bin/env python3
"""
Dump ground-truth positions from the test pickle into CSV
so they can be directly compared with TrAISformer predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from config_trAISformer import Config


def main():
    cf = Config()

    init_seqlen = cf.init_seqlen
    time_horizon = 5 #in hours
    time_steps = time_horizon * 6  # 10-min intervals
    max_sequlen = init_seqlen + time_steps

    test_pkl = cf.testset_name  # full path defined in config
    print("Reading:", test_pkl)
    with open(test_pkl, "rb") as f:
        l_data = pickle.load(f)

    # denormalize helpers (same bounds used when making the dataset)
    lat_min, lat_max = cf.lat_min, cf.lat_max
    lon_min, lon_max = cf.lon_min, cf.lon_max

    rows = []
    for V in l_data:
        mmsi = int(V["mmsi"])
        traj = V["traj"][init_seqlen:max_sequlen, :]
        # traj columns: [LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO]

        if len(traj) == 0 or np.isnan(traj[:, :2]).any():
            continue

        # denormalize to degrees
        lat_deg = traj[:, 0] * (lat_max - lat_min) + lat_min
        lon_deg = traj[:, 1] * (lon_max - lon_min) + lon_min
        t_unix  = traj[:, 5].astype(np.int64)

        for t, la, lo in zip(t_unix, lat_deg, lon_deg):
            rows.append((mmsi, int(t), float(la), float(lo)))

    df = pd.DataFrame(rows, columns=["mmsi", "t_unix", "lat_deg", "lon_deg"])

    out_dir = os.path.join(cf.savedir, "eval_inputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ground_truth_positions.csv")

    df.sort_values(["t_unix", "mmsi"]).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
