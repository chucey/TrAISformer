#%%
import os
from config_trAISformer import Config
import torch
import numpy as np
import pandas as pd
import pickle
from models import TrAISformer
from trainers import sample
#%%
cf = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# load data
data_path = cf.testset_name
with open(data_path, 'rb') as f:
    data: list[dict] = pickle.load(f)
print(f"Loaded {len(data)} samples from {data_path}")
#%%
# load trained model
traisformer_model_path = '/home/chucey/GQP/TrAISformer/results/us_continent_2024-pos-pos_vicinity-10-40-blur-True-False-2-1.0-data_size-8000-20000-60-360-embd_size-256-256-128-128-head-8-8-bs-32-lr-0.0006-seqlen-36-96-epochs-50/model.pt'

traisformer = TrAISformer(cf, partition_model=None)
traisformer.load_state_dict(torch.load(traisformer_model_path, map_location=device))
# %%
def prepare_traisformer_input(data: np.ndarray, device=device):
    tensor = torch.tensor(data[:, :4], dtype=torch.float32).unsqueeze(0).to(device)  # add batch dimension
    return tensor

# convert normalized back to lat/lon
lat_min, lat_max = cf.lat_min, cf.lat_max
lon_min, lon_max = cf.lon_min, cf.lon_max
max_sog = cf.sog_max
max_cog = cf.cog_max

def denormalize_coordinates(coordinates:np.ndarray) -> np.ndarray:
    lat = coordinates[:, 0] * (lat_max - lat_min) + lat_min
    lon = coordinates[:, 1] * (lon_max - lon_min) + lon_min
    sog = coordinates[:, 2] * max_sog  # speed over ground
    cog = coordinates[:, 3] * max_cog  # course over ground
    return np.column_stack((lat, lon, sog, cog))
# tensor = prepare_traisformer_input(data[0]['traj'], device)
# tensor.shape  # should be (1, seq_len, 4)
# %%
# make predictions
init_seqlen = 24

rows = []
seen_mmsis = set()
traj_threshold = 60 # max number of points in traj to be included
count = 0
max_count = 1000

for idx, V in enumerate(data):
    mmsi = int(V["mmsi"])
    traj = V["traj"]

    if mmsi in seen_mmsis or len(traj) > traj_threshold or len(traj) <= init_seqlen:
        continue
    
    try:
        print(f'Predicting voyage {mmsi}, {idx}, voyage length: {len(traj)}')
        seen_mmsis.add(mmsi)
        tensor = prepare_traisformer_input(traj, device)
        # print(tensor.shape)
        max_seqlen = len(traj) # 8 hours
        init_seq = tensor[:, :init_seqlen, :]
        traisformer.eval()
        with torch.no_grad():
            pred_seq = sample(model = traisformer,
                            seqs = init_seq,
                            steps = max_seqlen - init_seqlen,
                            sample=True)  # (1, pred_seqlen, 4)
        # pred_seq stacks the input on top of the predictions
        preds_np = pred_seq.detach().cpu().numpy().squeeze(0)  # (max_seqlen, 4)
        preds_np = preds_np[init_seqlen:, :] 

        # denormalize
        preds_denorm = denormalize_coordinates(preds_np)

        t_unix = traj[init_seqlen:, 5].astype(np.int64)
        lat_deg = preds_denorm[:, 0]
        lon_deg = preds_denorm[:, 1]
        sog_knots = preds_denorm[:, 2]
        cog_deg = preds_denorm[:, 3]

        for t, la, lo, sog, cog in zip(t_unix, lat_deg, lon_deg, sog_knots, cog_deg):
            rows.append((mmsi, int(t), float(la), float(lo), float(sog), float(cog)))
    except Exception as e:
        print(f'Error predicting voyage {mmsi}, {idx}: {e}')
        continue
   
    count += 1
    print(f'Voyages predicted {count} / {max_count}')
    if count >= max_count:
        break

df = pd.DataFrame(rows, columns=["mmsi", "t_unix", "lat_deg", "lon_deg", "sog_knots", "cog_deg"]) 
df.head()

# %%
# save predicitions
out_dir = os.path.join(cf.savedir, "eval_inputs")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "traisformer_preds_test.csv")

df.sort_values(["t_unix", "mmsi"]).to_csv(out_csv, index=False)
print("Wrote:", out_csv)
# %%
