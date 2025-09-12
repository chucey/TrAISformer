import os, pickle, numpy as np

# -------- config --------
dataset_name = "ct_dma"          # this becomes the filename prefix
datadir = f"./data/{dataset_name}/"
infile = "data/ct_dma/ct_from_csv.pkl"   # your single source file (Nx6 array)
ratios = (0.8, 0.1, 0.1)         # train/valid/test
seed = 42

# -------- load --------
with open(infile, "rb") as f:
    arr = pickle.load(f)         # expect shape (N, 6): [lat, lon, sog, cog, unix_ts, mmsi]

# ensure float array; keep mmsi for grouping (often stored as float)
arr = np.asarray(arr, dtype=np.float32)

# unique MMSIs as integers for grouping
mmsi_int = arr[:, 5].astype(np.int64)
uniq = np.unique(mmsi_int)

# shuffle vessels reproducibly
rng = np.random.default_rng(seed)
rng.shuffle(uniq)

n = len(uniq)
n_train = int(n * ratios[0])
n_valid = int(n * (ratios[0] + ratios[1]))

m_train = set(uniq[:n_train])
m_valid = set(uniq[n_train:n_valid])
m_test  = set(uniq[n_valid:])

mask_train = np.isin(mmsi_int, list(m_train))
mask_valid = np.isin(mmsi_int, list(m_valid))
mask_test  = np.isin(mmsi_int, list(m_test))

def sort_by_mmsi_time(x):
    if len(x) == 0: return x
    # primary key: mmsi (col 5), secondary: unix_ts (col 4)
    order = np.lexsort((x[:, 4], x[:, 5]))
    return x[order]

train = sort_by_mmsi_time(arr[mask_train])
valid = sort_by_mmsi_time(arr[mask_valid])
test  = sort_by_mmsi_time(arr[mask_test])

os.makedirs(datadir, exist_ok=True)
with open(f"{datadir}{dataset_name}_train.pkl", "wb") as f: pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"{datadir}{dataset_name}_valid.pkl", "wb") as f: pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"{datadir}{dataset_name}_test.pkl",  "wb") as f: pickle.dump(test,  f, protocol=pickle.HIGHEST_PROTOCOL)

print("rows:", len(train), len(valid), len(test))
print("vessels:", len(m_train), len(m_valid), len(m_test))