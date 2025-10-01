import csv
from pathlib import Path

class PredictionWriter:
    """Write predicted lat/lon for each (mmsi, t) to CSV."""
    def __init__(self, out_dir="predictions_out"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writers = {}

    def _get_writer(self, split="val"):
        p = self.out_dir / f"traisformer_preds_{split}.csv"
        if split not in self._writers:
            f = open(p, "w", newline="")
            w = csv.writer(f)
            w.writerow(["mmsi", "t_unix", "lat_deg", "lon_deg"])
            self._writers[split] = (f, w)
        return self._writers[split][1]

    def close(self):
        for f, _ in self._writers.values():
            f.close()

    def write_batch(self, split, mmsi_batch, t_unix_batch, latlon_batch):
        """
        Args:
          split: "train"/"val"/"test"
          mmsi_batch: (B,) list/1D tensor of MMSI (ints)
          t_unix_batch: (B, T_pred) unix seconds for each future step
          latlon_batch: (B, T_pred, 2) lat/lon in degrees
        """
        w = self._get_writer(split)
        B, T = len(mmsi_batch), len(t_unix_batch[0])
        for i in range(B):
            mmsi = int(mmsi_batch[i])
            for t in range(T):
                t_unix = int(t_unix_batch[i][t])
                lat = float(latlon_batch[i][t][0])
                lon = float(latlon_batch[i][t][1])
                w.writerow([mmsi, t_unix, lat, lon])
