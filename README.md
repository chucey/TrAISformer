# Maritime Vessel Analysis

This repo builds on the TrAISformer work found [here](https://github.com/CIA-Oceanix/TrAISformer). That repo contains a Pytorch implementation of TrAISformer---A generative transformer for AIS trajectory prediction (https://arxiv.org/abs/2109.03958). The transformer part is adapted from: https://github.com/karpathy/minGPT

To build on previous work, we adopted the TrAISformer architechure and tuned hyperparameters to make predictions on US vessel data. From there, we used those predicitions to detect the future positions of neighboring vessels to a target vessel.

We also, applied anomaly detection to detect normal and anomalous vessel trajectories, and behavior detection to charaterize vessel behaviors over time.

---

<p align="center">
  <img width="600" height="450" src="./figures/us_data.png">
</p>

### Requirements:

See requirements.txt

### Datasets:

The data gathered for this project was obtained from the [Marine Cadastre](https://hub.marinecadastre.gov/pages/vesseltraffic) and was preprocessed using the process described below.

**Note**
The data used in the original TrAISformer work was provided by the [Danish Maritime Authority (DMA)](https://dma.dk/safety-at-sea/navigational-information/ais-data). Please refer to [the paper](https://arxiv.org/abs/2109.03958) for the details of the pre-processing step. The code is available here: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py

A processed dataset can be found in `./data/ct_dma/`
(the format is `[lat, log, sog, cog, unix_timestamp, mmsi]`).

### Run

For data preprocssing run,`0-Data-prep/csv2pkl_optimized.py` to convert a list of .CSV files to pickle files in a highly optimzed way. Then, run `0-Data-prep/data_prep_v2.py` To further preprocess the data for use. The final format of the preprocessed data is: `[LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO],` and is saved in three pickle files for training, testing and validation.

Run `1-TrAISformer-code/trAISformer.py` to train and evaluate the model.
(Please note that the values given by the code are in km, while the values presented in the paper were converted to nautical mile.)

Run python find_neighbors.py \
  --preds_csv_target /path/to/predictions.csv \
  --radius_nm 5 \
  --out_csv neighbors_out/neighbor_list.csv 

to find neighbors based on a radius of 5nm and 300s time gate

Run python evaluate_neighbors_optimized.py \
  --pred_csv /path/to/predictions.csv \
  --gt_csv /path/to/ground_truth.csv \
  --radius_nm 5 \
  --time_tolerance_sec 300 \
  --neighbor_csv neighbors_out/neighbor_list.csv \
  --out_dir neighbors_eval_out

to evaluate the neighbors returned against the ground truths. 

Run `3-Anomaly-detection/dbscan.py` to apply DBSCAN to the cleaned dataset obtained from `0-Data-prep/data_prep_v2.py.` The file outputs the dbscan labels as a pickle file, and several evaluation metrics as a JSON file. Run `3-Anomaly-detection/label_and_train.py` to label the cleaned data, train a classifier and save the trained model as a pickle file.

We used three clustering methods in this project for Behavior Detection: K-Means, GMM, and AHC.
The notebooks contain the training/fitting code used to generate centroids: K means and GMM.ipynb, GMM and AHC.

Instead of re-running those notebooks every time, we saved the final centroids from all three methods into: Centroids.xlsx. This file contains the centroids for all 3 clustering methods and to generate behavior labels for trajectories, Make sure you have:
Centroids.xlsx
traj_list.pkl (input trajectories file) and finally Run:Final Function.ipynb.
This reads the centroids from Centroids.xlsx, loads the input trajectories from traj_list.pkl, and outputs the final behavior labels.

### License

See `LICENSE`

### Contact

For any questions, please open an issue and assign it to @dnguyengithub.

### Reference:

- AccessAIS - MarineCadastre.gov. (n.d.). Marinecadastre.gov. https://marinecadastre.gov/accessais/
