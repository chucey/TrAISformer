# TrAISformer

Pytorch implementation of TrAISformer---A generative transformer for AIS trajectory prediction (https://arxiv.org/abs/2109.03958).

The transformer part is adapted from: https://github.com/karpathy/minGPT

---
<p align="center">
  <img width="600" height="450" src="./figures/t18_3.png">
</p>


#### Requirements: 
See requirements.yml

### Datasets:

The data used in this paper are provided by the [Danish Maritime Authority (DMA)](https://dma.dk/safety-at-sea/navigational-information/ais-data). 
Please refer to [the paper](https://arxiv.org/abs/2109.03958) for the details of the pre-processing step. The code is available here: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py

A processed dataset can be found in `./data/ct_dma/`
(the format is `[lat, log, sog, cog, unix_timestamp, mmsi]`).

### Run

Run `trAISformer.py` to train and evaluate the model.
(Please note that the values given by the code are in km, while the values presented in the paper were converted to nautical mile.)


We used three clustering methods in this project for Behavior Detection: K-Means, GMM, and AHC.
The notebooks contain the training/fitting code used to generate centroids: K means and GMM.ipynb, GMM and AHC.
Instead of re-running those notebooks every time, we saved the final centroids from all three methods into: Centroids.xlsx. This file contains the centroids for all 3 clustering methods and
to generate behavior labels for trajectories, Make sure you have:
Centroids.xlsx
traj_list.pkl (input trajectories file) and finally Run:Final Function.ipynb.
This reads the centroids from Centroids.xlsx, loads the input trajectories from traj_list.pkl, and outputs the final behavior labels.


### License

See `LICENSE`

### Contact
For any questions, please open an issue and assign it to @dnguyengithub.

