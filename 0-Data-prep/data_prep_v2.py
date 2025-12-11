# %%
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
from tqdm import tqdm_notebook as tqdm
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO
# import csv2pkl
from tqdm import tqdm
import argparse

# %%
vessel_type = 'fishing'  # Set to None to process all vessel types, or specify a type like 'tankers_and_cargo', 'passenger', etc. 

# define a dictionary to map vessel types to preprocessing parameters
vessel_params = {
    'tankers_and_cargo': {'SPEED_MAX': 30,
                          'DURATION_MAX': 24, # 1 day
                          'INTERVAL_MAX': 2*3600, # 2h
                          'LEN_VOYAGE': 20, # 20 sequence points
                          'DURATION': 4*3600, # 4 hours
                          'LOW_SPEED_THRESHOLD': 0.8}, 
    'passenger': {'SPEED_MAX': 30,
                  'DURATION_MAX': 12,
                  'INTERVAL_MAX': 2*3600,
                  'LEN_VOYAGE': 20,
                  'DURATION': 4*3600,
                  'LOW_SPEED_THRESHOLD': 0.8},
    'fishing': {'SPEED_MAX': 30,
                'DURATION_MAX': 24,     # Increased to 24 hours like other vessels
                'INTERVAL_MAX': 4*3600, # Increased to 4 hours - fishing vessels have more sparse reporting
                'LEN_VOYAGE': 10,       # Increased minimum points for better quality
                'DURATION': 2*3600, # Increased to 2 hours minimum duration
                'LOW_SPEED_THRESHOLD': 0.9},    # Increased to 0.9
    # Add more vessel types and their parameters as needed
}

if vessel_type is not None:
    print(f"Processing vessel type: {vessel_type}")
    dataset_dir = os.path.join(os.getcwd(), "data", "US_data", vessel_type)
    l_output_filepath = os.path.join(os.getcwd(),"data","US_data","cleaned_data", vessel_type)
else:
    print("Processing all vessel types")
    dataset_dir = os.path.join(os.getcwd(),"data","US_data")
    l_output_filepath = os.path.join(os.getcwd(),"data","US_data","cleaned_data")

l_input_filepath = [
    "us_continent_2024_valid_track.pkl",
    "us_continent_2024_train_track.pkl",
    "us_continent_2024_test_track.pkl"
]

# %%
# These are coordinate bounds for the US dataset used in this project. These only include areas off the coast of mainland US but including Alaska.

# You may need to change them for other datasets
LAT_MIN = 20      
LAT_MAX = 60       
LON_MIN = -160      
LON_MAX = -60    

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30  # knots
DURATION_MAX = 24 #h

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO  = list(range(11))

#%%

for filename in l_input_filepath:
    dict_list = []
    filename_list = []
    filename_list.append(filename)

    
    with open(os.path.join(dataset_dir,filename),"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)
    print(f"Loaded {filename}, length: {len(temp)}")       
    print(f" Removing erroneous timestamps and erroneous speeds from file {filename}...")
    # print(temp)

    Vs = dict()
    for Vi,file in zip(dict_list,filename_list):
        # print(Vi,file)
        for mmsi in list(Vi.keys()):       
            # Boundary
            lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                    (Vi[mmsi][:,LAT] < LAT_MIN))
            Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]
            lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                    (Vi[mmsi][:,LON] < LON_MIN))
            Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]
    #
            SPEED_MAX_VESSEL = vessel_params[vessel_type]['SPEED_MAX'] if vessel_type else SPEED_MAX
            abnormal_speed_idx = Vi[mmsi][:,SOG] > SPEED_MAX_VESSEL
            Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_speed_idx)]
            # Deleting empty keys
            if len(Vi[mmsi]) == 0:
                del Vi[mmsi]
                continue
            if mmsi not in list(Vs.keys()):
                Vs[mmsi] = Vi[mmsi]
                del Vi[mmsi]
            else:
                Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
                del Vi[mmsi]
    del dict_list, Vi, abnormal_speed_idx

    print(f" After removing erroneous timestamps and speeds from file: {file}, length: {len(Vs)}\n")

     ## STEP 2: VOYAGES SPLITTING 
    #======================================
    # Cutting discontiguous voyages into contiguous ones
    print("Cutting discontiguous voyages into contiguous ones...\n")
    count = 0
    voyages = dict()
    INTERVAL_MAX = vessel_params[vessel_type]['INTERVAL_MAX'] if vessel_type else 2*3600 # Use vessel-specific or default 2h
    for mmsi in list(Vs.keys()):
        v = Vs[mmsi]
        # Intervals between successive messages in a track
        intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
        idx = np.where(intervals > INTERVAL_MAX)[0]
        if len(idx) == 0:
            voyages[count] = v
            count += 1
        else:
            tmp = np.split(v,idx+1)
            for t in tmp:
                voyages[count] = t
                count += 1

    # STEP 3: REMOVING SHORT VOYAGES
    #======================================
    # Removing AIS track whose length is smaller than threshold or duration too short
    LEN_VOYAGE = vessel_params[vessel_type]['LEN_VOYAGE'] if vessel_type else 20
    MIN_DURATION = vessel_params[vessel_type]['DURATION'] if vessel_type else 4*3600
    print(f"Removing AIS track whose length is smaller than {LEN_VOYAGE} or those last less than {MIN_DURATION/3600:.1f} hours...")

    removed_count = 0
    for k in list(voyages.keys()):
        duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
        if (len(voyages[k]) < LEN_VOYAGE) or (duration < MIN_DURATION):
            voyages.pop(k, None)
            removed_count += 1
    
    if vessel_type == 'fishing':
        print(f"  Removed {removed_count} short voyages from fishing vessels")
    print(f" After removing short voyages, length: {len(voyages)}\n")

    ## STEP 4: SAMPLING
    #======================================
    # Sampling, resolution = 10 min
    print('Sampling...')
    Vs = dict()
    count = 0
    for k in tqdm(list(voyages.keys())):
        v = voyages[k]
        sampling_track = np.empty((0, 11)) # [Lat, Lon, SOG, COG, Heading, Timestamp, MMSI, ShipType, Length, Width, Cargo]
        for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 600): # 10 min
            tmp = utils.interpolate(t,v)
            if tmp is not None:
                sampling_track = np.vstack([sampling_track, tmp])
            else:
                sampling_track = None
                break
        if sampling_track is not None:
            Vs[count] = sampling_track
            count += 1
    print(f" After sampling file: {file}, length: {len(Vs)}\n")

    ## STEP 5: RE-SPLITTING
    #======================================
    print(f'Re-Splitting file: {file}...')
    Data = dict()
    count = 0
    DURATION_MAX_VESSEL = vessel_params[vessel_type]['DURATION_MAX'] if vessel_type else DURATION_MAX
    for k in tqdm(list(Vs.keys())): 
        v = Vs[k]
        # Split AIS track into small tracks whose duration <= vessel-specific max duration
        idx = np.arange(0, len(v), 6*DURATION_MAX_VESSEL)[1:]  # 6 samples per hour with 10-minute intervals
        tmp = np.split(v,idx)
        for subtrack in tmp:
            # only use tracks whose duration >= vessel-specific minimum
            MIN_DURATION_SAMPLES = int(vessel_params[vessel_type]['DURATION'] / 600) if vessel_type else 12*4  # Convert seconds to 10-minute samples
            if len(subtrack) >= MIN_DURATION_SAMPLES:
                Data[count] = subtrack
                count += 1
    print(f" After re-splitting file: {file}, length: {len(Data)}\n")

    ## STEP 6: REMOVING LOW SPEED TRACKS
    #======================================
    print(f"Removing 'low speed' tracks from file: {file}...")
    # More lenient for fishing vessels
    low_speed_threshold = vessel_params[vessel_type]['LOW_SPEED_THRESHOLD'] if vessel_type else 0.8
    print(f"  Using low speed threshold: {low_speed_threshold:.1%} for {vessel_type if vessel_type else 'all'} vessels")
    
    removed_low_speed = 0
    for k in tqdm(list(Data.keys())):
        d_L = float(len(Data[k]))
        if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > low_speed_threshold:
            Data.pop(k,None)
            removed_low_speed += 1
    
    if vessel_type == 'fishing':
        print(f"  Removed {removed_low_speed} low-speed tracks from fishing vessels")
    print(f" After removing 'low speed' tracks from file: {file}, length: {len(Data)}\n")

    ## STEP 7: NORMALISATION
    #======================================
    print(f'Normalisation file: {file}...')
    for k in tqdm(list(Data.keys())):
        v = Data[k]
        v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
        v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
        SPEED_MAX_VESSEL = vessel_params[vessel_type]['SPEED_MAX'] if vessel_type else SPEED_MAX
        v[:,SOG][v[:,SOG] > SPEED_MAX_VESSEL] = SPEED_MAX_VESSEL
        v[:,SOG] = v[:,SOG]/SPEED_MAX_VESSEL
        v[:,COG] = v[:,COG]/360.0

    ## STEP 8: REARRANGE DATA FOR TRAISFORMER
    #======================================
    print(f'Rearranging data for TrAISformer file: {file}...')
    data_list = []
    for key in tqdm(list(Data.keys())):
        data_dict = {}
        data_dict['mmsi'] = int(Data[key][0, MMSI])
        data_dict['traj'] = Data[key]
        # Rearrange data for TrAISformer
        data_list.append(data_dict)

    ## STEP 9: SAVE CLEANED DATA
    #======================================
    print(f'Saving cleaned data to {l_output_filepath}...\n')
    if not os.path.exists(l_output_filepath):
        os.makedirs(l_output_filepath)
    output_filepath = os.path.join(l_output_filepath, file)
    with open(output_filepath, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Saved cleaned data to {output_filepath}, length: {len(data_list)}\n")
    
    # Summary for fishing vessels
    if vessel_type:
        print(f"=== {vessel_type.upper()} VESSEL PROCESSING SUMMARY for {file} ===")
        print(f"Final tracks: {len(data_list)}")
        print(f"Processing parameters used:")
        print(f"  INTERVAL_MAX: {vessel_params[vessel_type]['INTERVAL_MAX']/3600:.1f} hours")
        print(f"  LEN_VOYAGE: {vessel_params[vessel_type]['LEN_VOYAGE']} points")
        print(f"  MIN_DURATION: {vessel_params[vessel_type]['DURATION']/3600:.1f} hours") 
        print(f"  Low speed threshold: {low_speed_threshold:.1%}")
        print("="*50)
#%%
# cleaned_data_dir = os.path.join(os.getcwd(),"data","US_data","cleaned_data")
# os.listdir(cleaned_data_dir)

# for file in os.listdir(cleaned_data_dir):
#     with open(os.path.join(cleaned_data_dir,file),"rb") as f:
#         temp = pickle.load(f)
#         print(f"Loaded {file}, length: {len(temp)}")
    
#         print(f'Rearranging data for TrAISformer file: {file}...')
#         data_list = []
#         for key in tqdm(list(temp.keys())):
#             data_dict = {}
#             data_dict['mmsi'] = int(temp[key][0, MMSI])
#             data_dict['traj'] = temp[key]
#             # Rearrange data for TrAISformer
#             data_list.append(data_dict)
        
#     with open(os.path.join(cleaned_data_dir,file), 'wb') as f:
#         pickle.dump(data_list, f)
#     print(f"Saved cleaned data to {os.path.join(cleaned_data_dir,file.split('.pkl')[0]+'_track.pkl')}, length: {len(data_list)}\n")

# %%
