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
import multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

# %%
vessel_type = 'tankers_and_cargo'  # Set to None to process all vessel types, or specify a type like 'tankers_and_cargo', 'passenger', etc. 

if vessel_type is not None:
    print(f"Processing vessel type: {vessel_type}")
    dataset_dir = os.path.join(os.getcwd(), "data", "US_data", vessel_type)
else:
    print("Processing all vessel types")
    dataset_dir = os.path.join(os.getcwd(),"data","US_data")

l_input_filepath = [
    "us_continent_2024_valid_track.pkl",
    "us_continent_2024_train_track.pkl",
    "us_continent_2024_test_track.pkl"
]

if vessel_type is not None:
    l_output_filepath = os.path.join(os.getcwd(),"data","US_data","cleaned_data", vessel_type)
else:
    l_output_filepath = os.path.join(os.getcwd(),"data","US_data","cleaned_data")

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

# Optimization configuration
NUM_PROCESSES = min(mp.cpu_count(), 8)  # Use up to 8 processes to avoid memory issues
CHUNK_SIZE = 1000  # Process vessels in chunks to manage memory
print(f"Using {NUM_PROCESSES} processes for parallel processing")

def process_vessel_batch(args):
    """
    Process a batch of vessels through all preprocessing steps.
    This function will be run in parallel.
    """
    vessels_batch, filename = args
    
    # Constants
    INTERVAL_MAX = 2*3600  # 2h
    processed_vessels = {}
    
    for mmsi, v in vessels_batch.items():
        try:
            # Step 1: Already done (boundary and speed filtering)
            
            # Step 2: Voyage splitting
            intervals = v[1:, TIMESTAMP] - v[:-1, TIMESTAMP]
            idx = np.where(intervals > INTERVAL_MAX)[0]
            
            if len(idx) == 0:
                voyages = [v]
            else:
                voyages = np.split(v, idx+1)
            
            # Step 3: Remove short voyages
            valid_voyages = []
            for voyage in voyages:
                duration = voyage[-1, TIMESTAMP] - voyage[0, TIMESTAMP]
                if (len(voyage) >= 20) and (duration >= 4*3600):
                    valid_voyages.append(voyage)
            
            # Step 4: Optimized sampling with vectorized interpolation (600-second intervals)
            for voyage_idx, voyage in enumerate(valid_voyages):
                sampled_voyage = optimized_sampling(voyage)
                if sampled_voyage is not None and len(sampled_voyage) > 0:
                    # Step 5: Re-splitting (24h max duration)
                    resplit_tracks = resplit_voyage(sampled_voyage)
                    
                    for track in resplit_tracks:
                        # Minimum 4 hours: 4 hours * 6 samples/hour = 24 samples
                        if len(track) >= 24:  # >= 4 hours with 600-second (10-minute) sampling
                            # Step 6: Remove low speed tracks
                            if np.count_nonzero(track[:, SOG] < 2) / len(track) <= 0.8:
                                # Step 7: Normalization
                                normalized_track = normalize_track(track.copy())
                                
                                # Store processed track
                                key = f"{mmsi}_{voyage_idx}_{len(processed_vessels)}"
                                processed_vessels[key] = normalized_track
        
        except Exception as e:
            print(f"Error processing vessel {mmsi}: {e}")
            continue
    
    return processed_vessels

def optimized_sampling(voyage):
    """
    Optimized sampling function that reduces interpolation calls.
    Samples every 600 seconds (10 minutes).
    """
    sampling_track = []
    start_time = int(voyage[0, TIMESTAMP])
    end_time = int(voyage[-1, TIMESTAMP])
    
    # Pre-compute all required timestamps - every 600 seconds (10 minutes)
    timestamps = np.arange(start_time, end_time, 600)  # 10 min intervals
    
    for t in timestamps:
        interpolated = utils.interpolate(t, voyage)
        if interpolated is not None:
            sampling_track.append(interpolated)
        else:
            return None
    
    return np.array(sampling_track) if sampling_track else None

def resplit_voyage(voyage):
    """
    Split voyage into 24h segments.
    With 600-second (10-minute) sampling: 6 samples/hour * 24 hours = 144 samples per day.
    """
    DURATION_MAX = 24  # hours
    samples_per_hour = 6  # 6 samples per hour (1 sample per 10 minutes)
    samples_per_day = samples_per_hour * DURATION_MAX  # 144 samples per day
    
    idx = np.arange(0, len(voyage), samples_per_day)[1:]
    return np.split(voyage, idx) if len(idx) > 0 else [voyage]

def normalize_track(track):
    """
    Vectorized normalization of a track.
    """
    track[:, LAT] = (track[:, LAT] - LAT_MIN) / (LAT_MAX - LAT_MIN)
    track[:, LON] = (track[:, LON] - LON_MIN) / (LON_MAX - LON_MIN)
    track[:, SOG] = np.clip(track[:, SOG], 0, SPEED_MAX) / SPEED_MAX
    track[:, COG] = track[:, COG] / 360.0
    return track

def process_single_file(filename):
    """
    Process a single file with all optimization steps.
    """
    dataset_dir = os.path.join(os.getcwd(),"data","US_data")
    
    print(f"\n{'='*50}")
    print(f"Processing {filename}")
    print(f"{'='*50}")
    
    # Load data
    with open(os.path.join(dataset_dir, filename), "rb") as f:
        temp = pickle.load(f)
    print(f"Loaded {filename}, length: {len(temp)}")
    
    # Step 1: Remove erroneous data (vectorized)
    print("Removing erroneous timestamps and speeds...")
    Vs = {}
    for mmsi, track in tqdm(temp.items(), desc="Filtering"):
        # Vectorized boundary filtering
        valid_mask = (
            (track[:, LAT] >= LAT_MIN) & (track[:, LAT] <= LAT_MAX) &
            (track[:, LON] >= LON_MIN) & (track[:, LON] <= LON_MAX) &
            (track[:, SOG] >= 0) & (track[:, SOG] <= SPEED_MAX) &
            (track[:, TIMESTAMP] >= 0)
        )
        filtered_track = track[valid_mask]
        
        if len(filtered_track) > 0:
            Vs[mmsi] = filtered_track
    
    print(f"After filtering: {len(Vs)} vessels")
    
    # Parallel processing of vessel batches
    print("Processing vessels in parallel...")
    vessel_items = list(Vs.items())
    
    # Split into batches for parallel processing
    batches = []
    for i in range(0, len(vessel_items), CHUNK_SIZE):
        batch = dict(vessel_items[i:i + CHUNK_SIZE])
        batches.append((batch, filename))
    
    # Process batches in parallel
    all_processed = {}
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        future_to_batch = {executor.submit(process_vessel_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
            try:
                batch_result = future.result()
                all_processed.update(batch_result)
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    print(f"After processing: {len(all_processed)} tracks")
    
    # Convert to list format for TrAISformer
    data_list = []
    for key, track in all_processed.items():
        data_dict = {
            'mmsi': int(track[0, MMSI]),
            'traj': track
        }
        data_list.append(data_dict)
    
    # Save results
    if not os.path.exists(l_output_filepath):
        os.makedirs(l_output_filepath)
    
    output_filepath = os.path.join(l_output_filepath, filename)
    with open(output_filepath, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"Saved {len(data_list)} tracks to {output_filepath}")
    return len(data_list)

#%%

if __name__ == '__main__':
    # Process files
    l_input_filepath = [
        "us_continent_2024_valid_track.pkl",
        "us_continent_2024_train_track.pkl", 
        "us_continent_2024_test_track.pkl"
    ]
    
    start_time = time.time()
    total_tracks = 0
    
    for filename in l_input_filepath:
        file_start = time.time()
        tracks_processed = process_single_file(filename)
        file_time = time.time() - file_start
        total_tracks += tracks_processed
        
        print(f"File {filename} completed in {file_time/60:.1f} minutes")
        print(f"Processed {tracks_processed} tracks")
        print()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL FILES COMPLETED!")
    print(f"Total processing time: {total_time/3600:.2f} hours")
    print(f"Total tracks processed: {total_tracks}")
    print(f"Average processing rate: {total_tracks/(total_time/3600):.0f} tracks/hour")
    print(f"{'='*60}")
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
