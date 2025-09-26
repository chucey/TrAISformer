#!/usr/bin/env python3
"""
Optimized version of csv2pkl.py with significant performance improvements
"""
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Configuration
NUM_PROCESSES = min(mp.cpu_count(), 8)  # Use up to 8 processes
CHUNK_SIZE = 10000  # Process files in chunks
print(f"Using {NUM_PROCESSES} processes for parallel processing")

# Parameters (same as original)
LAT_MIN = 20
LAT_MAX = 60
LON_MIN = -160
LON_MAX = -60
D2C_MIN = 2000
SOG_MAX = 30

# File paths
dataset_path = "/home/chucey/GQP/"
pkl_filename = "us_continent_2024_track.pkl"
pkl_filename_train = "us_continent_2024_train_track.pkl"
pkl_filename_valid = "us_continent_2024_valid_track.pkl"
pkl_filename_test = "us_continent_2024_test_track.pkl"
cargo_tanker_filename = "us_continent_2024_cargo_tanker.npy"

# Time periods
t_train_min = time.mktime(time.strptime("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"))
t_train_max = time.mktime(time.strptime("2024-02-20T21:59:59", "%Y-%m-%dT%H:%M:%S"))
t_valid_min = time.mktime(time.strptime("2024-02-21T22:00:00", "%Y-%m-%dT%H:%M:%S"))
t_valid_max = time.mktime(time.strptime("2024-02-23T22:59:59", "%Y-%m-%dT%H:%M:%S"))
t_test_min = time.mktime(time.strptime("2024-02-24T23:00:00", "%Y-%m-%dT%H:%M:%S"))
t_test_max = time.mktime(time.strptime("2024-02-29T23:59:59", "%Y-%m-%dT%H:%M:%S"))
t_min = time.mktime(time.strptime("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"))
t_max = time.mktime(time.strptime("2024-02-29T23:59:59", "%Y-%m-%dT%H:%M:%S"))

# Column indices
LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO = list(range(11))
CARGO_TANKER_ONLY = False

def process_csv_file(args):
    """
    Process a single CSV file using pandas for speed.
    Returns processed data as numpy array.
    """
    csv_filename, dataset_path = args
    
    try:
        data_path = os.path.join(dataset_path, 'AISVesselTracks2024', csv_filename)
        
        print(f"Processing {csv_filename}...")
        
        # Read CSV with pandas (much faster than line-by-line)
        # Specify column names and types for speed
        column_names = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading',
                       'VesselName', 'IMO', 'CallSign', 'VesselType', 'Status', 'Length', 
                       'Width', 'Draft', 'Cargo', 'TranscieverClass']
        
        # Read with optimal dtypes
        df = pd.read_csv(data_path, 
                        names=column_names,
                        skiprows=1,  # Skip header
                        dtype={
                            'MMSI': 'int32',
                            'LAT': 'float32', 
                            'LON': 'float32',
                            'SOG': 'float32',
                            'COG': 'float32',
                            'Heading': 'float32',
                            'VesselType': 'int16',
                            'Length': 'int16',
                            'Width': 'int16',
                            'Cargo': 'float32'
                        },
                        parse_dates=['BaseDateTime'],
                        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S"))
        
        # Convert timestamp to unix timestamp
        df['Timestamp'] = df['BaseDateTime'].astype('int64') / 1e9
        
        # Filter data immediately to reduce memory usage
        mask = (
            (df['LAT'] >= LAT_MIN) & (df['LAT'] <= LAT_MAX) &
            (df['LON'] >= LON_MIN) & (df['LON'] <= LON_MAX) &
            (df['SOG'] >= 0) & (df['SOG'] <= SOG_MAX) &
            (df['COG'] >= 0) & (df['COG'] <= 360) &
            (df['Timestamp'] >= t_min) & (df['Timestamp'] <= t_max)
        )
        
        df_filtered = df[mask].copy()
        del df  # Free memory immediately
        
        if len(df_filtered) == 0:
            print(f"No valid data in {csv_filename}")
            return np.array([]).reshape(0, 11)
        
        # Create output array
        result = np.column_stack([
            df_filtered['LAT'].values.astype(np.float32),
            df_filtered['LON'].values.astype(np.float32),
            df_filtered['SOG'].values.astype(np.float32),
            df_filtered['COG'].values.astype(np.float32),
            df_filtered['Heading'].values.astype(np.float32),
            df_filtered['Timestamp'].values.astype(np.float64),
            df_filtered['MMSI'].values.astype(np.int32),
            df_filtered['VesselType'].values.astype(np.int16),
            df_filtered['Length'].values.astype(np.int16),
            df_filtered['Width'].values.astype(np.int16),
            df_filtered['Cargo'].values.astype(np.float32)
        ])
        
        print(f"Processed {csv_filename}: {len(result):,} valid messages")
        return result
        
    except Exception as e:
        print(f"Error processing {csv_filename}: {e}")
        return np.array([]).reshape(0, 11)

def split_and_save_data(m_msg):
    """
    Split data into train/valid/test and save to pickle files.
    """
    print("Splitting data into train/validation/test sets...")
    
    # Vectorized filtering for each set
    train_mask = (m_msg[:, TIMESTAMP] >= t_train_min) & (m_msg[:, TIMESTAMP] <= t_train_max)
    valid_mask = (m_msg[:, TIMESTAMP] >= t_valid_min) & (m_msg[:, TIMESTAMP] <= t_valid_max)
    test_mask = (m_msg[:, TIMESTAMP] >= t_test_min) & (m_msg[:, TIMESTAMP] <= t_test_max)
    
    m_msg_train = m_msg[train_mask]
    m_msg_valid = m_msg[valid_mask]
    m_msg_test = m_msg[test_mask]
    
    print(f"Train set: {len(m_msg_train):,} messages")
    print(f"Valid set: {len(m_msg_valid):,} messages") 
    print(f"Test set: {len(m_msg_test):,} messages")
    
    # Convert to vessel dictionaries in parallel
    datasets = [
        (m_msg_train, "train"),
        (m_msg_valid, "valid"),
        (m_msg_test, "test")
    ]
    
    for data, split_name in datasets:
        print(f"Processing {split_name} set...")
        vessel_dict = create_vessel_dict_parallel(data)
        
        # Save to pickle
        filename = f"us_continent_2024_{split_name}_track.pkl"
        output_path = os.path.join('data', 'US_data', filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(vessel_dict, f)
        
        print(f"Saved {split_name} set: {len(vessel_dict)} vessels to {filename}")

def create_vessel_dict_parallel(messages):
    """
    Create vessel dictionary from messages using parallel processing.
    """
    if len(messages) == 0:
        return {}
    
    # Group by MMSI using pandas for speed
    df = pd.DataFrame(messages, columns=['LAT', 'LON', 'SOG', 'COG', 'HEADING', 
                                       'TIMESTAMP', 'MMSI', 'SHIPTYPE', 'LENGTH', 
                                       'WIDTH', 'CARGO'])
    
    vessel_dict = {}
    for mmsi, group in tqdm(df.groupby('MMSI'), desc="Creating vessel tracks"):
        # Sort by timestamp
        track = group.sort_values('TIMESTAMP').values
        vessel_dict[int(mmsi)] = track.astype(np.float32)
    
    return vessel_dict

def main():
    """Main processing function with optimizations."""
    
    start_time = time.time()
    
    # Get list of CSV files
    ais_data_path = os.path.join(dataset_path, 'AISVesselTracks2024')
    if not os.path.exists(ais_data_path):
        print(f"Error: Directory {ais_data_path} does not exist!")
        return
    
    l_csv_filename = [f for f in os.listdir(ais_data_path) if f.endswith('.csv')]
    print(f"Found {len(l_csv_filename)} CSV files to process")
    
    if len(l_csv_filename) == 0:
        print("No CSV files found!")
        return
    
    # Prepare arguments for parallel processing
    file_args = [(csv_filename, dataset_path) for csv_filename in l_csv_filename]
    
    # Process CSV files in parallel
    print("\\n" + "="*60)
    print("LOADING AND PROCESSING CSV FILES")
    print("="*60)
    
    all_data = []
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_csv_file, args): args[0] for args in file_args}
        
        # Collect results
        for future in tqdm(as_completed(future_to_file), total=len(file_args), 
                          desc="Processing CSV files"):
            try:
                result = future.result()
                if len(result) > 0:
                    all_data.append(result)
            except Exception as e:
                filename = future_to_file[future]
                print(f"Error processing {filename}: {e}")
    
    # Combine all data
    print("\\nCombining all data...")
    if all_data:
        m_msg = np.vstack(all_data)
    else:
        print("No data to process!")
        return
    
    print(f"Total AIS messages: {len(m_msg):,}")
    print(f"Data shape: {m_msg.shape}")
    
    # Print statistics
    print("\\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Latitude range: {np.min(m_msg[:,LAT]):.2f} to {np.max(m_msg[:,LAT]):.2f}")
    print(f"Longitude range: {np.min(m_msg[:,LON]):.2f} to {np.max(m_msg[:,LON]):.2f}")
    print(f"Speed range: {np.min(m_msg[:,SOG]):.2f} to {np.max(m_msg[:,SOG]):.2f}")
    print(f"Unique vessels: {len(np.unique(m_msg[:,MMSI])):,}")
    
    # Save all messages to .npy file
    print("\\nSaving all AIS messages to .npy file...")
    os.makedirs(os.path.join('data', 'US_data'), exist_ok=True)
    np.save(os.path.join('data', 'US_data', 'all_msgs.npy'), m_msg)
    
    # Split and save data
    split_and_save_data(m_msg)
    
    # Performance summary
    total_time = time.time() - start_time
    print("\\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Messages processed: {len(m_msg):,}")
    print(f"Processing rate: {len(m_msg)/(total_time):.0f} messages/second")
    print(f"Files processed: {len(l_csv_filename)}")

if __name__ == '__main__':
    main()