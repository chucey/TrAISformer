import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from collections import defaultdict


pkl_filename = "us_continent_2024_track.pkl"
pkl_filename_train = "us_continent_2024_train_track.pkl"
pkl_filename_valid = "us_continent_2024_valid_track.pkl"
pkl_filename_test  = "us_continent_2024_test_track.pkl"

cargo_tanker_filename = "us_continent_2024_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"))
t_train_max = time.mktime(time.strptime("2024-02-20T21:59:59", "%Y-%m-%dT%H:%M:%S"))
t_valid_min = time.mktime(time.strptime("2024-02-21T22:00:00", "%Y-%m-%dT%H:%M:%S"))
t_valid_max = time.mktime(time.strptime("2024-02-23T22:59:59", "%Y-%m-%dT%H:%M:%S"))
t_test_min  = time.mktime(time.strptime("2024-02-24T23:00:00", "%Y-%m-%dT%H:%M:%S"))
t_test_max  = time.mktime(time.strptime("2024-02-29T23:59:59", "%Y-%m-%dT%H:%M:%S"))

t_min = time.mktime(time.strptime("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"))
t_max = time.mktime(time.strptime("2024-02-29T23:59:59", "%Y-%m-%dT%H:%M:%S"))

CARGO_TANKER_ONLY = False
if  CARGO_TANKER_ONLY:
    pkl_filename = "ct_"+pkl_filename
    pkl_filename_train = "ct_"+pkl_filename_train
    pkl_filename_valid = "ct_"+pkl_filename_valid
    pkl_filename_test  = "ct_"+pkl_filename_test
    
print(pkl_filename_train)

LAT_MIN = 20      #9.0
LAT_MAX = 60       #14.0
LON_MIN = -160      #-71.0
LON_MAX = -60     #-66.0

SOG_MAX = 30

LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO  = list(range(11))

# Multiprocessing configuration
NUM_PROCESSES = min(mp.cpu_count(), 8)  # Use max 8 processes to avoid memory issues
print(f"Using {NUM_PROCESSES} processes for parallel processing")

def process_message_chunk(chunk_data):
    """
    Process a chunk of messages and group them by MMSI.
    Returns a dictionary of MMSI -> list of messages.
    """
    messages, cargo_tanker_only, l_cargo_tanker = chunk_data
    vessel_dict = defaultdict(list)
    
    for v_msg in messages:
        mmsi = int(v_msg[MMSI])
        if not cargo_tanker_only or mmsi in l_cargo_tanker:
            vessel_dict[mmsi].append(v_msg[:11])
    
    # Convert lists to numpy arrays
    for mmsi in vessel_dict:
        vessel_dict[mmsi] = np.array(vessel_dict[mmsi])
    
    return dict(vessel_dict)

def merge_vessel_dicts(dict_list):
    """
    Merge multiple vessel dictionaries from parallel workers.
    """
    merged = defaultdict(list)
    
    for vessel_dict in dict_list:
        for mmsi, messages in vessel_dict.items():
            merged[mmsi].extend(messages)
    
    # Convert lists back to numpy arrays
    final_dict = {}
    for mmsi, messages in merged.items():
        final_dict[mmsi] = np.array(messages)
    
    return final_dict

def get_timestamp(m_entry):
    """
    Helper function to extract timestamp from message entry.
    Module-level function for multiprocessing compatibility.
    """
    return m_entry[TIMESTAMP]

def sort_single_vessel(item):
    """
    Sort a single vessel's trajectory by timestamp.
    This function needs to be at module level for multiprocessing to work.
    """
    mmsi, track = item
    sorted_track = np.array(sorted(track, key=get_timestamp))
    return mmsi, sorted_track

def sort_vessel_tracks_parallel(vessel_dict, num_processes=None):
    """
    Sort vessel trajectories by timestamp in parallel.
    """
    if num_processes is None:
        num_processes = NUM_PROCESSES
    
    with Pool(num_processes) as pool:
        sorted_items = pool.map(sort_single_vessel, vessel_dict.items())
    
    return dict(sorted_items)

def create_vessel_tracks_parallel(messages, cargo_tanker_only=False, l_cargo_tanker=None, num_processes=None):
    """
    Create vessel tracks from messages using parallel processing.
    """
    if num_processes is None:
        num_processes = NUM_PROCESSES
    
    if l_cargo_tanker is None:
        l_cargo_tanker = []
    
    # Split messages into chunks for parallel processing
    chunk_size = max(1, len(messages) // (num_processes * 4))  # 4 chunks per process
    chunks = []
    
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        chunks.append((chunk, cargo_tanker_only, l_cargo_tanker))
    
    print(f"Processing {len(messages)} messages in {len(chunks)} chunks using {num_processes} processes...")
    
    # Process chunks in parallel
    with Pool(num_processes) as pool:
        chunk_results = list(tqdm(
            pool.imap(process_message_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # Merge results
    print("Merging results...")
    merged_dict = merge_vessel_dicts(chunk_results)
    
    # Sort trajectories in parallel
    print("Sorting trajectories...")
    sorted_dict = sort_vessel_tracks_parallel(merged_dict, num_processes)
    
    return sorted_dict

m_msg = np.load(os.path.join('data', 'US_data','all_msgs.npy'))

if __name__ == '__main__':

## Vessel Type    
#======================================
# print("Selecting vessel type ...")
# def sublist(lst1, lst2):
#    ls1 = [element for element in lst1 if element in lst2]
#    ls2 = [element for element in lst2 if element in lst1]
#    return (len(ls1) != 0) and (ls1 == ls2)

# VesselTypes = dict()
# l_mmsi = []
# n_error = 0
# for v_msg in tqdm(m_msg):
#     try:
#         mmsi_ = v_msg[MMSI]
#         type_ = v_msg[SHIPTYPE]
#         if mmsi_ not in l_mmsi :
#             VesselTypes[mmsi_] = [type_]
#             l_mmsi.append(mmsi_)
#         elif type_ not in VesselTypes[mmsi_]:
#             VesselTypes[mmsi_].append(type_)
#     except:
#         n_error += 1
#         continue
# print(n_error)
# for mmsi_ in tqdm(list(VesselTypes.keys())):
#     VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])
    
    l_cargo_tanker = []
    # l_fishing = []
    # for mmsi_ in list(VesselTypes.keys()):
    #     if sublist(VesselTypes[mmsi_], list(range(70,80))) or sublist(VesselTypes[mmsi_], list(range(80,90))):
    #         l_cargo_tanker.append(mmsi_)
    #     if sublist(VesselTypes[mmsi_], [30]):
    #         l_fishing.append(mmsi_)


    # print("Total number of vessels: ",len(VesselTypes))
    # print("Total number of cargos/tankers: ",len(l_cargo_tanker))
    # print("Total number of fishing: ",len(l_fishing))

    # print("Saving vessels' type list to ", cargo_tanker_filename)
    # np.save(cargo_tanker_filename,l_cargo_tanker)
    # np.save(cargo_tanker_filename.replace("_cargo_tanker.npy","_fishing.npy"),l_fishing)


    ## FILTERING 
    #======================================
    # Selecting AIS messages in the ROI and in the period of interest.

    ## LAT LON
    m_msg = m_msg[m_msg[:,LAT]>=LAT_MIN]
    m_msg = m_msg[m_msg[:,LAT]<=LAT_MAX]
    m_msg = m_msg[m_msg[:,LON]>=LON_MIN]
    m_msg = m_msg[m_msg[:,LON]<=LON_MAX]
    # SOG
    m_msg = m_msg[m_msg[:,SOG]>=0]
    m_msg = m_msg[m_msg[:,SOG]<=SOG_MAX]
    # COG
    m_msg = m_msg[m_msg[:,SOG]>=0]
    m_msg = m_msg[m_msg[:,COG]<=360]
    # D2C
    # m_msg = m_msg[m_msg[:,D2C]>=D2C_MIN]

    # TIME
    m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]

    m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
    m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]

    m_msg_train = m_msg[m_msg[:,TIMESTAMP]>=t_train_min]
    m_msg_train = m_msg_train[m_msg_train[:,TIMESTAMP]<=t_train_max]
    unique_m_msg_train_mmsi = set(m_msg_train[:, MMSI])

    m_msg_valid = m_msg[m_msg[:,TIMESTAMP]>=t_valid_min]
    m_msg_valid = m_msg_valid[m_msg_valid[:,TIMESTAMP]<=t_valid_max]
    unique_m_msg_valid_mmsi = set(m_msg_valid[:, MMSI])

    m_msg_test  = m_msg[m_msg[:,TIMESTAMP]>=t_test_min]
    m_msg_test  = m_msg_test[m_msg_test[:,TIMESTAMP]<=t_test_max]
    unique_m_msg_test_mmsi = set(m_msg_test[:, MMSI])

    print("Total msgs: ",len(m_msg))
    print("Number of msgs in the training set: ",len(m_msg_train))
    print("number of unique mmsi training set", len(unique_m_msg_train_mmsi))
    print("Number of msgs in the validation set: ",len(m_msg_valid))
    print("number of unique mmsi validation set", len(unique_m_msg_valid_mmsi))
    print("Number of msgs in the test set: ",len(m_msg_test))
    print("number of unique mmsi test set", len(unique_m_msg_test_mmsi))


    ## MERGING INTO DICT
    #======================================
    # Creating AIS tracks from the list of AIS messages.
    # Each AIS track is formatted by a dictionary.
    print("Convert to dicts of vessel's tracks...")

    #

    # Training set
    print("Creating training set tracks...")
    Vs_train = create_vessel_tracks_parallel(
        m_msg_train, 
        cargo_tanker_only=CARGO_TANKER_ONLY, 
        l_cargo_tanker=l_cargo_tanker
    )    # Vs_train_list = []
    # for item in tqdm(unique_m_msg_train_mmsi):
    #     Vs_train = dict()
    #     item = int(item)
    #     Vs_train['mmsi'] = item
    #     Vs_train['traj'] = m_msg_train[m_msg_train[:,MMSI]==item]
    #     Vs_train_list.append(Vs_train)

    # Validation set
    print("Creating validation set tracks...")
    Vs_valid = create_vessel_tracks_parallel(
        m_msg_valid, 
        cargo_tanker_only=CARGO_TANKER_ONLY, 
        l_cargo_tanker=l_cargo_tanker
    )

    # Vs_valid_list = []
    # for item in tqdm(unique_m_msg_valid_mmsi):
    #     Vs_valid = dict()
    #     item = int(item)
    #     Vs_valid['mmsi'] = item
    #     Vs_valid['traj'] = m_msg_valid[m_msg_valid[:,MMSI]==item]
    #     Vs_valid_list.append(Vs_valid)

    # Test set
    print("Creating test set tracks...")
    Vs_test = create_vessel_tracks_parallel(
        m_msg_test, 
        cargo_tanker_only=CARGO_TANKER_ONLY, 
        l_cargo_tanker=l_cargo_tanker
    )

    # Vs_test_list = []
    # for item in tqdm(unique_m_msg_test_mmsi):
    #     Vs_test = dict()
    #     item = int(item)
    #     Vs_test['mmsi'] = item
    #     Vs_test['traj'] = m_msg_test[m_msg_test[:,MMSI]==item]
    #     Vs_test_list.append(Vs_test)

    ## PICKLING
    #======================================
    for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
                                  [Vs_train,Vs_valid,Vs_test]
                                 ):
        print("Writing to ", os.path.join('data', 'US_data',filename),"...")
        with open(os.path.join('data', 'US_data',filename),"wb") as f:
            pickle.dump(filedict,f)
        print("Total number of tracks: ", len(filedict))
