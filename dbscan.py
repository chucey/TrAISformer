#%%
import os
import json
import numpy as np
import pickle
import pandas as pd
import multiprocessing as mp
from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
# %%
'''This file creates a DBSCAN clustering model for time series data using the sktime library.

The hyperparameters eps and min_samples have already been tuned and can be adjusted to optimize clustering performance.

This DBSCAN implementation uses DTW as the distance metric, which is suitable for time series data.

This model was trained on an entire dataset, and will be used to generate ground truth labels, which will be used to train a neural network.
'''
# %%
vessel_type = 'tankers_and_cargo'
data_dir = f'/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/{vessel_type}/dbscan_data/'
phases = ['train', 'test', 'valid']
# phases =['test']

# First pass: collect all lengths to calculate 95th percentile
all_lengths = []
arrays_to_stack = []
phase_lengths = []

for phase in phases:
    with open(os.path.join(data_dir, f'us_continent_2024_dbscan_{phase}_track.pkl'), 'rb') as f:
        data = pickle.load(f)
    phase_lengths.append((phase, len(data)))
    
    # Get the trajectories from each phase and take first 4 features
    for idx in range(len(data)):

        phase_array = data[idx]['traj'][:, :4]
        arrays_to_stack.append(phase_array)
        all_lengths.append(phase_array.shape[0])
        # print(f"{phase} original shape: {phase_array.shape}")

# Calculate 95th percentile as target length
target_length = int(np.percentile(all_lengths, 95))
# print(f"\nLengths: {all_lengths}")
print(f"95th percentile target length: {target_length}")

# Process each array to have exactly the target length
processed_arrays = []
for i, arr in enumerate(arrays_to_stack):
    current_length = arr.shape[0]
    
    if current_length > target_length:
        # Truncate to target length
        processed = arr[:target_length, :]
        # print(f"{phases[i]} truncated from {current_length} to {target_length}")
    elif current_length < target_length:
        # Pad with zeros to reach target length
        padding_needed = target_length - current_length
        padding = np.zeros((padding_needed, arr.shape[1]))
        processed = np.vstack([arr, padding])
        # print(f"{phases[i]} padded from {current_length} to {target_length}")
    else:
        # Already exactly target length
        processed = arr
        # print(f"{phases[i]} already {target_length} length")
    
    processed_arrays.append(processed)
    # print(f"{phases[i]} final shape: {processed.shape}")

# Stack all arrays to get shape (3, target_length, 4)
stacked_array = np.stack(processed_arrays, axis=0)
print(f"\nStacked array shape: {stacked_array.shape}")
# stacked_array.shape
# %%
def cluster(X: np.ndarray, eps: float, min_samples: int, distance_metric: str = 'dtw') -> dict:
    '''
    Perform DBSCAN clustering on time series data and return clustering results and metrics.
    '''
    print(f"Initializing TimeSeriesDBSCAN with {mp.cpu_count()} CPU cores...")
    dbscan = TimeSeriesDBSCAN(eps=eps, min_samples=min_samples, distance=distance_metric, n_jobs=-1)
    print(f"Fitting DBSCAN model on {X.shape[0]} samples with {distance_metric.upper()} distance...")
    dbscan.fit(X)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_samples = X.shape[0]
        
    return {
        'eps': eps,
        'min_samples': min_samples,
        'distance': distance_metric,
        'n_clusters': n_clusters,
        'labels': labels,
        'n_samples': n_samples,
        'n_noise': n_noise,
        }

def evaluate_clustering(X, labels, metric_name=""):
    """
    Comprehensive evaluation of clustering results for time series data
    """
    print(f"\n{metric_name} Clustering Evaluation:")
    print("-" * 40)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_samples = len(labels)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/n_samples*100:.1f}%)")
    print(f"Number of clustered points: {n_samples - n_noise} ({(n_samples-n_noise)/n_samples*100:.1f}%)")
    
    if n_clusters > 1 and n_noise < n_samples:
        # Only calculate metrics if we have meaningful clusters
        X_flat = X.reshape(X.shape[0], -1)
        
        # Remove noise points for internal metrics
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            X_clean = X_flat[non_noise_mask]
            labels_clean = labels[non_noise_mask]
            
            if len(set(labels_clean)) > 1:
                # Calinski-Harabasz Index (higher is better)
                ch_score = calinski_harabasz_score(X_clean, labels_clean)
                print(f"Calinski-Harabasz Index: {ch_score:.3f} (higher = better)")
                
                # Davies-Bouldin Index (lower is better)
                db_score = davies_bouldin_score(X_clean, labels_clean)
                print(f"Davies-Bouldin Index: {db_score:.3f} (lower = better)")
                
                # Silhouette Score (for comparison)
                sil_score = silhouette_score(X_clean, labels_clean)
                print(f"Silhouette Score: {sil_score:.3f} (higher = better)")
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique_labels, counts))}")
        
        # Assess cluster balance
        if len(counts) > 0:
            size_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            print(f"Cluster size ratio (max/min): {size_ratio:.2f} (closer to 1 = more balanced)")
    
    else:
        print("Cannot calculate internal metrics: insufficient clusters or all points are noise")
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise/n_samples,
        'n_samples': n_samples,
        'Calinski-Harabasz': ch_score if n_clusters > 1 and n_noise < n_samples else None,
        'Davies-Bouldin': db_score if n_clusters > 1 and n_noise < n_samples else None,
        'size_ratio': size_ratio
    }
# %%
# perform clustering with tuned hyperparameters
eps = 21.5
min_samples = 25
distance_metric = 'dtw'

X = stacked_array
# np.random.seed(42)
# sample_size = min(500, X.shape[0])  # Use max 500 samples for speed
# indices = np.random.choice(X.shape[0], sample_size, replace=False)
# X_sample = X[indices]
# print(f"Clustering on test array of shape: {X_sample.shape}")

# Display CPU information for clustering
total_cpus = mp.cpu_count()
print(f"\n{'='*60}")
print(f"CLUSTERING CONFIGURATION")
print(f"{'='*60}")
print(f"Dataset shape: {X.shape}")
print(f"Available CPUs: {total_cpus}")
print(f"CPUs used for clustering: {total_cpus} (n_jobs=-1)")
print(f"Distance metric: {distance_metric}")
print(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")
print(f"{'='*60}")
print("Starting clustering...")

clustering_results = cluster(X, eps=eps, min_samples=min_samples, distance_metric=distance_metric)
print("Clustering completed!")
evaluation = evaluate_clustering(X, clustering_results['labels'], metric_name="DTW")

label_save_dir = f'/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/{vessel_type}/dbscan_data/labels/'
if not os.path.exists(label_save_dir):
    os.makedirs(label_save_dir)

print(f"Clustering results: {clustering_results}")
print(f"Evaluation metrics: {evaluation}")
# Save clustering labels
with open(os.path.join(label_save_dir, f'us_continent_2024_dbscan_labels_eps{eps}_min{min_samples}.pkl'), 'wb') as f:
    pickle.dump(clustering_results['labels'], f)
print(f"Clustering labels saved to {label_save_dir}")
# save clustering and evaluation results to json (excluding numpy arrays)
# Create a clean results dictionary with only scalar values and metadata
json_results = {
    'parameters': {
        'eps': float(eps),
        'min_samples': int(min_samples),
        'distance_metric': distance_metric
    },
    'clustering_metrics': {
        'n_clusters': int(clustering_results['n_clusters']),
        'n_samples': int(clustering_results['n_samples']),
        'n_noise': int(clustering_results['n_noise']),
        'noise_ratio': float(evaluation['noise_ratio']),
        'num_clustered_points': int(evaluation['n_samples'] - evaluation['n_noise'])
    },
    'quality_metrics': {
        'Calinski_Harabasz_Index': float(evaluation['Calinski-Harabasz']) if evaluation['Calinski-Harabasz'] is not None else None,
        'Davies_Bouldin_Index': float(evaluation['Davies-Bouldin']) if evaluation['Davies-Bouldin'] is not None else None,
        'size_ratio': float(evaluation['size_ratio']) if evaluation['size_ratio'] != float('inf') else None
    }
}

# Save to JSON file (without numpy arrays)
json_filename = f'us_continent_2024_dbscan_results_eps{eps}_min{min_samples}.json'
with open(os.path.join(label_save_dir, json_filename), 'w') as f:
    json.dump(json_results, f, indent=4)

print(f"Clustering results (metadata only) saved to JSON: {json_filename}")
print("Note: Full labels array is saved separately as .pkl file for efficient loading")

# %%
