#%%
import os
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sktime.classification.deep_learning import GRUFCNNClassifier
#%%
'''This file is intended to assign the cluster labels otained from dbscan.py and train a classifer to predict the cluster labels for new vessel trajectories

It uses the DBSCAN clustering results saved in a pickle file and trains a classifier using the clustered data.

Please be sure to run dbscan.py first to generate the required labels.'''
#%%
LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO  = list(range(11))
#%%
# Load the clustered data labels
label_file = '/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/tankers_and_cargo/dbscan_data/labels/us_continent_2024_dbscan_labels_eps18.5_min580.pkl' # <- CHANGE THIS
with open(label_file, 'rb') as f:
    dbscan_labels = pickle.load(f)
#%%
# Assign labels to each phase of the dataset and save the labeled data
vessel_type = 'tankers_and_cargo'
data_dir = f'/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/{vessel_type}/dbscan_data'
phases = ['train', 'test', 'valid']
# phases = ['train']
data_lengths = {'train': 10_000, 'test': 800, 'valid': 800}

for phase in phases:
    if phase=='train':
        labels_array = np.array(dbscan_labels[:data_lengths['train']])
    elif phase=='test':
        labels_array = np.array(dbscan_labels[10_000:10_800])
    else:
        labels_array = np.array(dbscan_labels[10_800:11_600])

    phase_file = f'us_continent_2024_dbscan_{phase}_track.pkl'
    phase_file_path = os.path.join(data_dir, phase_file)
    print(f'Processing {phase} data from {phase_file_path}...')
    
    with open(phase_file_path, 'rb') as f:
        data = pickle.load(f)
    for idx in range(len(data)):
        data[idx]['dbscan_label'] = labels_array[idx]
    # Save the updated data with labels
    output_file = f'us_continent_2024_dbscan_{phase}_track_labeled.pkl'
    output_file_path = os.path.join(data_dir, 'labels', output_file)
    with open(output_file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f'Labeled {len(data)} entries for phase {phase}.')
    print(f'Saved labeled data to {output_file_path}')
#%%
def load_pickled_data(phase: str) -> list[dict]:
    """Load pickled data from the specified file path."""
    file_path = f'/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/{vessel_type}/dbscan_data/labels/us_continent_2024_dbscan_{phase}_track_labeled.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def pad_zeros(series: pd.Series, target_length: int) -> np.ndarray:
    """Pad a pandas Series with zeros to reach the target length."""
    current_length = len(series)
    if current_length >= target_length:
        return series.values[:target_length]
    else:
        padding = np.zeros(target_length - current_length)
        return np.concatenate([series, padding])

def create_dataframe(data: list[dict]) -> pd.DataFrame:
    '''Create a pandas DataFrame from the list of trajectory data dictionaries.'''
    #df_cols = ['mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog', 'vessel_type', 'dbscan_label']
    lat = []
    lon = []
    sog = [] 
    cog = []
    timestamp = []
    mmsi = []
    vessel_type = []
    dbscan_label = []
    for entry in data:
        traj = entry['traj']
        lat.append(pd.Series(traj[:, LAT]))
        lon.append(pd.Series(traj[:, LON]))
        sog.append(pd.Series(traj[:, SOG]))
        cog.append(pd.Series(traj[:, COG]))
        timestamp.append(pd.Series(traj[:, TIMESTAMP]))
        mmsi.append(pd.Series(traj[:, MMSI]))
        vessel_type.append(traj[0,SHIPTYPE])
        dbscan_label.append(entry['dbscan_label'].item())
    df = pd.DataFrame({
        'mmsi': mmsi,
        'timestamp': timestamp,
        'lat': lat,
        'lon': lon,
        'sog': sog,
        'cog': cog,
        'vessel_type': vessel_type,
        'dbscan_label': dbscan_label
    })
    return df

#%%
train_data = load_pickled_data('train')
test_data = load_pickled_data('test')
valid_data = load_pickled_data('valid')

train_df = create_dataframe(train_data)
test_df = create_dataframe(test_data)
valid_df = create_dataframe(valid_data)

combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
#%%
features = ['lat', 'lon', 'sog', 'cog']
X = combined_df[features]
y = combined_df['dbscan_label']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

#%%
# make arrays equal length by padding with zeros
target_length = max(X_train['lat'].apply(len).max(), X_test['lat'].apply(len).max())
print(f'Target length for padding: {target_length}')
for col in features:
    X_train[col] = X_train[col].apply(lambda x: pad_zeros(x, target_length))
    X_test[col] = X_test[col].apply(lambda x: pad_zeros(x, target_length))

# convert to numpy 3D arrays for classification
def df_to_numpy(df: pd.DataFrame, n_timesteps: int = target_length) -> np.ndarray:
    '''convert a Dataframe to a 3d numpy array for classification
    of shape (n_samples, timesteps, num_features)'''
    n_samples = len(df)
    n_features = len(df.columns)
    arr = np.zeros((n_samples, n_timesteps, n_features))

    #fill in the array
    for i in range(n_samples):
        for j, col in enumerate(df.columns):
            arr[i, :, j] = df.iloc[i][col]
    return arr

# prep data for training 
X_train_arr = df_to_numpy(X_train)
X_test_arr = df_to_numpy(X_test)
print(f"===Training data=== Features: {X_train_arr.shape} Labels: {y_train.shape}" )
print(f"===Testing data=== Features: {X_test_arr.shape} Labels: {y_test.shape}" )
#%%
# Train the GRU-FCNN Classifier
gru_fcnn = GRUFCNNClassifier(
    hidden_dim=144, 
    gru_layers=2, 
    conv_layers=[128, 256, 128],  # Correct parameter name
    gru_dropout=0.2,
    dropout=0.2,
    num_epochs=10 ,  # Correct parameter name
    batch_size=64, 
    random_state=42,
    verbose=True  # To see training progress
)

gru_fcnn.fit(X_train_arr, y_train)
# Make predictions
print("Making predictions...")
y_pred = gru_fcnn.predict(X_test_arr)
X_test['predicted_label'] = y_pred
# Evaluate results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show some additional metrics
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#%%
# Save the trained model
model_path = '/home/chucey/GQP/TrAISformer/data/US_data/cleaned_data/tankers_and_cargo/dbscan_data/labels/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = f'gru_fcnn_dbscan_model.pkl'
with open(os.path.join(model_path, model_file), 'wb') as f:
    pickle.dump(gru_fcnn, f)