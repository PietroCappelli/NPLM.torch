import os 
# import scipy.stats as stats
import h5py
import numpy as np

def read_data_with_nu(data_path: str, nu: float, version: str = None) -> np.ndarray:
    """
    Read data from file .h5 
    """
    
    fname = f"data_1dexpon_ref_nus_{nu}sigmas.h5"
    
    if version is not None:
        fname = f"data_1dexpon_ref_nus_{nu}sigmas_v{version}.h5"
    
    data_file= os.path.join(data_path, fname)
    with h5py.File(data_file, "r") as f:
        data = f["data"][:]
    
    return data


# Function to normalize data
def normalize_data(feature, mean_ref, std_ref):
    for j in range(feature.shape[1]):
        vec = feature[:, j]
        mean = mean_ref[j]
        std = std_ref[j]
        
        if np.min(vec) < 0:
            vec = (vec - mean) / std
        elif np.max(vec) > 1.0:
            vec = vec / mean
            
        feature[:, j] = vec
    return feature