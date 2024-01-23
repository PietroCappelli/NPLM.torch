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
        vec  = feature[:, j]
        mean = mean_ref[j]
        std  = std_ref[j]
        
        if np.min(vec) < 0:
            vec = (vec - mean) / std
        elif np.max(vec) > 1.0:
            vec = vec / mean
            
        feature[:, j] = vec
    return feature


def calculate_statistics(data):
    """
    Calculates mean and standard deviation of the given data.

    Args:
        data (numpy.ndarray): Data for which statistics are to be calculated.

    Returns:
        tuple: Mean and standard deviation of the data.
    """
    mean_data = np.mean(data, axis=0)
    std_data  = np.std(data, axis=0)
    
    # if mean or std is dimensionless, reshape to (1, 1)
    if mean_data.ndim == 0:
        mean_data = mean_data.reshape(1, 1)
    
    return mean_data, std_data


def read_nu_variation(file_path, nu):
    """
    Read the nuisance parameter variation hypothesis.
    """
    data = read_data_with_nu(data_path=file_path, nu=nu)
    w    = np.ones(data.shape[0])
    idx  = np.arange(data.shape[0])
    return data, w


def luminosity_scale(ref, data, w_ref, w_data, nr0):
    """
    Scale datasets to match the experiment luminosity.
    """
    
    idx_ref = np.arange(ref.shape[0])
    idx_data = np.arange(data.shape[0])

    np.random.shuffle(idx_ref)
    np.random.shuffle(idx_data)
    
    mask_ref  = idx_ref < nr0
    mask_data = idx_data < nr0
    
    # Scale the reference to the luminosity of the experiment
    ref    = ref[mask_ref]
    w_ref  = w_ref[mask_ref]
    
    # Scale the data to the luminosity of the experiment
    data   = data[mask_data]
    w_data = w_data[mask_data]
    
    return ref, data, w_ref, w_data
    
    
def create_analysis_objects(ref, data, w_ref, w_data, nu_std):
    """Combine the reference and alternative samples into one dataset.

    Args:
        ref (np.array): Reference sample.
        data (np.array): Alternative sample.
        w_ref (np.array): Reference weights.
        w_data (np.array): Alternative weights.
        nu_std (float): Standard deviation of the nuisance parameter.

    Returns:
        np.array: Combined sample.
        np.array: Combined target.
        np.array: Combined weights.
        np.array: Combined nuisance parameter.
    """
    
    # Combine the samples
    combined = np.concatenate(
        (ref, data), 
        axis=0
    )
    
    # Create combined target
    target_combined = np.concatenate(
        (np.zeros(ref.shape[0]), np.ones(data.shape[0])), 
        axis=0
    )
    
    # Combine the weights
    w_combined = np.concatenate(
        (w_ref, w_data),
        axis=0
    )
    
    # Combine the nuisance parameter
    nu_combined = np.ones(combined.shape[0]) * nu_std
    
    
    return combined, target_combined, w_combined, nu_combined    


def combine_analysis_objects(feature, targets, weights, nuisance, new_samples):
    """
    Combines existing samples with new samples.

    Args:
        feature (numpy.ndarray): Existing features.
        targets (numpy.ndarray): Existing targets.
        weights (numpy.ndarray): Existing weights.
        nuisance (numpy.ndarray): Existing nuisance parameters.
        new_samples (tuple): New samples to combine with existing ones.

    Returns:
        tuple: Combined features, targets, weights, and nuisance parameters.
    """
    new_feature, new_targets, new_weights, new_nuisance = new_samples

    if feature.shape[0] == 0:
        return new_feature, new_targets, new_weights, new_nuisance
    else:
        feature  = np.concatenate((feature,  new_feature),  axis=0)
        targets  = np.concatenate((targets,  new_targets),  axis=0)
        weights  = np.concatenate((weights,  new_weights),  axis=0)
        nuisance = np.concatenate((nuisance, new_nuisance), axis=0)
        return feature, targets, weights, nuisance