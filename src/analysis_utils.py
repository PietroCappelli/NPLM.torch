import torch 
import h5py
import os 
import datetime

import scipy.stats as stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep

from hepstats.modeling import bayesian_blocks
from nn_utils import delta_nu_poly


def find_results(directory, name):
    """
    Find files in the given directory that end with the given name.
    """
    results = []
    for file in os.listdir(directory):
        if file.endswith(name):
            results.append(file)
    results.sort(key=lambda x: int(x.split("_")[0]))
    return results


def load_results(directory, losses, weights):
    """
    Load the losses and weights from the given directory.
    """
    losses_data  = []
    weights_data = []
    for i in range(len(losses)):
        losses_data.append(torch.load(directory + losses[i]))
        weights_data.append(torch.load(directory + weights[i]))
    return np.array(losses_data), np.array(weights_data)


def compute_t_from_loss(loss: np.array) -> np.array:
    """
    Compute the test statistic t from the given loss history.
    """
    return -2*loss


def compute_date() -> str:
    """Computes a seed for the random number generator."""
    return str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)+str(datetime.datetime.now().second)+str(datetime.datetime.now().microsecond)


def compute_df(architecture: list) -> int:
    """Computes the number of parameters in a neural network architecture.

    Args:
        architecture (list): A list of integers representing the number of neurons in each layer of the neural network.

    Returns:
        int: The total number of parameters in the neural network.
    """
    df = sum([x*(y+1) for x, y in zip(architecture[1:], architecture[:-1])])
    return df



def bayesian_blocks_binning(data: np.ndarray, weights: np.ndarray = None, p0: float = 0.05):
    """
    Compute the Bayesian blocks binning of the input data.

    Parameters:
    -----------
    data : np.ndarray
        The input data to be binned.
    weights : np.ndarray, optional
        The weights of each data point. If None, all points are assumed to have equal weight.
    p0 : float, optional
        The false alarm probability of the Poisson process used to model the data.

    Returns:
    --------
    edges : np.ndarray
        The edges of the bins computed by the Bayesian blocks algorithm.
    """
    if len(data.shape) > 1:
        data = data.flatten()
    
    return bayesian_blocks(data=data, weights=weights, p0=p0)


def simple_binning(data: np.ndarray, weights: np.ndarray = None, n_bins: int = 100, bins_range: tuple = None, bins_width: float = None):
    """
    Compute the bin edges for simple binning of the input data.

    Args:
        data (np.ndarray): Input data to be binned.
        weights (np.ndarray, optional): Weights for each data point. Defaults to None.
        n_bins (int, optional): Number of bins. Defaults to 100.
        bins_range (tuple, optional): Range of the bins. Defaults to None.
        bins_width (float, optional): Width of each bin. Defaults to None.

    Returns:
        np.ndarray: Bin edges for simple binning.
    """
    if bins_range is None:
        bins_range = (data.min(), data.max())
        
    if bins_width is None:
        bins_width = (bins_range[1] - bins_range[0]) / n_bins
        
    bins = np.arange(bins_range[0], bins_range[1] + bins_width, bins_width)
    
    return bins


def produce_bins(data: np.ndarray, method: str = "simple", **kwargs):
    """
    Produce binning reference for given data using the specified method.

    Parameters:
    data (np.ndarray): The data to be binned.
    method (str): The method to be used for binning. Valid options are "simple" and "bayesian".
    **kwargs: Additional keyword arguments to be passed to the binning function.

    Returns:
    The binning reference produced by the specified method.
    """
    # check if method is valid
    if method not in ["simple", "bayesian"]:
        raise ValueError("method should be either 'simple' or 'bayesian'")

    # if method is simple, return the simple binning reference
    if method == "simple":
        return simple_binning(data=data, **kwargs)

    # if method is bayesian, return the bayesian binning reference
    if method == "bayesian":
        return bayesian_blocks_binning(data=data, **kwargs)


def save_binning(bins: np.ndarray, path: str, name: str):
    """
    Save binning data to an HDF5 file.

    Args:
        bins (np.ndarray): The binning data to save.
        path (str): The path to the directory where the file will be saved.
        name (str): The name of the file to be saved.

    Returns:
        None
    """
    with h5py.File(path + name, "w") as f:
        f.create_dataset("bins", data=bins)
        
        
def load_binning(path: str, name: str):
    """Load binning data from an HDF5 file.

    Args:
        path (str): The path to the directory containing the HDF5 file.
        name (str): The name of the HDF5 file.

    Returns:
        numpy.ndarray: The binning data.
    """
    with h5py.File(path + name, "r") as f:
        bins = f["bins"][:]
    return bins


def process_nuisance_data(feature, target, nu_std):
    """
    Helper function to process the data for a given nuisance parameter.
    """
    x_ref = feature[(target[:, 2] == nu_std) & (target[:, 0] == 0)][:, 0]
    w_ref = target[:, 1][(target[:, 2] == nu_std) & (target[:, 0] == 0)]
    
    x_nu = feature[(target[:, 2] == nu_std) & (target[:, 0] == 1)][:, 0]
    w_nu = target[:, 1][(target[:, 2] == nu_std) & (target[:, 0] == 1)]

    return x_ref, w_ref, x_nu, w_nu


def compute_log_ratio(x_ref, w_ref, x_nu, w_nu, bins):
    """
    Helper function to compute the logarithmic ratio of histograms.
    """
    histo_ref, _ = np.histogram(x_ref, bins=bins, weights=w_ref)
    histo_nu, _ = np.histogram(x_nu, bins=bins, weights=w_nu)

    histo_ref_sq, _ = np.histogram(x_ref, bins=bins, weights=w_ref ** 2)
    histo_nu_sq, _ = np.histogram(x_nu, bins=bins, weights=w_nu ** 2)

    ratio = histo_nu * 1.0 / histo_ref
    ratio_err = ratio * np.sqrt((histo_nu_sq * 1.0 / histo_nu ** 2) + (histo_ref_sq * 1.0 / histo_ref ** 2))

    log_ratio = np.log(ratio)
    log_ratio_err = ratio_err / ratio

    return log_ratio, log_ratio_err



def compute_learned_log_ratio_nu(feature, target, model, nu_std, bins, device):
    """
    Helper function to compute the learned logarithmic ratio using the model.
    """
    maskR = (target[:, -1] == nu_std) & (target[:, 0] == 0)
    maskD = (target[:, -1] == nu_std) & (target[:, 0] == 1)

    featR = feature[maskR]
    featD = feature[maskD]
    
    pred = model(feature.to(device))
        
    poly_test = delta_nu_poly(target.to(device), pred)
    
    if device.type == "cuda":
        poly_test = poly_test.cpu()
        deltaR = np.exp(poly_test[maskR].detach().numpy())
    elif device.type == "cpu":
        deltaR = np.exp(poly_test[maskR])
    
    weig = target[:, 1]
    weigR = weig[maskR]
    weigD = weig[maskD]

    hist_sumD = np.histogram(featD[:, 0], weights=weigD, bins=bins)[0]
    hist_sumW = np.histogram(featR[:, 0], weights=weigR * deltaR, bins=bins)[0]
    hist_sum = np.histogram(featR[:, 0], weights=weigR, bins=bins)[0]

    return np.log(hist_sumW / hist_sum)

