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