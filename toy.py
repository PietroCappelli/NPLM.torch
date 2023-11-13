from plot_utils import plot_ref_data, plot_ref_data_reco, plot_loss_history
from analysis_utils import compute_df, produce_bins, save_binning, load_binning, compute_seed
from nn_utils import NPLMnetwork, loss_function

from argparse import ArgumentParser
import json, datetime
import torch

import numpy as np


def load_config(path: str):
    """
    Load configuration data from a JSON file.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The configuration data.
    """
    with open(path, "r") as jsonfile:
        config_json = json.load(jsonfile)

    return config_json


def generate_data(config_json: dict):
    """Generate synthetic data for training and testing a machine learning model.

    Args:
        config_json (dict): A dictionary containing the configuration parameters for generating the data.

    Returns:
        tuple: A tuple containing the feature tensor and target tensor.
            The feature tensor has shape (N_samples, N_features) and contains the input features for each sample.
            The target tensor has shape (N_samples, 2) and contains the target labels and weights for each sample.
    """
    
    # poisson fluctuate the number of events in each sample
    N_bkg_p = int(torch.distributions.Poisson(rate=config_json["N_Bkg"]).sample())
    N_sig_p = int(torch.distributions.Poisson(rate=config_json["N_Sig"]).sample())
    
    # the reference rate will not have nuisance parameters
    feature_ref_dist = torch.distributions.Exponential(rate=1)

    # the data rate will have nuisance parameters   
    feature_bkg_dist = torch.distributions.Exponential(rate=1)
    feature_sig_dist = torch.distributions.Normal(loc=config_json["SIG_LOC"], scale=config_json["SIG_STD"])
    
    # generate the features
    feature_ref  = feature_ref_dist.sample((config_json["N_Ref"], 1))
    feature_data = torch.cat(
        (
            feature_bkg_dist.sample((N_bkg_p, 1)),
            feature_sig_dist.sample((N_sig_p, 1))
        )
    )

    # concatenate the features
    feature = torch.cat((feature_ref, feature_data), dim=0)

    # generate the target
    target_ref  = torch.zeros((config_json["N_Ref"], 1))
    target_data = torch.ones((N_bkg_p + N_sig_p, 1))

    target = torch.cat((target_ref, target_data), dim=0)

    # generate the weights
    weights_ref  = torch.ones((config_json["N_Ref"], 1)) * (config_json["N_Bkg"] / config_json["N_Ref"])
    weights_data = torch.ones((N_bkg_p + N_sig_p, 1))

    weights = torch.cat((weights_ref, weights_data), dim=0)

    # concatenate the weights to the target
    target = torch.cat((target, weights), dim=1)
    
    return feature, target


def initialize_nplm(
    architecture      : list, 
    activation_func   : callable, 
    weight_clip_value : float, 
    trainable         : bool  = True, 
    learning_rate     : float = 0.001
    ):
    """
    Initializes an NPLM network with the given architecture, activation function, weight clip value, device, 
    and whether the network is trainable or not. Also initializes an optimizer with the given learning rate.
    
    Args:
    - architecture (list): A list of integers representing the number of neurons in each layer of the network.
    - activation_func (callable): The activation function to use in the network.
    - weight_clip_value (float): The maximum absolute value of the weights in the network.
    - trainable (bool, optional): Whether the network is trainable or not. Defaults to True.
    - learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 0.001.
    
    Returns:
    - nplm_model (NPLMnetwork): The initialized NPLM network.
    - optimizer (torch.optim.Adam): The initialized optimizer.
    """
    
    # NPLM network
    nplm_model = NPLMnetwork(
        architecture      = architecture,
        activation_func   = activation_func,
        weight_clip_value = weight_clip_value,
        trainable         = trainable
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(nplm_model.parameters(), lr=learning_rate)
    
    return nplm_model, optimizer


def log_results(obj, path, name):
    torch.save(obj, path + name + ".pth")
    


def main(args, seed, device):
    """
    Main function for running the NPLM network.

    Args:
        args (argparse.Namespace): Command line arguments.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device to run the network on.

    Returns:
        None
    """
    
    # Load the configuration file
    config_json = load_config(args.jsonfile)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Statistics                                                                                                                                                   
    # N_REF      = config_json["N_Ref"]
    # N_BKG      = config_json["N_Bkg"]
    # N_SIG      = config_json["N_Sig"]       
    # SIG_LOC    = config_json["SIG_LOC"]
    # SIG_STD    = config_json["SIG_STD"]
    
    # Samples weights N_D/N_R
    # N_R        = N_REF
    # N_D        = N_BKG

    # Training parameters
    N_EPOCHS   = config_json["epochs"]
    PATIENCE   = config_json["patience"]

    # Network parameters
    ARCHITECTURE = config_json["architecture"]
    N_INPUTS     = ARCHITECTURE[0]
    WCLIP        = config_json["weight_clipping"]
    ACTIVATION   = torch.nn.Sigmoid()
    DF           = compute_df(ARCHITECTURE)
    
    # Paths
    INPUT_PATH     = config_json["input_directory"]
    OUTPUT_PATH    = config_json["output_directory"]

    
    feature, target = generate_data(config_json)
    
    nplm_model, optimizer = initialize_nplm(
        architecture      = ARCHITECTURE,
        activation_func   = ACTIVATION,
        weight_clip_value = WCLIP,
        device            = device,
        trainable         = True,
        learning_rate     = 0.001
    )
    
    ## GPU PUSHING ##
    # Push the data to the device
    feature = feature.to(device)
    target  = target.to(device)

    # Push the model to the device
    nplm_model.to(device)
    #################
    
    ## TRAINING ##
    
    losses = []

    # Loop over the epochs
    for epoch in range(1, N_EPOCHS + 1):
        
        # Zero the gradients
        optimizer.zero_grad() 
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        pred = nplm_model(feature)
        
        # Calculate the loss
        loss = loss_function(target, pred)
        losses += [loss.item()]
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        optimizer.step()
        
        # Clip the weights
        nplm_model.clip_weights()
        
        # Callback every PATIENCE epochs
        if epoch % PATIENCE == 0:
            print(f"Epoch {epoch}/{N_EPOCHS} - Loss: {loss:.6f} - t: {-2*loss:.6f}")
            
    ##############

    # After training, save the final model and the loss history
    log_results(nplm_model.state_dict(), OUTPUT_PATH, str(seed)+"_nplm_weights")
    log_results(losses, OUTPUT_PATH, str(seed)+"_losses")
    
    


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonfile", type=str, help="json file", required=True)
    args = parser.parse_args()
    
    seed = compute_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args, seed, device)