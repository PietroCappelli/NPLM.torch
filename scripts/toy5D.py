import os
import sys
sys.path.insert(0, "../src")

from plot_utils import plot_ref_data, plot_ref_data_reco, plot_loss_history
from analysis_utils import compute_df, produce_bins, save_binning, load_binning
from nn_utils import NPLMnetwork, loss_function, Dataset5D

from argparse import ArgumentParser
from time import time
import json, datetime
import torch
import h5py
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

def load_dataset(DATA_PATH : str, OUTPUT_PATH : str,  NP = 'EFT'):
    
    Data_dir  = [DATA_PATH+file+'/' for file in os.listdir(DATA_PATH)]
    DATA = [row+item for row in Data_dir for item in os.listdir(row)]
    DATA = [file for file in DATA if not "/._" in file]

    REF_file  = [file for file in DATA if "DiLepton_SM" in file ]
    DATA_file = [file for file in DATA if not "DiLepton_SM" in file]

    DATA_file_Zprime = [file for file in DATA if "DiLepton_Zprime" in file]
    DATA_file_EFT    = [file for file in DATA if "DiLepton_EFT" in file]

    '''
    LOOK IF THERE ARE SOME CORRUPTED FILES
    '''
    ref_dataset = Dataset5D(REF_file)
    ref_list  = []
    data_list = []
    corr_ref = None
    corr_data = None
    
    for i in range(len(ref_dataset)):
        try:
            ref_list.append(ref_dataset[i])
        except Exception as e:
            print("corrupted reference file", REF_file[i])
            corr_ref = i

    if NP=="EFT":
        np_dataset = Dataset5D(DATA_file_EFT, normalize=True)
        for i in range(len(np_dataset)):
            try:
                data_list.append(np_dataset[i])
            except Exception as e:
                print("corrupted reference file", DATA_file_EFT[i])
                corr_data = i

    elif NP=="Z":
        np_dataset = Dataset5D(DATA_file_Zprime, normalize=True)
        for i in range(len(np_dataset)):
            try:
                data_list.append(np_dataset[i])
            except Exception as e:
                print("corrupted data file", DATA_file_Zprime[i])
                corr_data = i
    else:
        raise Exception("No valid NewPhysics dataset (EFT or Z)")  
      
    del data_list, ref_list
    
    REF_FEAT  = torch.cat([ref_dataset[i] for i in range(len(ref_dataset)) if i!= corr_ref], dim=0)
    DATA_FEAT = torch.cat([np_dataset[i] for i in range(len(np_dataset)) if i!= corr_data ], dim=0)

    # if corr_data:
    #     DATA_FEAT = torch.cat([np_dataset[i] for i in range(len(np_dataset)) if i!= corr_data ], dim=0)
    # else: 
    #     DATA_FEAT = torch.cat([np_dataset[i] for i in range(len(np_dataset))], dim=0)

    torch.save({'REF_features':REF_FEAT, "DATA_features": DATA_FEAT}, OUTPUT_PATH)



def generate_data(config_json: dict):
    """Generate synthetic data for training and testing a machine learning model.

    Args:
        config_json (dict): A dictionary containing the configuration parameters for generating the data.
        ref_dataset (torch.tensor): tensor containing all the ref data features imported thanks to load_dataset function
        ref_dataset (torch.tensor): tensor containing all the np  data features imported thanks to load_dataset function

    Returns:
        tuple: A tuple containing the feature tensor and target tensor.
            The feature tensor has shape (N_samples, N_features) and contains the input features for each sample.
            The target tensor has shape (N_samples, 2) and contains the target labels and weights for each sample.
    """
    loaded_data = torch.load(config_json["TENSOR_PATH"])
    
    REF_FEAT  = loaded_data['REF_features']
    DATA_FEAT = loaded_data['DATA_features']
    
    N_EVENTS_R = REF_FEAT.shape[0]
    N_EVENTS_D = DATA_FEAT.shape[0]
    
    # poisson fluctuate the number of events in each sample
    N_bkg_p = int(torch.distributions.Poisson(rate=config_json["N_Bkg"]).sample())
    N_sig_p = int(torch.distributions.Poisson(rate=config_json["N_Sig"]).sample())
    
    # EXTRACT RANDOM EVENTS FROM THE INPUT DATASET
    # indeces for the Reference and the BKG dataset
    idxs    = torch.multinomial(torch.ones(N_EVENTS_R), num_samples=(config_json["N_Ref"] + N_bkg_p), replacement=False)
    idx_R   = idxs[:config_json["N_Ref"]]
    idx_bkg = idxs[config_json["N_Ref"]:]
    
    # REF and BACKGROUND
    feature_ref = REF_FEAT[idx_R]
    feature_bkg = REF_FEAT[idx_bkg]
    
    # SIGNAL
    if (config_json["N_Sig"]!=0):
        idx_data = torch.multinomial(torch.ones(DATA_FEAT.shape[0]), num_samples=N_sig_p, replacement=False)
        feature_sig = DATA_FEAT[idx_data]
    elif(config_json["N_Sig"]==0):
        feature_sig = torch.empty((0,5))
    
    feature_data = torch.cat((feature_sig, feature_bkg))

    # set target: 0 for ref and 1 for data 
    target_ref = torch.zeros((config_json["N_Ref"],1), dtype=torch.float32) 
    target_data = torch.ones((N_bkg_p + N_sig_p,1), dtype=torch.float32)

    # initialize weights as ones * N_D / N_R
    weights_ref  = torch.ones((config_json["N_Ref"], 1), dtype=torch.float32)   * (config_json["N_Bkg"] / config_json["N_Ref"])
    weights_data = torch.ones((N_sig_p + N_bkg_p, 1), dtype=torch.float32) 

    feature = torch.cat((feature_ref, feature_data), dim=0)
    target  = torch.cat((target_ref,  target_data),  dim=0)
    weights = torch.cat((weights_ref, weights_data), dim=0)

    # concatenate the weights to the target
    target = torch.cat((target, weights), dim=1)
    # SET to the same type
    target  = target
    feature = feature.type(torch.float32)

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
    


def main(args, device):
    """
    Main function for running the NPLM network.

    Args:
        args (argparse.Namespace): Command line arguments.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device to run the network on.

    Returns:
        None
    """
    args.debug = False
    
    toy_start = time()
    
    
    if args.debug:
        print("DEBUGGGING!!")
        toy_start = time()
    
    # Load the configuration file
    config_json = load_config(args.jsonfile)
    
    # datetime of launch
    date = config_json["date"]
    date = date.replace("_", "")
    
    # Set seed
    random_seed = int(date[-6:]) + args.index
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
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

    
    if args.debug:
        toy_data_gen_start = time()
        
        
    '''
        IMPORT OR GENERATE DATA
    '''
    
    feature, target = generate_data(config_json)
    
    if args.debug:
        toy_data_gen_end = time()
        toy_data_gen_time = toy_data_gen_end - toy_data_gen_start
        print(f"Data generation time: {toy_data_gen_time:.5f} s")
        
    
    if args.debug:
        toy_nplm_init_start = time()
    
    nplm_model, optimizer = initialize_nplm(
        architecture      = ARCHITECTURE,
        activation_func   = ACTIVATION,
        weight_clip_value = WCLIP,
        trainable         = True,
        learning_rate     = 0.001
    )
    
    if args.debug:
        toy_nplm_init_end = time()
        toy_nplm_init_time = toy_nplm_init_end - toy_nplm_init_start
        print(f"NPLM initialization time: {toy_nplm_init_time:.5f} s")

        
    
    if args.debug:
        toy_push_start = time()
        
    # Push the data to the device
    feature = feature.to(device)
    target  = target.to(device)

    # Push the model to the device
    nplm_model.to(device)

    if args.debug:
        toy_push_end = time()
        toy_push_time = toy_push_end - toy_push_start
        print(f"Pushing to device time: {toy_push_time:.5f} s")
        
        
    
    ## TRAINING ##
    
    losses = []
    
    if args.debug:
        toy_epoch_times = []
        toy_zero_grad_times = []
        toy_forward_times = []
        toy_loss_times = []
        toy_backward_times = []
        toy_step_times = []
        toy_clip_times = []

    if args.debug:
        toy_training_start = time()
    
    toy_training_start = time()

    # Loop over the epochs
    for epoch in range(1, N_EPOCHS + 1):
        
        if args.debug:
            toy_epoch_start = time()
        
        if args.debug:
            toy_zero_grad_start = time()
            
        # Zero the gradients
        optimizer.zero_grad() 
        
        if args.debug:
            toy_zero_grad_end = time()
            toy_zero_grad_time = toy_zero_grad_end - toy_zero_grad_start
            if epoch % PATIENCE == 0:
                print(f"Zero grad time: {toy_zero_grad_time:.5f} s")
            
        if args.debug:
            toy_forward_start = time()
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        pred = nplm_model(feature)
        
        if args.debug: 
            toy_forward_end = time()
            toy_forward_time = toy_forward_end - toy_forward_start
            if epoch % PATIENCE == 0:
                print(f"Forward pass time: {toy_forward_time:.5f} s")
        
        if args.debug:
            toy_loss_start = time()
            
        # Calculate the loss
        loss = loss_function(target, pred)
        
        if args.debug:
            toy_loss_end = time()
            toy_loss_time = toy_loss_end - toy_loss_start
            if epoch % PATIENCE == 0:
                print(f"Loss time: {toy_loss_time:.5f} s")
            
        losses += [loss.item()]
        
        if args.debug:
            toy_backward_start = time()
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        if args.debug:
            toy_backward_end = time()
            toy_backward_time = toy_backward_end - toy_backward_start
            if epoch % PATIENCE == 0:
                print(f"Backward pass time: {toy_backward_time:.5f} s")
        
        if args.debug:
            toy_step_start = time()
        
        # Perform a single optimization step (parameter update)
        optimizer.step()
        
        if args.debug:
            toy_step_end = time()
            toy_step_time = toy_step_end - toy_step_start
            if epoch % PATIENCE == 0:
                print(f"Step time: {toy_step_time:.5f} s")
            
        if args.debug:
            toy_clip_start = time()
        
        # Clip the weights
        nplm_model.clip_weights()
        
        if args.debug:
            toy_clip_end = time()
            toy_clip_time = toy_clip_end - toy_clip_start
            if epoch % PATIENCE == 0:
                print(f"Clip time: {toy_clip_time:.5f} s")
        
        if args.debug:
            toy_epoch_end = time()
            toy_epoch_time = toy_epoch_end - toy_epoch_start
            if epoch % PATIENCE == 0:
                print(f"Epoch time: {toy_epoch_time:.5f} s")
        
        
        # Callback every PATIENCE epochs
        if epoch % PATIENCE == 0:
            print(f"Epoch {epoch}/{N_EPOCHS} - Loss: {loss:.6f} - t: {-2*loss:.6f}")
            
        # save per-epoch timing in lists
        if args.debug:
            toy_epoch_times     += [toy_epoch_time]
            toy_zero_grad_times += [toy_zero_grad_time]
            toy_forward_times   += [toy_forward_time]
            toy_loss_times      += [toy_loss_time]
            toy_backward_times  += [toy_backward_time]
            toy_step_times      += [toy_step_time]
            toy_clip_times      += [toy_clip_time]
            
        
    ##############
    log_results(nplm_model.state_dict(), OUTPUT_PATH, str(args.index)+"_nplm_weights")
    log_results(losses, OUTPUT_PATH, str(args.index)+"_losses")
    print(f"data are saved in {OUTPUT_PATH}\n")
    
    if args.debug:
        toy_training_end = time()
        toy_training_time = toy_training_end - toy_training_start
        print(f"Training time: {toy_training_time:.5f} s")

    toy_training_end = time()
    toy_start = time()
        
        
    if args.debug:
        toy_end = time()
        toy_time = toy_end - toy_start
        print(f"Total toy time: {toy_time:.5f} s")
        
    toy_time = toy_end - toy_start
    toy_training_time = toy_training_end - toy_training_start
    print(f"Total toy time: {toy_time:.5f} s")
    f = open("/home/ubuntu/NPLM.torch/notebooks/tensor/timing.txt", "a")
    f.write("toy: ", toy_time, "\t training: ", toy_training_time, "\n")
    f.close()
        
    # save the final timing
    if args.debug:
        
        with h5py.File(OUTPUT_PATH + "debug.h5", "a") as timing_file:
            timing_file["data_gen"].resize((timing_file["data_gen"].shape[0] + 1,))
            timing_file["data_gen"][-1] = toy_data_gen_time
            
            timing_file["nplm_init"].resize((timing_file["nplm_init"].shape[0] + 1,))
            timing_file["nplm_init"][-1] = toy_nplm_init_time
        
            timing_file["push"].resize((timing_file["push"].shape[0] + 1,))
            timing_file["push"][-1] = toy_push_time
        
            timing_file["training"].resize((timing_file["training"].shape[0] + 1,))
            timing_file["training"][-1] = toy_training_time
        
            timing_file["toy"].resize((timing_file["toy"].shape[0] + 1,))
            timing_file["toy"][-1] = toy_time
            
            timing_file["epoch"].resize((timing_file["epoch"].shape[0] + len(toy_epoch_times),))
            timing_file["epoch"][-len(toy_epoch_times):] = toy_epoch_times
            
            timing_file["zero_grad"].resize((timing_file["zero_grad"].shape[0] + len(toy_zero_grad_times),))
            timing_file["zero_grad"][-len(toy_zero_grad_times):] = toy_zero_grad_times
            
            timing_file["forward"].resize((timing_file["forward"].shape[0] + len(toy_forward_times),))
            timing_file["forward"][-len(toy_forward_times):] = toy_forward_times
            
            timing_file["loss"].resize((timing_file["loss"].shape[0] + len(toy_loss_times),))
            timing_file["loss"][-len(toy_loss_times):] = toy_loss_times
            
            timing_file["backward"].resize((timing_file["backward"].shape[0] + len(toy_backward_times),))
            timing_file["backward"][-len(toy_backward_times):] = toy_backward_times
            
            timing_file["step"].resize((timing_file["step"].shape[0] + len(toy_step_times),))
            timing_file["step"][-len(toy_step_times):] = toy_step_times
            
            timing_file["clip"].resize((timing_file["clip"].shape[0] + len(toy_clip_times),))
            timing_file["clip"][-len(toy_clip_times):] = toy_clip_times
            
        
            
    # After training, save the final model and the loss history
    # log_results(nplm_model.state_dict(), OUTPUT_PATH, str(args.index)+"_nplm_weights")
    # log_results(losses, OUTPUT_PATH, str(args.index)+"_losses")
    # print(f"data are saved in {OUTPUT_PATH}\n")
    


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonfile", type=str,  help="json file", required=True)
    parser.add_argument("-i", "--index",    type=int,  help="index",     required=True)
    parser.add_argument("-d", "--debug",    type=bool, help="debug",     required=True)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nThe device used is: {device}\n')
    # force cpu
    # device = torch.device("cpu")
    
    main(args, device)