import os
import sys
sys.path.insert(0, "../src")

# from plot_utils import plot_ref_data, plot_ref_data_reco, plot_loss_history
from analysis_utils import compute_df
from nn_utils import loss_function
from torch.utils.data import Dataset

from argparse import ArgumentParser
import time
import json, datetime
import torch
import h5py
import numpy as np

from NPLM.NNutils import *
from NPLM.PLOTutils import *

class Dataset5D(Dataset):

    # return a tensor of the data contained in a single file

    def __init__(self, file_paths, normalize=True, cut = True):
        self.file_paths = file_paths
        self.normalize = normalize
        self.cut = cut
        
        self.means = []
        self.stds = []

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Select only the 5 features
            pt1  = torch.tensor(f['pt1'][:])
            pt2  = torch.tensor(f['pt2'][:])
            eta1 = torch.tensor(f['eta1'][:])
            eta2 = torch.tensor(f['eta2'][:])
            delta_phi = torch.tensor(f['delta_phi'][:])
            mll = torch.tensor(f['mll'][:])
            
            features = torch.stack([pt1,pt2,eta1,eta2,delta_phi, mll], dim=1)

        return features   
    
def standardize_dataset(feature, mean_REF=[], std_REF=[]):                                                                                                                                
    feature_std = np.copy(feature)
    for j in range(feature.shape[1]):
        vec  = feature_std[:, j]
        if len(mean_REF)==0:
            mean = np.mean(vec)
        else:
            mean = mean_REF[j]
        if len(std_REF)==0:
            std  = np.std(vec)
        else:
            std  = std_REF[j]
        if np.min(vec) < 0:
            vec = vec- mean
            vec = vec *1./ std
        elif np.max(vec) > 1.0:                                                                               
            vec = vec *1./ mean
        feature_std[:, j] = vec
    return feature_std

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
    corr_ref  = None
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
    return REF_FEAT.shape[0], DATA_FEAT.shape[0]

def generate_data(config_json: dict, used_idx_bkg, used_idx_sig):
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
    
    ## Filter the data
    idx_available_bkg = np.setdiff1d(np.arange(N_EVENTS_R), used_idx_bkg ) 
    idx_available_sig = np.setdiff1d(np.arange(N_EVENTS_D), used_idx_sig )

    idxs    = np.random.choice(idx_available_bkg, size=(config_json["N_Ref"] + config_json['N_Bkg']), replace=False)   
    idx_R   = idxs[:config_json["N_Ref"]]
    idx_bkg = idxs[config_json["N_Ref"]:]
    
    # REF and BACKGROUND
    feature_ref = REF_FEAT[idx_R]
    feature_bkg = REF_FEAT[idx_bkg]
    
    # SIGNAL
    if (config_json["N_Sig"]!=0):
        # idx_sig = torch.multinomial(torch.ones(N_EVENTS_D), num_samples=config_json['N_Sig'], replacement=False)
        idx_sig = np.random.choice(idx_available_sig, size=config_json['N_Sig'], replace=False)
        feature_sig = DATA_FEAT[idx_sig]
    elif(config_json["N_Sig"]==0):
        idx_sig=np.array(())
        feature_sig = torch.empty((0,6))
    
    feature_data = torch.cat((feature_sig, feature_bkg))

    ## features cuts
    feature_ref = feature_ref[ feature_ref[:,5] >= 60 ]
    feature_ref = feature_ref[ torch.logical_and(feature_ref[:,0] >= 20, feature_ref[:,1] >=20 ) ]
    feature_ref = feature_ref[ torch.logical_and(abs(feature_ref[:,2]) <= 2.4 , abs(feature_ref[:,3]) <= 2.4 )]

    feature_data = feature_data[ feature_data[:,5] >= 60 ]
    feature_data = feature_data[ torch.logical_and(feature_data[:,0] >= 20, feature_data[:,1] >=20 ) ]
    feature_data = feature_data[ torch.logical_and(abs(feature_data[:,2]) <= 2.4 , abs(feature_data[:,3]) <= 2.4 )]

    ## delete the mll feature
    feature_ref  = feature_ref[:,:5]
    feature_data = feature_data[:,:5]

    ## normalize the features
    standardize_dataset(feature_ref)
    standardize_dataset(feature_data)
    
    # set target: 0 for ref and 1 for data 
    target_ref = torch.zeros((feature_ref.shape[0],1), dtype=torch.float32) 
    target_data = torch.ones((feature_data.shape[0],1), dtype=torch.float32)

    # initialize weights as ones * N_D / N_R
    weights_ref  = torch.ones((feature_ref.shape[0], 1), dtype=torch.float32)   * (config_json["N_Bkg"] / config_json["N_Ref"])
    weights_data = torch.ones((feature_data.shape[0], 1), dtype=torch.float32) 

    feature = torch.cat((feature_ref, feature_data), dim=0)
    target  = torch.cat((target_ref,  target_data),  dim=0)
    weights = torch.cat((weights_ref, weights_data), dim=0)

    # concatenate the weights to the target
    target = torch.cat((target, weights), dim=1)
    # SET to the same type
    feature = feature.type(torch.float32)

    torch.save({'features':feature, "target": target}, config_json['DATASET_PATH'])

    return idx_bkg, idx_sig


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
    
    toy_start = time.time()
    
    
    if args.debug:
        print("DEBUGGGING!!")
        toy_start = time.time()
    
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
    DF           = 96
    # DF           = compute_df(ARCHITECTURE)
    
    
    # Paths
    INPUT_PATH     = config_json["input_directory"]
    OUTPUT_PATH    = config_json["output_directory"]

    
    if args.debug:
        toy_data_gen_start = time.time()
        
        
    '''
        IMPORT OR GENERATE DATA
    '''
    # feature, target = generate_data(config_json)
    
    loaded_data = torch.load(config_json['DATASET_PATH'])
    
    feature = loaded_data['features'].numpy()
    target  = loaded_data['target'].numpy()

    if args.debug:
        toy_data_gen_end = time.time()
        toy_data_gen_time = toy_data_gen_end - toy_data_gen_start
        print(f"Data generation time: {toy_data_gen_time:.5f} s")
        
    
    if args.debug:
        toy_nplm_init_start = time.time()
    
    '''
    DEFINE THE MODEL
    '''
    nplm_model = imperfect_model(input_shape=(None, feature.shape[1]),
                      NU_S=[], NUR_S=[], NU0_S=[], SIGMA_S=None, 
                      NU_N=None, NUR_N=None, NU0_N=None, SIGMA_N=None,
                      correction="", shape_dictionary_list=[],
                      BSMarchitecture=ARCHITECTURE, BSMweight_clipping=WCLIP, train_f=True, train_nu=False)
    print(nplm_model.summary())
    
    if args.debug:
        toy_nplm_init_end = time.time()
        toy_nplm_init_time = toy_nplm_init_end - toy_nplm_init_start
        print(f"NPLM initialization time: {toy_nplm_init_time:.5f} s")

        
    
    if args.debug:
        toy_push_start = time.time()
        
    # # Push the data to the device
    # feature = feature.to(device)
    # target  = target.to(device)

    # # Push the model to the device
    # nplm_model.to(device)

    if args.debug:
        toy_push_end = time.time()
        toy_push_time = toy_push_end - toy_push_start
        print(f"Pushing to device time: {toy_push_time:.5f} s")
        
        
    
    ## TRAINING ##
    
    nplm_model.compile(loss=imperfect_loss, optimizer='adam')
    
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
        toy_training_start = time.time()
    
    toy_training_start = time.time()

    hist_model = nplm_model.fit(feature, target, batch_size=feature.shape[0], epochs=N_EPOCHS, verbose=False)

    # metrics                      
    loss_tau  = np.array(hist_model.history['loss'])

    # test statistic                                         
    final_loss = loss_tau[-1]
    tau_OBS    = -2*final_loss

    # save t                                                                                                               
    log_t = OUTPUT_PATH+str(args.index)+'_ttest.txt'
    out   = open(log_t,'w')
    out.write("%f\n" %(tau_OBS))
    out.close()
    
    # # save the training history                                       
    # log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_history.h5'
    # f           = h5py.File(log_history,"w")
    # epoch       = np.array(range(total_epochs_tau))
    # keepEpoch   = epoch % patience_tau == 0
    # f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
    # for key in list(hist_tau.history.keys()):
    #     monitored = np.array(hist_tau.history[key])
    #     print('%s: %f'%(key, monitored[-1]))
    #     f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
    # f.close()
    
    # save the model    
    log_weights = OUTPUT_PATH+str(args.index)+'_TAU_weights.h5'
    nplm_model.save_weights(log_weights)

    if args.debug:
        toy_training_end = time.time()
        toy_training_time = toy_training_end - toy_training_start
        print(f"Training time: {toy_training_time:.5f} s")

    toy_training_end = time.time()
    toy_end = time.time()
        
        
    if args.debug:
        toy_end = time.time()
        toy_time = toy_end - toy_start
        print(f"Total toy time: {toy_time:.5f} s")
        
    toy_time = toy_end - toy_start
    # toy_training_time = toy_training_end - toy_training_start
    # print(f"Total toy time: {toy_time:.5f} s")
    # print(f"Total toy training time: {toy_training_time:.5f} s")
    f = open("/home/ubuntu/NPLM.torch/notebooks/tensor/timing.txt", "a")
    f.write(f'{toy_time}')
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
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # force cpu
    device = torch.device("cpu")
    print(f'\nThe device used is: {device}\n')
    
    main(args, device)