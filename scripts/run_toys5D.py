import os
import json
import argparse
import h5py
import numpy as np

from datetime import datetime
from time import time
from toy5D import load_dataset

INPUT_DIRECTORY  = "./"
OUTPUT_DIRECTORY = "../output/5D/first/"

# configuration dictionary
# config_json = {
#     "N_Ref"             : 200_000,
#     "N_Bkg"             : 2_000,
#     "N_Sig"             : 10,
#     "SIG_LOC"           : 6.40,
#     "SIG_STD"           : 0.16,
#     "output_directory"  : OUTPUT_DIRECTORY,
#     "input_directory"   : INPUT_DIRECTORY,
#     "epochs"            : 200_000,
#     "patience"          : 5_000,
#     "architecture"      : [1, 4, 1],
#     "weight_clipping"   : 9,
# }

def create_config_file(config_table, path, name="config"):
    with open("%s/%s.json"%(path, name), "w") as outfile:
        json.dump(config_table, outfile, indent=4)
    return "%s/%s.json"%(path, name)


def create_debug_files(path: str, name: str):
    """
    Create the debug files for the toy example.

    Args:
        path (str): The path to the output directory.
        name (str): The name of the output file.

    Returns:
        None
    """
    # Create the output directory
    os.makedirs(path, exist_ok=True)

    # Create the debug files
    with h5py.File(path + name + ".h5", "w") as h5file:
        h5file.create_dataset("epoch",     dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("zero_grad", dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("forward",   dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("loss",      dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("backward",  dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("step",      dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("clip",      dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("toy",       dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("training",  dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("data_gen",  dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("nplm_init", dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("push",      dtype=np.float64, shape=(1,), maxshape=(None,))
        h5file.create_dataset("global",    dtype=np.float64, shape=(1,), maxshape=(None,))




def main(args, config_json):
    
    if args.debug:
        # start the global timer
        global_start = time()
        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      + "_"
    current_date += str(datetime.now().microsecond)
    config_json["date"] = current_date
    
    # build the storage directory name from the configuration dictionary
    toy_id  = str(config_json["architecture"][0]) + "D"         + "/"
    toy_id += "NRef_"     + str(config_json["N_Ref"])           + "_"
    toy_id += "NBkg_"     + str(config_json["N_Bkg"])           + "_"
    toy_id += "NSig_"     + str(config_json["N_Sig"])           + "_"
    toy_id += "NP_"       + str(config_json["NP"])              + "_"
    toy_id += "Epochs_"   + str(config_json["epochs"])          + "_"
    toy_id += "Patience_" + str(config_json["patience"])        + "_"
    toy_id += "WClip_"    + str(config_json["weight_clipping"]) + "_"
    toy_id += "Arch_"     + str(config_json["architecture"]).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_")
    toy_id += "/"
    toy_id += current_date + "/"

    # add the toy_id to the output directory
    config_json["output_directory"] += toy_id
    
    # save the name of the python script to execute
    config_json["pyscript"] = args.pyscript
    
    # save the number of toys to expect
    config_json["toys"] = args.toys
    
    # create the output directory if it does not exist
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
        
    # if args.debug:
    #     # create the debug files
    #     create_debug_files(config_json["output_directory"], "debug")
    
    # create the config file
    config_name = f"config_{current_date}"
    config_json["jsonfile"] = create_config_file(config_json, config_json["output_directory"], name=config_name)
    
    """
    LOAD DATASET and save the tensor in a file in the tensor_path 
    """
    
    load_dataset(config_json["DATA_PATH"], config_json["TENSOR_PATH"], config_json["NP"])
    
    if args.local:
        # launch toys
        for i in range(args.toys):
            print("Running toy %i / %i" %(i+1, args.toys))
            # os.system("python %s/%s -j %s -i %i --debug %s" %(os.getcwd(), args.pyscript, config_json["jsonfile"], i, str(args.debug)))
            os.system("python %s/%s -j %s -i %i --debug %s" %(os.getcwd(), args.pyscript, config_json["jsonfile"], i, False))
    
    if not args.local:
        # launch toys
        label = "folder-log-jobs"
        os.system("mkdir -p %s" %(label))
        for i in range(args.toys):        
            # src file
            script_src = open("%s/%i.src" %(label, i) , 'w')
            
            script_src.write("#!/bin/bash\n")
            script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh\n")
            script_src.write("python %s/%s -j %s -i %i -d %s" %(os.getcwd(), args.pyscript, config_json["jsonfile"], i, str(args.debug)))
            script_src.close()
            os.system("chmod a+x %s/%i.src" %(label, i))
            # condor file
            script_condor = open("%s/%i.condor" %(label, i) , 'w')
            script_condor.write("executable = %s/%i.src\n" %(label, i))
            script_condor.write("universe = vanilla\n")
            script_condor.write("output = %s/%i.out\n" %(label, i))
            script_condor.write("error =  %s/%i.err\n" %(label, i))
            script_condor.write("log = %s/%i.log\n" %(label, i))
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
            script_condor.write("queue\n")
            script_condor.close()
            # condor file submission
            os.system("condor_submit %s/%i.condor" %(label,i))

    if args.debug:
        global_end = time()
        global_time = global_end - global_start
        print("Total time: %.2f s" %(global_time))

        # write the total time to the debug file
        with h5py.File(config_json["output_directory"] + "debug.h5", "a") as h5file:
            h5file["global"][0] = global_time


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pyscript", type=str,   help="name of python script to execute", required=True)
    parser.add_argument("-t", "--toys",     type=int,   help="number of toys to run",            required=True)
    parser.add_argument("-l", "--local",    type=bool,  help="run locally",                      default=False)
    parser.add_argument("-d", "--debug",    type=bool,  help="debug timing",                     default=False)
    parser.add_argument("-w", "--wclip",    type=float, help="w clipping value",                 required=True)
    
    args = parser.parse_args()
    print(f"args: {args}")
    print(f"args: {args.wclip}")
    
    c_json = {
        "DATA_PATH"         : "/home/ubuntu/NPLM.torch/notebooks/data/",
        "TENSOR_PATH"       : "/home/ubuntu/NPLM.torch/notebooks/tensor/data_tensor.pt",
        "NP"                : "Z",
        "N_Ref"             : 1_000_000,
        "N_Bkg"             : 200_000,
        "N_Sig"             : 0,
        "output_directory"  : OUTPUT_DIRECTORY,
        "input_directory"   : INPUT_DIRECTORY,
        "epochs"            : 500_000,
        "patience"          : 5_000,
        "architecture"      : [5, 5, 5, 5, 1],
        "weight_clipping"   : args.wclip,
    }
    
    print(c_json)
    
    main(args, config_json = c_json)