import os
import json
import argparse

from analysis_utils import compute_date

INPUT_DIRECTORY  = "./"
OUTPUT_DIRECTORY = "./output/"

# configuration dictionary
config_json = {
    "N_Ref"             : 200_000,
    "N_Bkg"             : 2_000,
    "N_Sig"             : 10,
    "SIG_LOC"           : 6.40,
    "SIG_STD"           : 0.16,
    "output_directory"  : OUTPUT_DIRECTORY,
    "input_directory"   : INPUT_DIRECTORY,
    "epochs"            : 200_000,
    "patience"          : 5_000,
    "architecture"      : [1, 4, 1],
    "weight_clipping"   : 9, 
}

def create_config_file(config_table, path, name="config"):
    with open("%s/%s.json"%(path, name), "w") as outfile:
        json.dump(config_table, outfile, indent=4)
    return "%s/%s.json"%(path, name)


def main(args):
    
    current_date = compute_date()
    config_json["date"] = current_date
    
    # build the storage directory name from the configuration dictionary
    toy_id  = str(config_json["architecture"][0]) + "D"         + "/"
    toy_id += "NRef_"     + str(config_json["N_Ref"])           + "_"
    toy_id += "NBkg_"     + str(config_json["N_Bkg"])           + "_"
    toy_id += "NSig_"     + str(config_json["N_Sig"])           + "_"
    toy_id += "SigLoc_"   + str(config_json["SIG_LOC"])         + "_"
    toy_id += "SigStd_"   + str(config_json["SIG_STD"])         + "_"
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
    
    # create the output directory if it does not exist
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    
    # create the config file
    config_name = f"config_{current_date}.json"
    config_json["jsonfile"] = create_config_file(config_json, config_json["output_directory"], name=config_name)
    
    if args.local:
        # launch toys
        for i in range(args.toys):
            print("Running toy %i / %i" %(i+1, args.toys))
            os.system("python %s/%s -j %s -i %i" %(os.getcwd(), args.pyscript, config_json["jsonfile"], i))
    
    if not args.local:
        # launch toys
        label = "folder-log-jobs"
        os.system("mkdir -p %s" %(label))
        for i in range(args.toys):        
            # src file
            script_src = open("%s/%i.src" %(label, i) , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh\n")
            script_src.write("python %s/%s -j %s -i %i" %(os.getcwd(), args.pyscript, config_json["jsonfile"], i))
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

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pyscript", type=str,  help="name of python script to execute", required=True)
    parser.add_argument("-t", "--toys",     type=int,  help="number of toys to run",            required=True)
    parser.add_argument("-l", "--local",    type=bool, help="run locally",                      default=False)
    
    args = parser.parse_args()
    
    main(args)