# Modified from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
# Special thanks to Washington Garcia for help with the modifications

import sys
import os, glob
import random
import copy
import subprocess
import datetime
import json
import argparse

from math import *
from numpy import argsort
from multiprocessing import Pool
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--hp_load', type=str, help="Path to load a dictionary of hyperparameters. "
                                                "Change to None if you don't want to fix or schedule hyperparameters.",
                    default="scheduled_hp/fmnist_params.txt")
parser.add_argument('--hp_search', help="Whether or not to do a hyperparameter search", default=True,
                    type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument('--starting_num', type=int, help='The starting number of runs you want (accross all gpus)',
                    default=250)
parser.add_argument('--max_epochs', type=int, help="The maximum number epochs you want any run to go for", default=15)
parser.add_argument('--starting_epochs', type=int, help='The number of epochs you want to start out with per run',
                    default=3)
parser.add_argument('--decay_rate', type=int, help="How fast to narrow down your results when doing hp_search",
                    default=5)
parser.add_argument('--available_gpus', type=str, help="gpu numbers available (seperated by commas)",
                    default='4,5,6,7')
parser.add_argument('--models_per_gpu', type=int, help="numbers of models to run per gpu", default=3)

def hparams_as_typed(hparams):
    result = ""
    for key in hparams:
        result += "--"
        result += key
        result += "="
        result += str(hparams[key])
        result += " "

    return result

def get_random_hyperparameter_configuration():

    rotate=int(random.uniform(10, 30))
    shear=int(random.uniform(10, 30))
    scale=random.uniform(0.0, 0.3)
    brightness=random.uniform(0.0, 0.7)
    contrast=random.uniform(0.0, 0.7)
    saturation=random.uniform(0.0, 0.7)
    hue=random.uniform(0.0, 0.5)
    random_crop=int(random.uniform(20, 27))

    return rotate, shear, scale, brightness, contrast, saturation, hue, random_crop


def run_then_return_accuracy(hparams, num_iters, hyperparameters, gpu_id):
    orig_datetime = str(datetime.datetime.now()).split('.')
    first_part = orig_datetime[0].split()
    first_part.append(orig_datetime[1])
    datetime_string = '_'.join(first_part)
    # Do Epochs
    hparams["num_epochs"] = num_iters
    hparams["run_id"] = datetime_string
    hparams["gpu_id"] = gpu_id

    if args.hp_search: # Hyperparameters you want to search
        hparams["rotate"] = hyperparameters[0]
        hparams["shear"] = hyperparameters[1]
        hparams["scale"] = hyperparameters[2]
        hparams["brightness"] = hyperparameters[3]
        hparams["contrast"] = hyperparameters[4]
        hparams["saturation"] = hyperparameters[5]
        hparams["hue"] = hyperparameters[6]
        hparams["rand_crop_size"] = hyperparameters[7]


    # gpu_id_flag = "CUDA_VISIBLE_DEVICES={}".format(gpu_id)
    # "/dev/null 2>&1
    command = "{0} {1}".format("python run.py", hparams_as_typed(hparams))
    print(" ----- \n{}".format(command))

    os.system(command)

    print(" %%%%% Job finished! \n{}".format(hparams_as_typed(hparams)))

    # Gets the associated text file id

    dict_path = os.popen('ls param_json/*{0}_params.txt'.format(hparams["run_id"])).read().strip()                      # Gets the unique parameters associated with our run

    if dict_path == '':
        return 0.0

    if '\n' in dict_path:
        dict_path = dict_path.split('\n')[-1] # Handles the rare case that use the same run id more that once.

    with open(dict_path) as json_file:
        loaded_params = json.load(json_file) # Loads in the associated dictionary

    return loaded_params["max_clustering_accuracy"]

def HYPERBAND(hparams):

    # starting_num = 5  # Default is 400
    # min_epochs = 2      # Default is 2
    # max_epochs = 10     # Default is 25
    # decay_rate = 5      # Default is 5

    starting_num = args.starting_num

    if args.hp_search:
        min_epochs = args.starting_epochs
        max_epochs = args.max_epochs
        decay_rate = args.decay_rate

        r = starting_num
        run_sizes = []

        while r > 1:
            run_sizes.append(r)
            r = r // decay_rate

        l = len(run_sizes)
        get_curr_epochs = lambda x: int(2 ** x * (x / (l - 1)) * (max_epochs - min_epochs) / (2 ** (l - 1)) + min_epochs)
        epoch_sizes = [get_curr_epochs(i) for i in range(l)]
    else:
        run_sizes = [args.starting_num]
        l = len(run_sizes)
        epoch_sizes = [args.starting_epochs]

    print("Number of models:", run_sizes, "Epochs per model", epoch_sizes)

    models_per_gpu = args.models_per_gpu
    avail_gpus = [int(i) for i in args.available_gpus.split(",")]
    num_gpu = len(avail_gpus)

    T = [get_random_hyperparameter_configuration() for i in range(run_sizes[0])]

    for i in range(l):

        r_i = epoch_sizes[i]
        print(' ---- \nAt s: {}, i: {}, r_i: {}, T is: {}'.format(0, i, r_i, T))

        runs = [(copy.deepcopy(hparams), r_i, t) for t in T]
        print("Runs:", runs)
        # Now tag runs with a GPU id and add to pending jobs, until no more runs
        all_runs = []
        while len(runs) > 0:
            gpuPool = Pool(num_gpu * models_per_gpu)
            gpu_subprocess_hparams_list = []

            # this was intended for 8+ gpu on HPC and just rotated through each gpu assigning jobs round-robin
            for gpu_id in avail_gpus:
                # builds a list of functionals from your hparams, [(*hparams, gpu_id), ..., (*hparams_n, gpu_id_n)]
                model_hparams_per_gpu = [runs.pop() + (gpu_id,)
                                        for i in range(models_per_gpu) if len(runs) != 0]
                # hits run_then_return_accuracy with your functionals created earlier with replacement
                model_hparams_per_gpu = [(hparams_i, gpuPool.apply_async(run_then_return_accuracy, hparams_i))
                                        for hparams_i in model_hparams_per_gpu]

                gpu_subprocess_hparams_list.extend(model_hparams_per_gpu)

            all_runs.extend(gpu_subprocess_hparams_list)
            gpuPool.close()
            gpuPool.join()

        param_to_acc_pairs = [(x[0][2], x[1].get()) for x in all_runs]
        param_to_acc_pairs.sort(key=lambda x: x[1])

        print("Before")
        for param_i, acc_i in param_to_acc_pairs:
            print(f"{str(param_i)}\t->\t{acc_i}\n")

        if i < len(run_sizes) - 1:
            param_to_acc_pairs = param_to_acc_pairs[::-1][:run_sizes[i+1]]

        print("After")
        for param_i, acc_i in param_to_acc_pairs:
            print(f"{str(param_i)}\t->\t{acc_i}\n")
        T = [param_to_acc_pair[0] for param_to_acc_pair in param_to_acc_pairs]

        # Changes the seed after each run to find a configuration that generalizes to multiple seeds
        with open(args.hp_load) as json_file:
            loaded_params = json.load(json_file)
        loaded_params["fixed_seed"] += 10
        with open(args.hp_load, 'w') as outfile:
            json.dump(loaded_params, outfile)
        hparams = loaded_params

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.hp_search)
    hparams = dict()
    if args.hp_load is not None: # Load in a dictionary of fixed parameters
        with open(args.hp_load) as json_file:
            hparams = json.load(json_file)

    HYPERBAND(hparams)
