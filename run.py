
from PIL import Image
import torchvision.datasets.mnist
from torchvision import transforms
import argparse
from triplet import run_net

import json
import os

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--do_learn', type=bool, help='True or False, whether or not to train network', default=True)
parser.add_argument('--batch_size', '-s', type=int, help='batch size', default=16)
parser.add_argument('--starting_lr', '-l', type=float, help='set starting learning rate', default=0.001)
parser.add_argument('--num_epochs', '-e', type=int, help='number of epochs', default=50)
parser.add_argument('--weight_decay', '-w', type=int, help='weight decay', default=0.0001)
parser.add_argument('--dset', '-d', type=str, help='dataset name: MNIST, CIFAR10 or FASHIONMNIST (feel free to add your own datasets)', default='MNIST')
parser.add_argument('--show_plots', '-p', help='show histograms, confusion matricies, and images throughout training',
                    default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument('--rotate', type=int, help='constant that adjusts how the learning rate changes', default=10)
parser.add_argument('--shear', type=int, help='constant that adjusts how the learning rate changes', default=10)
parser.add_argument('--scale', type=float, help='float that tells us how big and small to adjust our images (less than 1.0)', default=0.3)
parser.add_argument("--contrast", type=float, help="Contrast change amount in the color jitter function", default=0.3)
parser.add_argument("--saturation", type=float, help="Saturation change amount in the color jitter function", default=0.3)
parser.add_argument("--hue", type=float, help="Hue change amount in the color jitter function", default=0.2)
parser.add_argument("--rand_crop_size", type=int, help="size of square random crop in pixels", default=28)
parser.add_argument("--brightness", type=float, help="Range to change the brightness of the image randomly (from 0 to 1)", default=0)
parser.add_argument('--test_fraction', type=float, help='float (between 0.0 and 1.0) to use only a fraction of the data when testing', default=1)
parser.add_argument('--train_fraction', type=float, help='float (between 0.0 and 1.0) to use only a fraction of the data when training', default=1)
parser.add_argument('--margin', type=float, help='margin for triplet and constrastive loss', default=1.0)
parser.add_argument('--clusterer', type=str, help="Choose clustering algorithm to use after each epoch (kmeans or hdbscan)", default="kmeans")
parser.add_argument('--rtm_index', type=int, help="Controls how similar we want the different pairs to be (Between 0 and num_pairs - 1)", default=3)
parser.add_argument('--num_pairs', type=int, help="Choses the number of random pairs to look at when making the similar pairs", default=10)
parser.add_argument("--model_path", type=str, help="Where the embedding model is saved (not triplet net)", default="")
parser.add_argument("--num_clusters", type=int, help="Number of clusters we are going for", default=10)
parser.add_argument("--run_id", type=str, help="ID used for the run, usually datetime string", default=None)
parser.add_argument("--gpu_id", type=int, help="GPU number to use", default=0)
parser.add_argument("--params_path", type=str, help="Specific parameter dictionary to load (json)", default=None)
parser.add_argument("--delete_arches", help="Whether or not to delete the architecture of network after network training is done",
                    default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument("--fixed_seed", type=int, help="A fixed random seed for reproducible results", default=None)

args = parser.parse_args()

params = {"do_learn":                 args.do_learn,
          "starting_lr":              args.starting_lr,
          "batch_size":               args.batch_size,
          "ending_lr":                args.starting_lr,
          "num_epochs":               args.num_epochs,
          "weight_decay":             args.weight_decay,
          "dset":                     args.dset,
          "show_plots":               args.show_plots,
          "rotate":                   args.rotate,
          "shear":                    args.shear,
          "scale":                    args.scale,
          "test_fraction":            args.test_fraction,
          "train_fraction":           args.train_fraction,
          "margin":                   args.margin,
          "pairwise_accuracy":        0.0,
          "clustering_accuracy":      0.0,
          "nmi":                      0.0,
          "ari":                      0.0,
          "clusterer":                args.clusterer,
          "rtm_index":                args.rtm_index,
          "num_pairs":                args.num_pairs,
          "similar_random_pair_accuracy": 0.0,
          "different_random_pair_accuracy": 0.0,
          "max_clustering_accuracy":  0.0,
          "contrast":                 args.contrast,
          "saturation":               args.saturation,
          "hue":                      args.hue,
          "rand_crop_size":           args.rand_crop_size,
          "train_fraction":           args.train_fraction,
          "embedding_net_path":       args.model_path,
          "curr_epoch":               0,
          "num_clusters":             args.num_clusters,
          "brightness":               args.brightness,
          "run_id":                   args.run_id,
          "gpu_id":                   args.gpu_id,
          "params_path":              args.params_path,
          "delete_arches":            args.delete_arches,
          "fixed_seed":               args.fixed_seed
          }

# Defines the transformations we do on the images
class DatasetTransforms:
   def __init__(self):
      if params["dset"]=="MNIST":
         self.augmentation = torchvision.transforms.Compose([
            transforms.RandomAffine(params["rotate"], shear=params["shear"], scale=(1-params["scale"], 1+params["scale"]), resample=Image.BILINEAR),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
         ])
      elif params["dset"] == "FASHIONMNIST":
         self.augmentation = transforms.Compose([
            transforms.RandomAffine(params["rotate"], scale=(1 - params["scale"], 1 + params["scale"]), resample=Image.BILINEAR),
            transforms.ColorJitter(brightness=params["brightness"], contrast=params["contrast"], saturation=params["saturation"], hue=params["hue"]),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
         ])
      elif params["dset"]=="CIFAR10":
         self.augmentation = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=params["brightness"], contrast=params["contrast"], saturation=params["saturation"], hue=params["hue"]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(params["rand_crop_size"], padding=None, pad_if_needed=False),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
         ])
      self.basic = transforms.Compose([
         transforms.ToTensor(),
      ])


if __name__=='__main__':

   if params["params_path"] is not None:
      with open(params["params_path"]) as json_file:
         new_params = json.load(json_file)
      for key in new_params.keys():
         params[key] = new_params[key] # Overide our parameters with loaded parameters

   trans = DatasetTransforms()
   run_net(params, trans)

   with open('param_json/{0:0.2f}_ACC_{1}_{2}_params.txt'.format(params["max_clustering_accuracy"] * 100,
                                                                 params["dset"], params["run_id"]), 'w') as outfile:
      json.dump(params, outfile)

   if params["delete_arches"]:
      os.remove(params["embedding_net_path"])

