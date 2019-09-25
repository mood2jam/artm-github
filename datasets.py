import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from util import get_embeddings
import matplotlib.pyplot as plt

from torch.nn import functional as F

class SingleDataset(Dataset):
   """
   Train: Creates a compilation of training data and test data with augmented similar pairs
   Test: Creates random triplets from training data. Use if labels are available
   """

   def __init__(self, dataset, augment=False, trans=None, params=None):
      self.basic = trans.basic
      self.augmentation = trans.augmentation
      self.dataset = dataset
      self.params = params
      self.augment = augment
      self.concatenated = (type(dataset) == torch.utils.data.dataset.ConcatDataset)
      if self.concatenated:
         if params["dset"] == "CIFAR10":
            self.data = torch.cat(
               [torch.from_numpy(self.dataset.datasets[i].data) for i in range(len(self.dataset.datasets))], dim=0)
            self.targets = torch.cat(
               [torch.tensor(self.dataset.datasets[i].targets) for i in range(len(self.dataset.datasets))], dim=0)
         elif params["dset"] == "MNIST" or params["dset"] == "FASHIONMNIST":
            self.data = torch.cat([self.dataset.datasets[i].data for i in range(len(self.dataset.datasets))], dim=0)
            self.targets = torch.cat([self.dataset.datasets[i].targets for i in range(len(self.dataset.datasets))],
                                     dim=0)
      else:
         self.targets = self.dataset.targets
         self.data = self.dataset.data

   def __getitem__(self, index):
      img, target = self.data[index], self.targets[index]
      if self.basic is not None:
         if self.params["dset"] == "CIFAR10":
            img = Image.fromarray(img.numpy(), mode='RGB')
         elif self.params["dset"] == "MNIST" or self.params["dset"] == "FASHIONMNIST":
            img = Image.fromarray(img.numpy(), mode='L')

         if self.augment:
            img = self.augmentation(img)
         else:
            img = self.basic(img)

      return img, target

   def __len__(self):
      return self.data.shape[0]

class RandomTripletMiningDataset(Dataset):
   """
   Train: Creates a compilation of training data and test data with augmented similar pairs
   Test: Creates random triplets from training data. Use if labels are available
   """

   def __init__(self, dataset, train, trans, device, params=None):
      self.dataset = dataset
      self.train = train
      self.basic = trans.basic
      self.augment = trans.augmentation
      self.params = params
      self.concatenated = (type(dataset) == torch.utils.data.dataset.ConcatDataset)
      if self.concatenated:
         if params["dset"]=="CIFAR10":
            self.data = torch.cat([torch.from_numpy(self.dataset.datasets[i].data) for i in range(len(self.dataset.datasets))], dim=0)
            self.targets = torch.cat([torch.tensor(self.dataset.datasets[i].targets) for i in range(len(self.dataset.datasets))], dim=0)
         elif params["dset"]=="MNIST" or params["dset"]=="FASHIONMNIST":
            self.data = torch.cat([self.dataset.datasets[i].data for i in range(len(self.dataset.datasets))], dim=0)
            self.targets = torch.cat([self.dataset.datasets[i].targets for i in range(len(self.dataset.datasets))], dim=0)
      else:
         self.targets = self.dataset.targets
         self.data = self.dataset.data

      if self.train:
         # Creates three lists of indices that will be called later.
         # The first index will be a specific image, the second will be the same image (augmented)
         # and the third will be a random image (most likely different if dataset is balanced)
         self.original_indices = np.arange(self.data.shape[0])

         regular_dataloader = torch.utils.data.DataLoader(
            SingleDataset(dataset, augment=False, trans=trans, params=params),
            batch_size=params["batch_size"], shuffle=False)  # Do not shuffle

         print("Running data through previous network...")
         regular_embeddings, augmented_embeddings = get_embeddings(regular_dataloader, device, params)
         random_indices = np.copy(self.original_indices)
         np.random.shuffle(random_indices)
         augmented_distances, random_distances = [], []

         # if params["show_plots"]:
         #    # plt.title("Augmented and Random Distances (boundary is {:.3f})".format(boundary))
         #    # plt.axvline(x=rand_min, color='r')
         #    # plt.axvline(x=rand_max, color='r')
         #    plt.hist(random_distances.cpu(), bins=200, label="Random")
         #    # plt.hist(combined.cpu(), bins=bins, label="Combined")
         #    # plt.hist(augmented_distances.cpu(), bins=bins, label="Augmented")
         #    plt.legend()
         #    plt.show()

         print("RTM index is {0} (number of pairs is {1})".format(params["rtm_index"], params["num_pairs"]))

         if params["rtm_index"] is None: # Do not use RTM
            # if params["curr_epoch"] == 0:
            indices_mask = np.arange(len(self.original_indices))
            self.original_indices = self.original_indices[indices_mask]
            self.similar_indices = np.copy(self.original_indices)
            self.different_indices = random_indices[indices_mask]
         else:
            random_distances, random_indices = [], []
            indices_matrix = np.random.choice(self.original_indices, (params["num_pairs"], len(self.original_indices)))
            for i in range(params["num_pairs"]):
               shuffled_indices = np.copy(self.original_indices)
               np.random.shuffle(shuffled_indices)
               # distances.append(F.cosine_similarity(embeddings, embeddings[different_indices], 1))
               random_distances.append(torch.norm(regular_embeddings[self.original_indices] - regular_embeddings[indices_matrix[i,:]], p=2, dim=1))
               random_indices.append(shuffled_indices)

            # random_indices = np.vstack(random_indices)
            random_distances = torch.stack(random_distances).cpu()
            different_selection = np.argpartition(random_distances, params["rtm_index"], axis=0)[
                                  params["rtm_index"], :].numpy()  # Gets the rtm_index-th nearest neighbor
            different_indices = indices_matrix[different_selection, np.arange(self.original_indices.shape[0])]
            self.different_indices = different_indices
            self.similar_indices = np.copy(self.original_indices)

         print("Dataset is size:", len(self.original_indices))

      else:
         # Creates three lists of indices that will be called later.
         # The first index will be a specific label, the second will be the same label,
         # and the third will be a different label
         self.labels_set = set(self.targets.numpy())
         x_original_indices, x_similar_indices, x_different_indices = [], [], []
         for label in self.labels_set:
            original_indices = np.arange(len(self.targets))[np.where(self.targets == label)[0]]
            similar_indices = np.copy(original_indices)
            np.random.shuffle(similar_indices)
            different_indices = np.random.choice(np.arange(len(self.targets))[np.where(self.targets != label)[0]],
                                                 len(original_indices))
            x_original_indices.append(torch.from_numpy(original_indices))
            x_similar_indices.append(torch.from_numpy(similar_indices))
            x_different_indices.append(torch.from_numpy(different_indices))
         self.original_indices = torch.cat(x_original_indices, dim=0)
         self.similar_indices = torch.cat(x_similar_indices, dim=0)
         self.different_indices = torch.cat(x_different_indices, dim=0)

   def __getitem__(self, index):
      orig_idx = self.original_indices[index]
      sim_idx = self.similar_indices[index]
      diff_idx = self.different_indices[index]

      img1, target1 = self.data[orig_idx], self.targets[orig_idx]
      img2, target2 = self.data[sim_idx], self.targets[sim_idx]
      img3, target3 = self.data[diff_idx], self.targets[diff_idx]
      targets = [target1, target2, target3]

      if self.params["dset"] == "CIFAR10":
         img1 = Image.fromarray(img1.numpy(), mode='RGB')
         img2 = Image.fromarray(img2.numpy(), mode='RGB')
         img3 = Image.fromarray(img3.numpy(), mode='RGB')
      if self.params["dset"] == "MNIST" or self.params["dset"] == "FASHIONMNIST":
         img1 = Image.fromarray(img1.numpy(), mode='L')
         img2 = Image.fromarray(img2.numpy(), mode='L')
         img3 = Image.fromarray(img3.numpy(), mode='L')

      assert self.basic is not None and self.augment is not None

      if self.train:
         img1 = self.augment(img1)
         img2 = self.augment(img2)
         img3 = self.augment(img3)
      else:
         img1 = self.basic(img1)
         img2 = self.basic(img2)
         img3 = self.basic(img3)

      return [img1, img2, img3], targets

   def __len__(self):
      return len(self.original_indices)