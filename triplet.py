import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from tensorboardX import SummaryWriter
import datetime
from datasets import RandomTripletMiningDataset
from util import show_datasets, get_pairwise_accuracy, write_to_tensorboard, print_clustering_accuracy
from networks import EmbeddingNet, TripletNet
from sklearn.cluster import KMeans
import hdbscan
from losses import TripletLoss
import random


def train(model, device, train_loader, train_loss, epoch, optimizer, sample_data, params=None):
   model.train()
   criterion = train_loss

   for batch_idx, (data, target) in enumerate(train_loader):
      for i in range(len(data)):
         data[i] = data[i].to(device)

      optimizer.zero_grad()
      anchor, positive, negative = model(data[0], data[1], data[2])

      loss = criterion(anchor, positive, negative, device=device)
      loss.backward()
      optimizer.step()

      if batch_idx % 100 == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx*params["batch_size"], len(train_loader.dataset),
            100. * batch_idx*params["batch_size"] / len(train_loader.dataset), loss.item()))
      if batch_idx * params["batch_size"] / len(train_loader.dataset) > params["train_fraction"]:
         break

def test(model, device, test_loader, writer, record_histograms=True, epoch=0, params=None):
   """
   Use if you have labels.
   """
   with torch.no_grad():
      model.eval()
      positive_distances, negative_distances, embeddings, embedding_targets = [], [], [], []
      indices = test_loader.dataset.original_indices
      for batch_idx, (data, targets) in enumerate(test_loader):
         for i in range(len(data)):
            data[i] = data[i].to(device)

         anchor, positive, negative = model(data[0], data[1], data[2])
         distance_positive = (anchor - positive).pow(2).sum(1).pow(.5)
         distance_negative = (anchor - negative).pow(2).sum(1).pow(.5)

         embeddings.append(anchor)
         embedding_targets.append(targets[0])
         positive_distances.append(torch.squeeze(distance_positive))
         negative_distances.append(torch.squeeze(distance_negative))

         if batch_idx % 500 == 0:
            print('Testing: [{}/{} ({:.0f}%)]'.format(
               batch_idx * params["batch_size"], len(test_loader.dataset),
               100. * batch_idx * params["batch_size"] / len(test_loader.dataset)))

         if batch_idx * params["batch_size"] / len(test_loader.dataset) > params["test_fraction"]:
            break

      positive_distances = torch.cat(positive_distances, dim=0)
      negative_distances = torch.cat(negative_distances, dim=0)
      embeddings = torch.cat(embeddings, dim=0)
      embedding_targets = torch.cat(embedding_targets, dim=0)
      all_distances = torch.cat((positive_distances, negative_distances), dim=0)

      if params["clusterer"] == "kmeans":
         kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings.cpu().numpy())
         print("Using KMEANS.")
         clustering_accuracy, nmi, ari = print_clustering_accuracy(params, kmeans.labels_, embedding_targets.cpu().numpy(), 10)
      else:
         clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=700)
         clusterer.fit(embeddings.cpu().numpy())
         num_clusters = clusterer.labels_.max() + 1
         clustered = np.where(clusterer.labels_ != -1)[0]
         print("Using HDBSCAN. Percentage clustered: ", np.round(len(clustered)/len(clusterer.labels_), 3))
         assert num_clusters <= 10
         clustering_accuracy, nmi, ari = print_clustering_accuracy(params, clusterer.labels_[clustered], embedding_targets.cpu().numpy()[clustered], 10)


      pairwise_accuracy, boundary = get_pairwise_accuracy(positive_distances, negative_distances)
      if params["show_plots"]:
         plt.title("Similar and Different Distances")
         plt.axvline(x=boundary)
         plt.hist(positive_distances.cpu(), bins=200, label="Similar")
         plt.hist(negative_distances.cpu(), bins=200, label="Different")
         plt.legend()
         plt.tight_layout()
         plt.show()

      pos_values = positive_distances.cpu()
      neg_values = negative_distances.cpu()

      max_val = max([torch.max(negative_distances).item(), torch.max(positive_distances).item()])

      if record_histograms:
         writer.add_histogram('distances', pos_values, epoch)
         writer.add_histogram('distances', neg_values, epoch)

      print("Pairwise Accuracy: {:.5f}\tClustering Accuracy: {:.5f}\tNMI: {:.5f}\tARI: {:.5f}\tMargin: {:.5f}".format(pairwise_accuracy, clustering_accuracy, nmi, ari, params["margin"]))
      if clustering_accuracy > params["max_clustering_accuracy"]:
         params["max_clustering_accuracy"] = clustering_accuracy

      params["pairwise_accuracy"] = pairwise_accuracy
      params["clustering_accuracy"] = clustering_accuracy
      params["ari"] = ari
      params["nmi"] = nmi

   return embeddings, embedding_targets, indices

def run_net(params, transforms):

   if params["fixed_seed"] is not None:
      torch.backends.cudnn.deterministic = True
      random.seed(1)
      torch.manual_seed(1)
      torch.cuda.manual_seed(1)
      np.random.seed(1)

   if params["gpu_id"] is not None:
      device = torch.device("cuda:{}".format(params["gpu_id"]))
   else:
      device = torch.device('cpu')

   if params["dset"] == "CIFAR10":
      concatenated = torch.utils.data.ConcatDataset([CIFAR10(root='./data', train=True, download=True, transform=None),
                                                     CIFAR10(root='./data', train=False, download=True, transform=None)])
   elif params["dset"] == "MNIST":
      concatenated = torch.utils.data.ConcatDataset([MNIST(root='./data', train=True, download=True, transform=None),
                                                     MNIST(root='./data', train=False, download=True, transform=None)])
   elif params["dset"] == "FASHIONMNIST":
      concatenated = torch.utils.data.ConcatDataset([FashionMNIST(root='./data', train=True, download=True, transform=None),
                                                     FashionMNIST(root='./data', train=False, download=True, transform=None)])

   triplet_test = RandomTripletMiningDataset(concatenated, train=False, trans=transforms, device=None, params=params)

   # Initialize model
   if params["dset"] == "CIFAR10":
      embedding_net = EmbeddingNet(in_channels=3, adjusting_constant=5) # Change this to VGG for CIFAR10
   elif params["dset"] == "MNIST" or params["dset"] == "FASHIONMNIST":
      embedding_net = EmbeddingNet()
   model = TripletNet(embedding_net).to(device)

   # Sets the datetime string to use as an identifier for the future

   if params["run_id"] is None:
        params["run_id"] = '_'.join((str(datetime.datetime.now()).split('.')[0].split()))
   params["embedding_net_path"] = "arches/embedding_net_{}".format(params["run_id"])

   # Train our model
   if params["do_learn"]:
      # Initialize loss functions
      train_loss = TripletLoss(margin=params["margin"])

      test_loader = torch.utils.data.DataLoader(triplet_test, batch_size=params["batch_size"], shuffle=True)  # Test data is the same as train data but test data is not preshuffled

      writer = SummaryWriter(comment="triplet_{0}_{1}_{2}".format(params["dset"], params["num_pairs"], params["rtm_index"]))
      optimizer = optim.Adam(model.parameters(), lr=params["starting_lr"], weight_decay=params["weight_decay"])

      test(model, device, test_loader, writer, record_histograms=False, params=params)
      for epoch in range(params["num_epochs"]):
         params["curr_epoch"] = epoch

         torch.save(model.embedding_net.state_dict(), params["embedding_net_path"])

         triplet_train = RandomTripletMiningDataset(concatenated, train=True, trans=transforms, device=device,
                                                    params=params)
         train_loader = torch.utils.data.DataLoader(triplet_train, batch_size=params["batch_size"], shuffle=True)
         sample_loader = torch.utils.data.DataLoader(triplet_train, batch_size=len(triplet_train) // 100,
                                                     shuffle=True)  # Sample our data to get a reference point after every so often
         sample_data, sample_targets = next(iter(sample_loader))
         if params["show_plots"]:
            show_datasets(sample_data, show_plots=params["show_plots"])
         similar_pair_accuracy = np.round(
            len(np.where(sample_targets[0] == sample_targets[1])[0]) / len(sample_targets[0]), 3)
         different_pair_accuracy = np.round(
            len(np.where(sample_targets[0] != sample_targets[2])[0]) / len(sample_targets[0]), 3)
         print("Similar pair accuracy:", similar_pair_accuracy)
         print("Different pair accuracy:", different_pair_accuracy)
         params["different_random_pair_accuracy"] = different_pair_accuracy
         params["similar_random_pair_accuracy"] = similar_pair_accuracy
         train(model, device, train_loader, train_loss, epoch, optimizer, sample_data, params=params)
         embeddings, targets, indices = test(model, device, test_loader, writer, epoch=epoch, params=params)

         write_to_tensorboard(params, writer, epoch) # Writes to tensorboard at the end of the epoch

      writer.export_scalars_to_json(".all_scalars.json")
      writer.close()
