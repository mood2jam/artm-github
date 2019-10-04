import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import torch
import sklearn
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from munkres import Munkres
from networks import get_embedding_net

nmi = normalized_mutual_info_score


def show_datasets(datasets, n=10, custom=None):
   fig = plt.figure(num=1, figsize=(2 * 3, 2 * n))
   gs1 = gridspec.GridSpec(n, 3)
   gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

   for i in range(3 * n):
      ax1 = plt.subplot(gs1[i])
      plt.axis('off')
      ax1.set_xticklabels([])
      ax1.set_yticklabels([])
      ax1.set_aspect('equal')
      if i % 3 == 0:
         if i == 0:
            plt.title("Originals")
         if custom is not None:
            plt.xlabel(custom[0][i // 3])
         plt.imshow(np.squeeze(np.moveaxis(datasets[0][(i) // 3].data.cpu().numpy(), 0, 2)), cmap="gray")

      elif i % 3 == 1:
         if i == 1:
            plt.title("Similar")
         if custom is not None:
            plt.xlabel(custom[1][i // 3])
         plt.imshow(np.squeeze(np.moveaxis(datasets[1][(i) // 3].data.cpu().numpy(), 0, 2)), cmap="gray")

      elif i % 3 == 2:
         if i == 2:
            plt.title("Different")
         if custom is not None:
            plt.xlabel(custom[2][i // 3])
         plt.imshow(np.squeeze(np.moveaxis(datasets[2][(i) // 3].data.cpu().numpy(), 0, 2)), cmap="gray")

   plt.show()

def write_to_tensorboard(params, writer, epoch):
   writer.add_scalar('data/clustering_accuracy', params["clustering_accuracy"], epoch)
   writer.add_scalar('data/pairwise_accuracy', params["pairwise_accuracy"], epoch)
   # writer.add_scalars('data/alphas', {'positive_alpha': params["alpha1"],
   #                                   'negative_alpha': params["alpha2"]}, epoch)
   # writer.add_scalar('data/losses/similar_loss', params["all_losses"][0][-1], epoch)
   # writer.add_scalar('data/losses/different_loss', params["all_losses"][1][-1], epoch)
   writer.add_scalars('data/random_pair_accuracy', {'similar_pairs': params["similar_random_pair_accuracy"],
                                                    'different_pairs': params["different_random_pair_accuracy"]}, epoch)

def get_pairwise_accuracy(similar_distances, different_distances):
   similar_distances = similar_distances.cpu().numpy()
   different_distances = different_distances.cpu().numpy()

   different_mean = np.mean(different_distances)
   similar_mean = np.mean(similar_distances)
   if different_mean > similar_mean:
      different_above = True
   else:
      different_above = False

   boundary = (np.mean(different_distances) + np.mean(similar_distances)) / 2

   if different_above:
      unique, counts = np.unique((similar_distances <= boundary), return_counts=True)
      counts_similar = dict(zip(unique, counts))
      unique, counts = np.unique((different_distances > boundary), return_counts=True)
      counts_different = dict(zip(unique, counts))
   else:
      unique, counts = np.unique((similar_distances > boundary), return_counts=True)
      counts_similar = dict(zip(unique, counts))
      unique, counts = np.unique((different_distances <= boundary), return_counts=True)
      counts_different = dict(zip(unique, counts))

   pairwise_accuracy = .5 * counts_similar[True] / len(similar_distances) + .5 * counts_different[True] / len(different_distances)

   return pairwise_accuracy, boundary

def get_embeddings(regular_dataloader, device, params):
   with torch.no_grad():
      embedding_net = get_embedding_net(params)
      embedding_net.load_state_dict(torch.load(params["embedding_net_path"]))
      embedding_net.eval()
      embedding_net.to(device)

      regular_embeddings, augmented_embeddings = [], []
      for batch_idx, (data, target) in enumerate(regular_dataloader):
         data = data.to(device)
         regular_embeddings.append(embedding_net(data))
         if batch_idx % 1000 == 0:
            print('Loading Regular Data: [{}/{} ({:.0f}%)]'.format(
               batch_idx * params["batch_size"], len(regular_dataloader.dataset),
               100. * batch_idx * params["batch_size"] / len(regular_dataloader.dataset)))

      regular_embeddings = torch.cat(regular_embeddings, dim=0)
   return regular_embeddings, augmented_embeddings

# The following functions relating to print_clustering_accuracy were modified from https://github.com/KlugerLab/SpectralNet
def calculate_cost_matrix(C, n_clusters):
   cost_matrix = np.zeros((n_clusters, n_clusters))

   for j in range(n_clusters):
      s = np.sum(C[:, j])  # number of examples in cluster i
      for i in range(n_clusters):
         t = C[i, j]
         cost_matrix[j, i] = s - t
   return cost_matrix


def get_cluster_labels_from_indices(indices):
   n_clusters = len(indices)
   clusterLabels = np.zeros(n_clusters)
   for i in range(n_clusters):
      clusterLabels[i] = indices[i][1]
   return clusterLabels


def get_y_preds(cluster_assignments, y_true, n_clusters):
   '''
   Computes the predicted labels, where label assignments now
   correspond to the actual labels in y_true (as estimated by Munkres)

   cluster_assignments:    array of labels, outputted by kmeans
   y_true:                 true labels
   n_clusters:             number of clusters in the dataset

   returns:    a tuple containing the accuracy and confusion matrix,
               in that order
   '''
   confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
   # compute accuracy based on optimal 1:1 assignment of clusters to labels
   cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
   indices = Munkres().compute(cost_matrix)
   kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
   y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
   return y_pred, confusion_matrix


def get_clustering_accuracy(cluster_assignments, y_true, n_clusters):
   '''
   Computes the accuracy based on the provided kmeans cluster assignments
   and true labels, using the Munkres algorithm

   cluster_assignments:    array of labels, outputted by kmeans
   y_true:                 true labels
   n_clusters:             number of clusters in the dataset

   returns:    a tuple containing the accuracy and confusion matrix,
               in that order
   '''
   y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
   # calculate the accuracy
   return np.mean(y_pred == y_true), confusion_matrix


def print_clustering_accuracy(params, cluster_assignments, y_true, n_clusters, extra_identifier='', aug_name="original"):
   """
   Convenience function: prints the accuracy
   """
   # get nmi score
   y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
   y_pred = np.squeeze(y_pred)
   y_true = np.squeeze(y_true)
   nmi_value = nmi(y_true, y_pred)
   ari_value = adjusted_rand_score(y_true, y_pred)
   # get accuracy
   clustering_accuracy, confusion_matrix = get_clustering_accuracy(cluster_assignments, y_true, n_clusters)
   # get the confusion matrix
   if params["show_plots"]:
      # Added in code to print and save confusion matrix and accuracy
      fig, ax = plt.subplots()
      ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

      for i in range(confusion_matrix.shape[0]):
         for j in range(confusion_matrix.shape[1]):
            c = confusion_matrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center')

      plt.title(
         "MNIST achieved {0} accuracy and {1} nmi".format(str(np.round(clustering_accuracy, 3)), str(np.round(nmi_value, 3))))
      plt.show()

   return clustering_accuracy, nmi_value, ari_value

