<h1 align="center">
Augmentations with Random Triplet Mining for Clustering (ARTM)
</h1">



### Usage
1. Start by cloning the repository.
2. Run the startup.sh and install requirements found in requirements.txt (especially make sure torchvision version 0.2.2 is installed)
2. Run ATRM through run.py. Feel free to change the "dset" argument. Current supported datasets are "MNIST", "CIFAR10", and "FASHIONMNIST."
3. Set "show_plots" to True to visualize augmented images, distance histograms, and confusion matricies for accuracy.
4. Add your own dataset by creating elif statements specific for your dataset in run.py, triplet.py, datasets.py, and/or networks.py. See the code for more details. You can copy what we did for the other datasets. 

Access the paper here: http://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Nina_A_Decoder-Free_Approach_for_Unsupervised_Clustering_and_Manifold_Learning_with_ICCVW_2019_paper.pdf

<h2 align="left">
BibTex
</h2">
  
Use the following to cite our paper:
  
@inproceedings{nina2019decoder,
  title={A Decoder-Free Approach for Unsupervised Clustering and Manifold Learning with Random Triplet Mining},
  author={Nina, Oliver and Moody, Jamison and Milligan, Clarissa},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
