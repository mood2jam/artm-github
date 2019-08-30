<h1 align="center">
Augmentations with Random Triplet Mining for Clustering (ARTM)
</h1">
<p align="center">
  <img width="500" height="500" src="https://restricted.vdl.afrl.af.mil/webdav/programs/atrpedia/Nontechnical_Materials/People/Moody_Jamison/Unsupervised_Clustering_Project/Resources/updated_our_method.gif">
</p>

### Usage
1. Start by cloning the repository.
2. Run the startup.sh and install requirements found in requirements.txt (especially make sure torchvision version 0.2.2 is installed)
2. Run ATRM through run.py. Feel free to change the "dset" argument. Current supported datasets are "MNIST", "CIFAR10", and "FASHIONMNIST."
3. Set "show_plots" to True to visualize augmented images, distance histograms, and confusion matricies for accuracy.
4. Add your own dataset by creating elif statements specific for your dataset in run.py, triplet.py, datasets.py, and/or networks.py. See the code for more details. You can copy what we did for the other datasets. 