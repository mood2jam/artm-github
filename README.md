<h1 align="center">
Augmentations with Random Triplet Mining for Clustering (ARTM)
</h1">

<p>
Access the paper here: https://drive.google.com/file/d/11ADht4hmCCN1hUuiP6l3pB3Tj3Uw1HJ1/view?usp=sharing
</p>

### Usage
1. Start by cloning the repository.
2. Run the startup.sh and install requirements found in requirements.txt (especially make sure torchvision version 0.2.2 is installed)
2. Run ATRM through run.py. Feel free to change the "dset" argument. Current supported datasets are "MNIST", "CIFAR10", and "FASHIONMNIST."
3. Set "show_plots" to True to visualize augmented images, distance histograms, and confusion matricies for accuracy.
4. Add your own dataset by creating elif statements specific for your dataset in run.py, triplet.py, datasets.py, and/or networks.py. See the code for more details. You can copy what we did for the other datasets. 

