# XraySegmentation

We've included a Jupyter Notebook that shows how to create the different results seen in the related paper.
Find this notebook in Project28.ipynb.


**Short Description of Project:**

For this project, a deep convolutional network will be created and trained in order to perform good automated segmentation of X-ray images. The dataset thus consists of these X-ray images that are in grayscale and various sizes. Raw and segmented images are included in this dataset, which will be used for training and validating the network.

We will divide the project into three phases. In the first phase, we will focus on finding candidates for the network architectures, and choose one (or more) to focus on. This phase will mostly consist of researching and exploring the current development in regards to deep convolutional networks.

Next, we will consider the dataset and find ways to augment the data in various ways to increase the size of the dataset. These augmentations will also influence the performance of the model in various ways. For instance, if the images are reduced in size, the model will likely be way faster at training and computing, but might also reduce its performance. These factors will be explored and evaluated in this phase. Finally, a dataset should have been made that is well suited for training the network.

The last phase will focus on tuning the network. Here, various parameters of the network will be tuned in order to find out which parameters produce the best results. These parameters will likely include: batch size, regularization, learning rate, choice of optimizer, number of epochs, etc. However, we will not try to change the architecture of the chosen model, as this would be out of the scope for this project.

For evaluating the network, we will also consider the different ways that this can be done. In the end, we hope to have a network that performs the segmentation well.

