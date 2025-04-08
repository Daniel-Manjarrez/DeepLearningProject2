# Project 2 | Convolutional Neural Networks & Computer Graphics
 Image manipulation and two player games are common graphics applications. In this project, we
 evaluate the performance of classifiers and regressors on an image segmentation task, and
 a simple two player game. Our single classifier program generates statistical accuracy and
 confusion matrices for several classifiers and regressors (linear regression, linear SVM, k-nearest
 neighbors, and multilayer perceptron) using provided images with normals, and game play datasets
 (*Tic Tac Toe*). 

## ImageSegmentation Installation and Usage

To get started with this project, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

To use ImageSegmentation program, run the following inside the `ImageSegmentation` directory:

```bash
Python ImageSegmentation.py
```

## Speed up gained from GPU implementation 
cpu_runtime_log.txt within the colorization/ directory has some instances
of the total runtime and training time for the CNN utilizing the CPU.
gpu_runtime_log.txt within the gpu_colorization/ directory has these 
measurements when we utilized a GPU. 

GPU implementation halfed the training time compared to the CPU implementation's training. 

## Fine Tuning Notes: 
Experiment of chanigng the number of feature maps for interior CNNs to see if higher accuracy: 
    - Average accuracy before: Test MSE: 0.001686 or 0.1686% 
    - Increase number of feature maps in deeper layers for more complex feature learning & decrease shallower layers
    to reduce computational cost: average accuracy of: Test MSE: 0.002243 or 0.2243% 
    - decrease number of channels in downsampling & increase upsampling: 
    average accuracy of: Test MSE: 0.001706 or 0.1706%
    - Conclusion: model was already fine tuned enough
    so changing these feature maps only increased
    the loss aka the changes were detrimental 

## Datasets 
Download the RGBN Datasets from this link: https://gfx.cs.princeton.edu/gfx/proj/rgbn/ (I left them zipped and sent to ec2 instance, then unzipped once at ec2 using ```unzip \*.zip```)

Download the Tic Tac Toe datasets from this link: https://courseworks2.columbia.edu/files/23246302/download?download_frd=1 (courseworks files section)
