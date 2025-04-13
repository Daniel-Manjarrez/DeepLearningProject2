# Project 2 Report | Convolutional Neural Networks & Computer Graphics
Image colorization model for estimating RGB colors for grayscale images or video frames to improve their aesthetic and perceptual quality. Data preparation script utilized to prepare data. Includes script for training and testing a CNN regressor for automatic colorization of an inputted black and white image. Also includes transfer learning exercise for developing a strategy to fine-tune the network for the best performance on a new benchmark dataset

## Part 1: Dataset Processing

### Dataset Processing Installation and Usage
To get started with this project, you need to install the required dependencies. You can do this by running the following command while still in the `DatasetProcessing` directory:

```bash
pip install -r requirements.txt
```

To use the `process_dataset` program, run the following inside the `DatasetProcessing` directory:

```bash
python process_dataset.py
```

### Explaining Implementation
The `process_dataset.py` program performs a series of steps to preprocess a set of facial images, augment them, and convert them into the L\*a\*b\* color space. The following steps outline the implementation:

1. **Resizing the Given Image**: Images are resized to 128x128 pixels for standardization.
2. **Image Augmentation**: Random transformations are applied, including:
   - Random flipping (50% probability)
   - Random cropping and resizing
   - Random brightness adjustment
3. **Converting to L\*a\*b\* Color Space**: Images are converted to the L\*a\*b\* color space, and each channel (L, a, b) is saved separately.
4. **Saving Augmented and LAB Images**: Augmented images are saved in RGB format and as separate LAB channels.

---

## Part 2: Regressor

### Regressor Installation and Usage
To get started with this project, you need to install the required dependencies. Run the following command while still in the `Regressor` directory:

```bash
pip install -r requirements.txt
```

To build the regressor model, run:

```bash
python regressor.py
```

To test the built regressor model, run:

```bash
python predict_mean_chrominance.py
```

### Explaining Implementation
The `regressor.py` program defines a CNN model for predicting mean chrominance values (a\* and b\*). Key components include:
- **Model Architecture**: 7 convolutional layers followed by a fully connected layer.
- **Training**: Uses MSE loss and Adam optimizer, with the best model saved based on validation loss.

---

## Part 3 and 4: Colorization and GPU Computing

### Installation and Usage
Install dependencies in the `colorization` or `gpu_colorization` directory:

```bash
pip install -r requirements.txt
```

To run the CPU version:

```bash
python cpu_colorized.py
```

To run the GPU version:

```bash
python gpu_colorized.py
```

### Explaining Implementation
The ColorizationCNN model uses 5 downsampling (encoder) and 5 upsampling (decoder) layers. Training involves:
1. Augmenting the dataset.
2. Training on L and a\*b\* tensors with MSE loss and Adam optimizer.
3. Testing on unaugmented images, with results saved in the `PredictedColorizedImg` folder.

---

## Part 5: Transfer Learning (and Hyperparameter Tuning)

### Installation and Usage
Install dependencies in the `fine_tuning/gpu_colorization_fine_tuned` directory:

```bash
pip install -r requirements.txt
```

Run the experimental code:

```bash
python gpu_color_img.py
```

### Fine-Tuning and Transfer Learning
- **Baseline Model**: Achieved a Test MSE of 0.001915 (0.1915%).
- **Experiments**: Modifying feature maps did not improve accuracy.
- **Transfer Learning**: Pretrained weights were fine-tuned on the NCD dataset using progressive unfreezing and learning rate adjustment.

---


## Datasets 
Download the RGBN Datasets from this link: https://gfx.cs.princeton.edu/gfx/proj/rgbn/ (I left them zipped and sent to ec2 instance, then unzipped once at ec2 using ```unzip \*.zip```)

Download the Tic Tac Toe datasets from this link: https://courseworks2.columbia.edu/files/23246302/download?download_frd=1 (courseworks files section)

---

## References
Anwar, S., Tahir, M., & Mian, A. (2020). Image Colorization: A Survey and Dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/TPAMI.2020.2977027
