# Military Aircraft Image Classification

## Project Overview

This project develops deep learning models to classify military aircraft images into **88 aircraft categories and one background class**.

The project compares a custom convolutional neural network with a transfer learning approach using a pretrained **ResNet-152** model. It includes dataset preparation, image augmentation, exploratory analysis, model training, checkpoint saving, and progressive fine-tuning.

## Dataset

The project uses the **AeroScan Military Aircraft Classification** dataset from Kaggle.

### Dataset Summary

* Training images: 20,717
* Validation images: 2,564
* Test images: 2,568
* Total classes: 89
* Aircraft classes: 88
* Background class: 1

Each aircraft image is paired with a text annotation containing its class ID. Images without a valid annotation are assigned to the background class.

Dataset source:

```text
ahnuf05/aeroscan-military-aircraft-classification
```

The notebook automatically downloads the dataset using `kagglehub` and stores it locally in:

```text
data/aeroscan_dataset/
```

## Aircraft Classes

The dataset contains aircraft such as:

* A-10
* B-2
* B-52
* C-130
* C-17
* F-14
* F-15
* F-16
* F-18
* F-22
* F-35
* MiG-29
* Rafale
* SR-71
* Su-57
* UH-60
* V-22

The complete dataset includes 88 aircraft types and one background category.

## Project Workflow

### 1. Environment Setup

The notebook installs and imports the required Python packages, creates local data and checkpoint directories, and downloads the dataset if it is not already available.

The required folders are created automatically:

```text
data/
checkpoints/
```

The notebook does not require Google Colab or Google Drive.

### 2. Custom Dataset Loader

A custom PyTorch `Dataset` class was created to:

* Load aircraft images
* Read class IDs from annotation files
* Assign missing annotations to the background class
* Convert images into RGB format
* Apply separate training and validation transformations

### 3. Image Preprocessing

All images are resized to 224 by 224 pixels and normalized using ImageNet normalization values.

Training data augmentation includes:

* Random horizontal flipping
* Random rotation
* Brightness and contrast adjustments
* Saturation and hue adjustments
* Random resized cropping

These transformations introduce visual variation and help reduce overfitting.

### 4. Exploratory Data Analysis

The notebook examines class frequencies and displays example images from the aircraft categories.

The background class was the largest category with 1,883 training images. Most aircraft classes contained approximately 200 to 240 images.

## Models

### Custom Convolutional Neural Network

The baseline model uses three convolutional blocks:

1. 32-filter convolution and max pooling
2. 64-filter convolution and max pooling
3. 128-filter convolution and max pooling

The extracted features are passed through:

* A 256-unit fully connected layer
* ReLU activation
* 50 percent dropout
* An 89-class output layer

The model was trained using:

* Adam optimizer
* Learning rate of 0.0003
* Weight decay of 0.0001
* Cross-entropy loss
* Batch size of 32

The best recorded validation accuracy for the baseline CNN was approximately **17.67 percent**.

### ResNet-152 Transfer Learning

A pretrained ResNet-152 model was used to improve classification performance through transfer learning.

Training was divided into three stages.

#### Stage 1: Train the Classification Layer

All pretrained ResNet layers were frozen and the original classifier was replaced with an 89-class output layer.

Only the final fully connected layer was trained.

#### Stage 2: Fine-Tune Layer 4

After training the classifier, the fourth ResNet block was unfrozen.

This allowed the model to adapt higher-level visual features to military aircraft classification while preserving most pretrained ImageNet features.

#### Stage 3: Fine-Tune Layers 3 and 4

The third and fourth ResNet blocks were then unfrozen and trained using a lower learning rate.

This progressive fine-tuning strategy reduces the risk of overwriting useful pretrained features while allowing deeper portions of the network to specialize in aircraft recognition.

## Checkpoint Management

Model checkpoints are saved locally in:

```text
checkpoints/
```

Each checkpoint contains:

* Model weights
* Optimizer state
* Current epoch
* Best validation accuracy
* Training loss history
* Validation loss history
* Validation accuracy history

Training can be resumed from an existing checkpoint if the checkpoint files are available.

Checkpoint files are excluded from GitHub because they can be large.

## Evaluation

Model performance is tracked using:

* Training loss
* Validation loss
* Validation accuracy
* Best-performing epoch
* Final validation accuracy

The notebook also generates loss curves and compares checkpoint performance across the ResNet fine-tuning stages.

## Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* NumPy
* Matplotlib
* Pillow
* scikit-learn
* KaggleHub
* Jupyter Notebook

## Repository Structure

```text
military-aircraft-image-classification/
│
├── military-aircraft-image-classification.ipynb
├── README.md
├── .gitignore
├── data/
│   └── aeroscan_dataset/
└── checkpoints/
```

The `data/` and `checkpoints/` directories are created automatically and are excluded from GitHub.

## Running the Project

1. Clone the repository.

```bash
git clone https://github.com/YOUR-USERNAME/military-aircraft-image-classification.git
```

2. Enter the project directory.

```bash
cd military-aircraft-image-classification
```

3. Open the notebook in Jupyter Notebook, JupyterLab, VS Code, or another compatible environment.

4. Run the setup cell to install `kagglehub`, create the local directories, and download the dataset.

5. Run the remaining cells in order.

6. Use a GPU-enabled environment for ResNet-152 training when possible.

## Requirements

The main required packages are:

```text
torch
torchvision
pandas
numpy
matplotlib
pillow
scikit-learn
kagglehub
tqdm
```

## Important Notes

The dataset and trained model checkpoint files are not included in the repository because of their size.

The notebook contains saved outputs from the original project, so the results and visualizations can be viewed without retraining the models.

Running the ResNet-152 training cells from scratch may take a significant amount of time without GPU acceleration.

## Future Improvements

Potential extensions include:

* Evaluating the final model on the held-out test set
* Producing precision, recall, and F1-score by aircraft class
* Generating a confusion matrix
* Addressing class imbalance with weighted loss
* Testing smaller pretrained models such as ResNet-50 or EfficientNet
* Adding early stopping
* Adding learning-rate scheduling
* Deploying the model through Streamlit or Gradio
* Displaying confidence scores for uploaded aircraft images

## Author

Developed by Russell Wagner as a final project for **STAT 421 at Texas A&M University**.
