# COVID19-Chest-Xray-Classification-CompBio


This repository contains a computational biology final project that benchmarks multiple deep convolutional neural network (CNN) architectures for multi-class classification of chest X-ray images into COVID-19, Viral Pneumonia, and Normal classes using the COVID-19 Radiography Database from Kaggle. The code implements a full pipeline from dataset download and stratified splitting to data augmentation, model training, and test-set evaluation across several state-of-the-art image classification backbones.

## Project Overview
The project automatically downloads the COVID-19 Radiography Database (COVID-19_Radiography_Dataset) using `opendatasets`, and restructures it into train/validation/test folders for three classes: COVID, Normal, and Viral Pneumonia. Chest X-ray images are loaded, visualized, and then fed into Keras `ImageDataGenerator` pipelines with model-specific preprocessing and light augmentations (horizontal flip, small rotations, zoom, brightness jitter) to simulate realistic variability while preserving diagnostic structure.

### Dataset
- Source: COVID-19 Radiography Database (Kaggle)  
- Classes:
  - COVID  
  - Normal  
  - Viral Pneumonia  
- Splits:
  - Stratified split into 70% train, 15% validation, 15% test using `train_test_split` to preserve class balance across splits  
- Input processing:
  - Images resized to 224×224 RGB, loaded via `flow_from_directory` for each split  

## Methodology

### Data preprocessing & augmentation
- Framework: `ImageDataGenerator` with model-specific preprocessing functions (DenseNet, ResNet50, MobileNetV2, EfficientNetB0)  
- Augmentations:
  - Horizontal flips  
  - Rotation up to 10°  
  - Zoom up to 10%  
  - Brightness range [0.9, 1.1]  

This setup aims to improve generalization while respecting the medical nature of chest radiographs.

### Model architectures
Four transfer learning baselines are implemented and fine-tuned:

- **DenseNet169**  
  - Pretrained on ImageNet, `include_top=False`, global average pooling  
  - Final Dense layer with 3-way softmax for COVID/Normal/Viral Pneumonia  

- **ResNet50**  
  - Pretrained ImageNet backbone, `include_top=False`, global average pooling  
  - Single Dense softmax layer (3 units) as classifier head  

- **MobileNetV2**  
  - Lightweight backbone with `include_top=False` and `GlobalAveragePooling2D` before the 3-unit softmax head  

- **EfficientNetB0**  
  - EfficientNetB0 feature extractor, `include_top=False` with global average pooling, followed by a 3-way softmax classifier  

All models are fine-tuned end-to-end (`trainable=True`) using the same optimizer and training setup for a fair comparison.

### Training setup
- Loss: Categorical cross-entropy  
- Optimizer: Adam with learning rate 1e-4  
- Batch size: 32, input size 224×224×3  
- Epochs: up to 20 with callbacks:
  - EarlyStopping (monitor `val_loss`, patience=5, restore best weights)  
  - ReduceLROnPlateau (factor 0.5, patience=2, minimum LR 1e-7)  
  - ModelCheckpoint (saving best model for each backbone based on validation loss)  

## Evaluation
After training, the best checkpoint for each model (DenseNet169, ResNet50, MobileNetV2, EfficientNetB0) is reloaded and evaluated on the held-out test set. For each architecture, test loss and accuracy are reported, allowing a direct comparison of performance across backbones under identical data splits and preprocessing pipelines.
