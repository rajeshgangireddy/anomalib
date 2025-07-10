# Dinomaly: Vision Transformer-based Anomaly Detection with Feature Reconstruction

This is the implementation of the Dinomaly model based on the [original implementation](https://github.com/guojiajeremy/Dinomaly).

Model Type: Segmentation

## Description

Dinomaly is a Vision Transformer-based anomaly detection model that uses an encoder-decoder architecture for feature reconstruction. The model leverages pre-trained DINOv2 Vision Transformer features and employs a reconstruction-based approach to detect anomalies by comparing encoder and decoder features.

### Feature Extraction

Features are extracted from multiple intermediate layers of a pre-trained DINOv2 Vision Transformer encoder. The model typically uses features from layers 2-9 for base models, providing multi-scale feature representations that capture both low-level and high-level semantic information.

### Architecture

The Dinomaly model consists of three main components:

1. **DINOv2 Encoder**: Pre-trained Vision Transformer that extracts multi-layer features
2. **Bottleneck MLP**: Compresses the multi-layer features before reconstruction
3. **Vision Transformer Decoder**: Reconstructs the compressed features back to the original feature space

### Anomaly Detection

Anomaly detection is performed by computing cosine similarity between encoder and decoder features at multiple scales. The model generates anomaly maps by analyzing the reconstruction quality of features, where poor reconstruction indicates anomalous regions. Both anomaly detection (image-level) and localization (pixel-level) are supported.

## Usage

`anomalib train --model Dinomaly --data MVTecAD --data.category <category>`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| Dinomaly       |   -   |   -    |   -   |    -    |   -   |   -   |   -    |   -   |    -    |    -     |     -     |   -   |   -   |     -      |     -      |   -    |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| Dinomaly       |   -   |   -    |   -   |    -    |   -   |   -   |   -    |   -   |    -    |    -     |     -     |   -   |   -   |     -      |     -      |   -    |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :--: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| Dinomaly       |   -   |   -    |   -   |    -    |   -  |  -   |   -    |   -   |    -    |    -     |     -     |   -   |   -   |     -      |     -      |   -    |
