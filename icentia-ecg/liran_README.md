# ECG Data Processing and Analysis

## Overview

This repository comprises a collection of Python scripts tailored for Electrocardiogram (ECG) data processing and analysis. These scripts cover a range of tasks, including data preparation, feature extraction, autoencoder training, and utility functions, facilitating in-depth exploration and analysis of ECG datasets.

## File Descriptions

### 1. prepare_emb_cf.py

This script specializes in preparing embeddings from ECG data, allowing users to choose from various encoding methods. It takes pre-existing embeddings as input, applies user-specified encoding techniques, and outputs the results to a new CSV file. This is particularly useful for exploring different feature representations.

### 2. prepare_emb_enc.py

The purpose of this script is to extract embeddings using an autoencoder model. By loading a pre-trained autoencoder, it encodes input data and saves the resulting embeddings to a CSV file. This script is valuable for leveraging unsupervised learning to uncover meaningful patterns in ECG data.

### 3. prepare_emb.py

Designed for flexibility, this script prepares embeddings from ECG datasets with customizable parameters such as data path, frame length, and segment length. The resulting embeddings are saved to a CSV file, providing a versatile tool for diverse ECG data preprocessing needs.

### 4. prepare_labels.py

This script focuses on generating labels for ECG data, considering segment and frame characteristics. It produces labeled data in a CSV file, offering insights into different classes present in the dataset. The script also provides options for label relabeling, enhancing the interpretability of results.

### 5. prepare_pca.py

For dimensionality reduction, this script employs Principal Component Analysis (PCA) on batched ECG data. It processes the data and saves PCA models with varying numbers of components. PCA is a powerful tool for condensing information while preserving key features, aiding in data exploration.

### 6. train_autoencoder.py

Central to deep learning, this script trains an autoencoder model using PyTorch. With functionalities for data loading, model initialization, and a customizable training loop, it supports the exploration of complex relationships within ECG data. The script intelligently saves the best model based on validation loss.

### 7. utils.py

A utility script, `utils.py` contains functions and constants shared across multiple scripts. It includes functions for creating indices, counting lines in files, and loading subsets of data. This centralization promotes code reuse and maintainability.

## Usage

To effectively utilize these scripts:

1. Ensure that necessary dependencies are installed (NumPy, Pandas, PyTorch, etc.).
2. Adjust script-specific parameters, such as file paths, model configurations, and training settings.
3. Execute the scripts in the specified order for comprehensive data preparation, model training, and analysis.

Feel free to explore, customize, and extend these scripts based on your specific ECG dataset and analysis requirements.