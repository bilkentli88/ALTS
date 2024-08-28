# ALTS (Advanced Learning of Time Series Shapelets)

## 1. Introduction

ALTS (Advanced Learning of Time Series Shapelets) is a time series classification method based on the shapelet paradigm, implemented using the PyTorch framework. The method is designed to identify and utilize representative subsequences (shapelets) within time series data to perform accurate and interpretable classification. ALTS leverages a neural network architecture that learns shapelets and uses various distance and similarity measures to distinguish between different classes.

## 2. Modules Overview

ALTS is composed of several modules, each responsible for different aspects of the model's functionality:

- **`get_dataset_names`**: Handles dataset selection based on user input, enabling flexibility in choosing which datasets to process.
- **`process_datasets`**: Loads, normalizes, augments, and converts time series data into bags (overlapping subsequences), preparing it for shapelet discovery.
- **`optimize_shapelets`**: Conducts hyperparameter optimization using Optuna, evaluating model performance through cross-validation, and identifying the best-performing hyperparameters.
- **`generate_shapelets`**: Defines the neural network architecture for learning shapelets, calculating distances and similarities, and performing classification.
- **`regularize_shapelets`**: Extends the Skorch NeuralNetClassifier to include additional regularization terms specific to shapelet-based models.
- **`get_train_results`**: Coordinates the overall training process by using the best hyperparameters across multiple datasets, saving the results, and logging the outcomes.
- **`get_test_results`**: Evaluates the best hyperparameters on the test set, ensuring that the model generalizes well to unseen data.
- **`plot_shapelets`**: Visualizes the most discriminative shapelets and their best matching time series segments, providing insights into the model's decision-making process.
- **`train_and_save_the_model`**: An auxiliary module used by `plot_shapelets` to train a model on a specific dataset (like BirdChicken) and save the trained model for shapelet visualization.

## 3. Model Architecture and Implementation

### 3.1 Shapelet Generation (`generate_shapelets`)

The core of ALTS lies in its ability to generate shapelets from time series data. The `ShapeletGeneration` class in the `generate_shapelets` module defines a neural network that learns shapelets and uses them for classification. This module handles:

- **Shapelet Initialization**: Shapelets are initialized as learnable parameters within the network, allowing them to be optimized during training.
- **Distance and Similarity Measures**: The model calculates pairwise Euclidean distances and cosine similarities between input time series subsequences and the learned shapelets.
- **Feature Extraction**: Based on user-specified features (min, max, mean, cos), the model extracts relevant features from the distance and similarity measures.
- **Classification**: A fully connected neural network is used to classify time series instances based on the extracted features.

### 3.2 Regularization (`regularize_shapelets`)

To prevent overfitting and encourage smoothness in the learned shapelets, the `regularize_shapelets` module extends the Skorch NeuralNetClassifier to include:

- **L2 Regularization on Shapelets**: Penalizes large values in the learned shapelets, encouraging simpler, more interpretable patterns.
- **L2 Regularization on Linear Layers**: Applies regularization to the weights of the linear layers in the classifier.
- **Fused Lasso Regularization**: Promotes smoothness in the shapelets by penalizing differences between consecutive elements within each shapelet.

### 3.3 Hyperparameter Optimization (`optimize_shapelets`)

Hyperparameter optimization is performed using the Optuna framework, as defined in the `optimize_shapelets` module. This module:

- **Searches for Optimal Hyperparameters**: Uses cross-validation to evaluate different hyperparameter combinations, such as `bag_size`, `n_prototypes`, `lambda_fused_lasso`, and `lambda_prototypes`.
- **Identifies the Best Hyperparameters**: The best-performing hyperparameters are identified and can be used to train the final model.

### 3.4 Data Processing (`process_datasets`)

The `process_datasets` module prepares the data for shapelet generation by:

- **Loading Datasets**: Loads training and testing data from files.
- **Normalization**: Standardizes the time series data to have zero mean and unit variance.
- **Data Augmentation**: Adds Gaussian noise to the training data, increasing the size and variability of the dataset.
- **Bag Conversion**: Converts time series into bags of overlapping subsequences, which are used for shapelet learning.

### 3.5 Visualization and Specialized Training (`plot_shapelets` and `train_and_save_the_model`)

- **plot_shapelets**: This module is used to visualize the most discriminative shapelets and their corresponding time series segments. It is particularly useful for understanding the model's decision-making process.
- **train_and_save_the_model**: An auxiliary module used by `plot_shapelets` to train a model on a specific dataset (like BirdChicken) with the best hyperparameters and save the trained model for shapelet visualization.

## 4. Training and Evaluation

The `get_train_results` and `get_test_results` modules serve as the main scripts for training and evaluating the ALTS model:

- **`get_train_results`**: This script orchestrates the training process using the best hyperparameters identified during the optimization phase. It saves the trained model results and logs the outcomes for multiple datasets.
- **`get_test_results`**: After training, this script evaluates the model on the test set, ensuring that the selected hyperparameters generalize well to unseen data.

## 5. Visualization of Shapelets (`plot_shapelets`)

The `plot_shapelets` module provides an interface for visualizing the most discriminative shapelets learned by the model:

- **Shapelet Extraction**: Extracts the shapelets from the trained model and identifies the most descriptive ones for each class.
- **Time Series Matching**: Finds the best matching time series segments for each shapelet, highlighting the portions of the data that the shapelet corresponds to.
- **Visualization**: Plots the shapelets and their corresponding time series segments, allowing users to interpret the model's classification decisions.

## 6. Conclusion

ALTS is a powerful tool for time series classification, combining the interpretability of shapelet-based methods with the flexibility of neural networks. Its modular design allows for easy customization and extension, making it suitable for a wide range of time series analysis tasks. The accompanying visualizations further enhance the model's interpretability, providing valuable insights into the patterns that drive classification decisions.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ALTS.git

# Navigate to the project directory
cd ALTS

# Install the required dependencies
pip install -r requirements.txt
