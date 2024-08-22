# ALTS (Advanced Learning Time Series Shapelets): Shapelet-based Learning for Time Series Classification

## Overview

ALTS is a PyTorch-based implementation of a time series classification method that employs the shapelet paradigm. It utilizes a neural network to automatically discover discriminative shapelets (subsequences) from time series data, enabling accurate and interpretable classification.

## Core Concepts

- **Shapelets**: Subsequences within time series data that exhibit distinctive patterns associated with specific classes.
- **Neural Network Architecture**: Fully connected neural network to learn shapelets and perform classification.
- **Distance and Similarity Measures**: Calculates multiple features between input time series instances and learned shapelets.
- **Regularization**: Incorporates L2 and Fused Lasso regularization.
- **Hyperparameter Optimization**: Utilizes Optuna for automated tuning.

## Modules

1. **get_dataset_names**
   - Retrieves dataset names based on command-line arguments.
   - Key function: `get_database_list_from_arguments(sys_argv)`

2. **process_datasets**
   - Handles loading, preprocessing, and augmenting time series datasets.
   - Key functions: `load_dataset()`, `normalize()`, `augment_data()`, `convert_to_bags()`

3. **optimize_shapelets**
   - Implements core ALTS functionality, including shapelet discovery, model training, evaluation, and hyperparameter optimization.
   - Key functions: `get_skorch_regularized_classifier()`, `evaluate_model_performance()`, `find_best_hyper_params_optuna_search()`

4. **generate_shapelets**
   - Defines the `ShapeletGeneration` class, encapsulating the neural network architecture.
   - Key methods: `pairwise_distances()`, `cosine_similarity()`, `get_output_from_prototypes()`, `forward()`

5. **regularize_shapelets**
   - Provides the `ShapeletRegularizedNet` class, extending Skorch's NeuralNetClassifier to incorporate regularization.
   - Key method: `get_loss()`

6. **get_train_results**
   - Executes hyperparameter optimization using Optuna for a given list of datasets.

7. **get_test_results**
   - Evaluates the best hyperparameters found during the search phase on the test set of each dataset.

## Hyperparameters

- `bag_size`: Length of the candidate shapelet
- `n_prototypes`: Number of shapelets created
- `lambda_fused_lasso`: Regularization parameter for Fused Lasso Regularization
- `lambda_prototypes`: Regularization parameter for L2 Regularization on shapelets
- `lambda_linear_params`: Regularization parameter for L2 regularization on linear layer parameters (default: 0.20)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 0.0001)
- `dropout_rate`: Dropout rate for regularization (default: 0.60)
- `STRIDE_RATIO`: Ratio of stride to bag_size for creating bags (default 0.01)
- `CV_COUNT`: Number of cross-validation folds (default 5)
- `FEATURES_TO_USE_STR`: Comma-separated string specifying features to use (default "min,max,mean,cos")

## Additional Considerations

- Utilizes stratified k-fold cross-validation for model evaluation and hyperparameter tuning
- Employs early stopping during training to prevent overfitting
- Uses `set_seed` function to ensure reproducibility of results

## Installation

To get started with ALTS, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/ALTS.git
cd ALTS
pip install -r requirements.txt


Usage
Preprocessing Data
Before training models, you need to preprocess your time series data. ALTS includes utilities for loading, normalizing, and augmenting datasets:
from process_datasets import load_dataset

# Load and preprocess the dataset
train_features, train_labels, test_features, test_labels = load_dataset('YourDatasetName')


