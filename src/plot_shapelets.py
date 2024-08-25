"""
This module is responsible for loading a trained shapelet model, extracting the most
discriminative shapelets, and visualizing them alongside the best matching time series
segments from the dataset.

The module performs the following main tasks:

Dependencies:
- torch: For neural network operations and GPU acceleration
- numpy: For numerical computations
- matplotlib: For plotting and visualization
- regularize_shapelets: Custom module containing the ShapeletRegularizedNet class
- generate_shapelets: Custom module containing the ShapeletGeneration class
- process_datasets: Custom module for loading datasets
- set_seed: Custom module for setting random seeds for reproducibility

Note: This script assumes that a pre-trained model and its corresponding dataset
are available.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from regularize_shapelets import ShapeletRegularizedNet
from generate_shapelets import ShapeletGeneration
from process_datasets import load_dataset
from set_seed import set_seed

# Set a random seed for reproducibility
set_seed(2019)

# Set device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model parameters
model_save_path = 'Results/BirdChicken_best_model.pth'

# Load the model architecture parameters based on prior training setup
bag_size = 87  # Length of the shapelet from BirdChicken results
n_prototypes = 44  # Number of shapelets from BirdChicken results
lambda_fused_lasso = 2.49E-08  # Fused Lasso regularization parameter
lambda_prototypes = 1.03E-02  # L2 Regularization parameter
n_classes = 2  # BirdChicken is a binary classification problem
stride_ratio = 0.01  # Stride ratio used during model training
features_to_use_str = "min,max,mean,cos"  # Features used in the model

# Recreate the model architecture based on the training setup
shapelet_model = ShapeletGeneration(
    n_prototypes=n_prototypes,
    bag_size=bag_size,
    n_classes=n_classes,
    stride_ratio=stride_ratio,
    features_to_use_str=features_to_use_str,
    dropout_rate=0.60  # Dropout rate as used in the original module
).to(device)

# Initialize the regularized network using the shapelet model
regularized_net = ShapeletRegularizedNet(
    module=shapelet_model,
    max_epochs=1000,  # Epochs used during training (not needed now, just for consistency)
    lr=0.0001,  # Learning rate used during training
    criterion=torch.nn.CrossEntropyLoss,  # Loss function used during training
    optimizer=torch.optim.Adam,  # Optimizer used during training
    iterator_train__shuffle=True,  # Shuffling training data (not needed now)
    device=device,
    lambda_prototypes=lambda_prototypes,
    lambda_fused_lasso=lambda_fused_lasso
)

# Load the saved model parameters
regularized_net.initialize()  # Initialize the Skorch model
regularized_net.load_params(f_params=model_save_path)  # Load the trained model parameters

# Load the dataset
dataset_name = "BirdChicken"
X_train, y_train, _, _ = load_dataset(dataset_name)

# Ensure that the data is in NumPy format and reshape y_train if needed
X_train = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
y_train = y_train.numpy() if hasattr(y_train, 'numpy') else y_train

# Convert y_train to one-dimensional if it is not already
if y_train.ndim > 1:
    y_train = np.argmax(y_train, axis=1)

# Define a function to find and plot the shapelets and best matching instances
def find_and_plot_shapelets_and_matches(model, class_names, X_train, y_train):
    """
    Find the most discriminative shapelets for each class and plot them alongside
    the best matching time series segments from the dataset.

    Args:
    model (ShapeletRegularizedNet): The trained shapelet model
    class_names (list): List of class names
    X_train (np.array): Training data
    y_train (np.array): Training labels

    Returns:
    None (displays the plot)
    """
    # Extract the shapelet prototypes from the trained model
    shapelet_prototypes = model.module_.prototypes.squeeze().detach().cpu().numpy()  # (n_prototypes, bag_size)

    # Number of shapelets assumed to be evenly split between classes
    n_prototypes = shapelet_prototypes.shape[0]
    n_shapelets_per_class = n_prototypes // len(class_names)

    # Placeholder to store results for visualization
    results = []

    for i in range(len(class_names)):
        # Select the shapelets corresponding to the current class
        start_idx = i * n_shapelets_per_class
        end_idx = start_idx + n_shapelets_per_class
        class_shapelets = shapelet_prototypes[start_idx:end_idx]

        # Sum the absolute values of the coefficients for each shapelet
        total_abs_coefficients = np.sum(np.abs(class_shapelets), axis=1)

        # Identify the most descriptive shapelet for this class
        most_descriptive_idx = np.argmax(total_abs_coefficients)
        most_descriptive_shapelet = class_shapelets[most_descriptive_idx]

        # Find the best matching time series segment for this class
        y_train_true = (y_train == i)
        X_train_this_class = X_train[y_train_true]

        # Calculate the distance between the most descriptive shapelet and each possible segment in each time series
        best_match_idx, best_match_start, best_match_distance = None, None, np.inf
        for ts_idx, ts in enumerate(X_train_this_class):
            for start in range(len(ts) - len(most_descriptive_shapelet) + 1):
                segment = ts[start:start + len(most_descriptive_shapelet)]
                distance = np.linalg.norm(segment - most_descriptive_shapelet)
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_idx = ts_idx
                    best_match_start = start

        # Store the best matching time series and its details
        best_matching_ts = X_train_this_class[best_match_idx]
        results.append(
            (best_match_idx, best_match_start, most_descriptive_idx, most_descriptive_shapelet, best_matching_ts))

        # Print out the best matching time series details for interpretability
        print(
            f"Class {i + 1} ({class_names[i]}): Best matching time series is row {best_match_idx} starting at column {best_match_start}.")

    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 16))

    # Colors for different classes and segments
    shapelet_axes = [ax1, ax2]
    timeseries_axes = [ax3, ax4]

    for i in range(len(class_names)):
        match_start = results[i][1]
        shapelet = results[i][3]
        ts = results[i][4]

        # Normalize the shapelet to have zero mean and unit variance
        mean, std = np.mean(shapelet), np.std(shapelet)
        normalized_shapelet = (shapelet - mean) / std if std != 0 else shapelet - mean

        # Apply transformation to shapelet values for the y-axis adjustment
        scaled_shapelet = normalized_shapelet / 10.0

        # Choose contrasting colors for the shapelet and time series
        colors = ['#1f77b4', '#ff7f0e']  # Blue for Bird, Orange for Chicken

        # Plot the scaled shapelet with distinct colors
        shapelet_axes[i].plot(scaled_shapelet, color=colors[i], linestyle='-', linewidth=2)
        shapelet_axes[i].set_title(f"Most Descriptive Shapelet for Class {i + 1} ({class_names[i]})",
                                   fontsize=12)
        shapelet_axes[i].set_xlabel('Time', fontsize=10)
        shapelet_axes[i].set_ylabel('Amplitude', fontsize=10)
        shapelet_axes[i].grid(True)

        # Set the y-axis limits to -1.0 to 1.0
        shapelet_axes[i].set_ylim(-1.0, 1.0)

        # Plot the full time series with the same contrasting colors
        timeseries_axes[i].plot(ts, color=colors[i], alpha=0.5, label='Full time series')

        # Highlight the time series segment that corresponds to the shapelet match
        segment = ts[match_start:match_start + len(shapelet)]
        timeseries_axes[i].plot(range(match_start, match_start + len(shapelet)), segment, color='#BE0032', linewidth=2,
                                linestyle='-', label='Matched segment')  # Red for matched segment
        # Add a shaded region to emphasize the matching segment
        timeseries_axes[i].axvspan(match_start, match_start + len(shapelet), color='#ADFF2F', alpha=0.2)

        # Overlay the scaled shapelet on the time series with a continuous line
        segment_min, segment_max = np.min(segment), np.max(segment)
        segment_range = segment_max - segment_min
        if segment_range != 0:
            # Apply scaling factor to make the shapelet more visible
            scaling_factor = 0.8
            normalized_shapelet_scaled = scaled_shapelet * scaling_factor * segment_range / 2 + (
                    segment_min + segment_max) / 2

            timeseries_axes[i].plot(range(match_start, match_start + len(shapelet)), normalized_shapelet_scaled,
                                    color='#702963', linestyle='-', linewidth=2,
                                    label='Best Shapelet')  # Purple for overlay
        timeseries_axes[i].set_title(f"Class {i + 1} ({class_names[i]}) Time Series",
                                     fontsize=12)
        timeseries_axes[i].set_xlabel('Time', fontsize=10)
        timeseries_axes[i].set_ylabel('Amplitude', fontsize=10)
        timeseries_axes[i].grid(True)
        timeseries_axes[i].legend(fontsize=8, loc='upper right')

    plt.tight_layout(pad=4.0, h_pad=10.0, w_pad=4.0)
    fig.subplots_adjust(top=0.88)

    # Add the title after adjusting the subplots
    fig.suptitle("Most Discriminative Shapelets of the BirdChicken Dataset", fontsize=12, color='black',
                 fontweight='bold', y=0.98)

    plt.show()

# Use the trained model to plot the shapelets and best matching instances
find_and_plot_shapelets_and_matches(regularized_net, ['Bird', 'Chicken'], X_train, y_train)