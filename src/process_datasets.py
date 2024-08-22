import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

# Main folder containing datasets
DATASET_MAIN_FOLDER = "Datasets"

# Determine device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset_name):
    """
    Loads a dataset by calling the load_data function with the specified dataset name
    and the main dataset folder.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        tuple: A tuple containing:
            - train_features (torch.Tensor): Normalized and augmented training features.
            - train_labels (torch.Tensor): One-hot encoded training labels.
            - test_features (torch.Tensor): Normalized testing features.
            - test_labels (torch.Tensor): One-hot encoded testing labels.
    """
    return load_data(DATASET_MAIN_FOLDER, dataset_name)


def load_data(folder, dataset, extensions=(".txt", "")):
    """
    Loads training and testing data for a given dataset from a specified folder.

    Handles file loading, data preprocessing (normalization, one-hot encoding),
    and data augmentation.

    Args:
        folder (str): Path to the folder containing the dataset files.
        dataset (str): Name of the dataset.
        extensions (tuple, optional): File extensions of the dataset files (default is (".txt", "")).

    Returns:
        tuple: A tuple containing:
            - train_features (torch.Tensor): Normalized and augmented training features.
            - train_labels (torch.Tensor): One-hot encoded training labels.
            - test_features (torch.Tensor): Normalized testing features.
            - test_labels (torch.Tensor): One-hot encoded testing labels.

    Raises:
        ValueError: If NaN values are found in the dataset features.
        FileNotFoundError: If the dataset files are not found with the provided extensions.
    """

    train_file_base = f"{dataset}_TRAIN"
    test_file_base = f"{dataset}_TEST"

    # Attempt to load the files with the provided extensions
    for ext in extensions:
        train_file = os.path.join(folder, train_file_base + ext)
        test_file = os.path.join(folder, test_file_base + ext)
        if os.path.exists(train_file) and os.path.exists(test_file):
            break
    else:
        raise FileNotFoundError(f"Dataset files for {dataset} not found with provided extensions.")

    # Load the dataset files
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    # Split data into features and labels
    train_features = train_data[:, 1:]
    test_features = test_data[:, 1:]

    # Check for NaN values in features
    if np.any(np.isnan(train_features)) or np.any(np.isnan(test_features)):
        raise ValueError("NaN values found in the dataset features. Please check the data.")

    # Normalize the features
    train_features = normalize(train_features)
    test_features = normalize(test_features)

    # One-hot encode the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_data[:, 0].reshape(-1, 1))

    train_labels = enc.transform(train_data[:, 0].reshape(-1, 1)).toarray()
    test_labels = enc.transform(test_data[:, 0].reshape(-1, 1)).toarray()

    # Apply noise augmentation to the training features and labels
    train_features, train_labels = augment_data(train_features, train_labels)

    return (torch.from_numpy(train_features).float().to(device), torch.from_numpy(train_labels).int().to(device),
            torch.from_numpy(test_features).float().to(device), torch.from_numpy(test_labels).int().to(device))


def normalize(features):
    """
    Normalizes the input features by subtracting the mean and dividing by the standard deviation
    along each time series.

    Handles cases where the standard deviation is zero by setting it to a small value
    to avoid division by zero.

    Args:
        features (np.ndarray): The input features to be normalized.

    Returns:
        np.ndarray: The normalized features.
    """
    mean = features.mean(axis=1).reshape(-1, 1)
    std = features.std(axis=1).reshape(-1, 1)
    std[std == 0] = 1e-8  # Handle division by zero by setting std to a small value
    return (features - mean) / std


def augment_data(features, labels, noise_level=0.01):
    """
    Augments the training data by adding Gaussian noise to the features.

    Doubles the size of the training set by adding noisy versions of the original features.

    Args:
        features (np.ndarray): The original training features.
        labels (np.ndarray): The original training labels.
        noise_level (float, optional): The standard deviation of the Gaussian noise to add (default is 0.01).

    Returns:
        tuple: A tuple containing:
            - augmented_features (np.ndarray): The augmented training features.
            - augmented_labels (np.ndarray): The augmented training labels.
    """
    augmented_data = []
    augmented_labels = []
    for feature, label in zip(features, labels):
        augmented_data.append(feature)
        augmented_labels.append(label)
        augmented_data.append(add_noise(feature, noise_level))
        augmented_labels.append(label)
    return np.array(augmented_data), np.array(augmented_labels)


def add_noise(data, noise_level=0.01):
    """
    Adds Gaussian noise to the input data with a specified noise level (standard deviation).

    Args:
        data (np.ndarray): The original data to which noise will be added.
        noise_level (float, optional): The standard deviation of the Gaussian noise (default is 0.01).

    Returns:
        np.ndarray: The data with added Gaussian noise.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def convert_to_bags(data, bag_size, stride_ratio):
    """
    Converts time series data into bags (overlapping subsequences) of a specified size and stride.

    Args:
        data (torch.Tensor or np.ndarray): The time series data.
        bag_size (int): The desired size of each bag (subsequence).
        stride_ratio (float): The ratio of stride to bag_size, controlling the overlap between bags.

    Returns:
        torch.Tensor: The converted data as bags (shape: [num_instances, num_bags, bag_size, num_features]).

    Raises:
        ValueError: If no bags are created due to invalid input data or parameters.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    bag_size = int(bag_size)
    stride = int(max(round(stride_ratio * bag_size), 1))  # Ensure stride is at least 1
    bags = []

    for i in range(data.shape[0]):
        instance = []
        size = data[i].shape[0]
        window = int(bag_size)
        while True:
            instance.append(data[i][window - bag_size: window])
            window += stride
            if window >= size:
                window = size
                instance.append(data[i][window - bag_size: window])
                break
        bags.append(np.array(instance))

    if len(bags) == 0:
        raise ValueError("No bags created. Please check the input data and parameters.")

    return torch.from_numpy(np.array(bags)).float()


def get_bag_size(dataset_name, bag_ratio):
    """
    Calculates the bag size (length of candidate shapelet) based on the length of time series in the dataset
    and the specified bag ratio.

    Args:
        dataset_name (str): The name of the dataset.
        bag_ratio (float): The ratio of bag size to the length of time series.

    Returns:
        int: The calculated bag size.
    """
    train_file = os.path.join(DATASET_MAIN_FOLDER, f"{dataset_name}_TRAIN.txt")
    train_data = np.loadtxt(train_file)
    time_series_size = train_data.shape[1] - 1  # Exclude the label column
    bag_size = int(time_series_size * bag_ratio)
    return bag_size
