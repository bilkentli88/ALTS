import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from process_datasets import convert_to_bags

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShapeletGeneration(nn.Module):
    """
    A PyTorch module for generating and using shapelets for time series classification.

    This module defines a neural network architecture that learns shapelets (representative subsequences)
    from time series data and uses them for classification. It calculates various distance and similarity
    measures between input time series and the learned shapelets, and then feeds these features into a
    fully connected neural network for classification.
    """

    def __init__(self,
                 n_prototypes,
                 bag_size,
                 n_classes,
                 stride_ratio,
                 features_to_use_str,
                 dropout_rate=0.60,
                 dataset_name=None):
        """
        Initializes the ShapeletGeneration module.

        Args:
            n_prototypes (int): The number of shapelets (prototypes) to learn.
            bag_size (int): The length of each shapelet.
            n_classes (int): The number of classes in the classification task.
            stride_ratio (float): The ratio of stride to bag_size for creating bags (overlapping subsequences).
            features_to_use_str (str): A comma-separated string specifying the features to use
                                       (e.g., "min,max,mean,cos").
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.60.
            dataset_name (str, optional): The name of the dataset (currently unused).
        """
        super(ShapeletGeneration, self).__init__()

        # Parse the features to use from the input string
        features_to_use = features_to_use_str.split(",")

        # Initialize prototypes (shapelets) as learnable parameters
        self.prototypes = nn.Parameter(torch.randn((1, int(n_prototypes), int(bag_size))) * 0.01)
        # Move prototypes to the appropriate device
        self.prototypes = self.prototypes.to(device)

        # Store configuration parameters
        self.n_p = n_prototypes
        self.bag_size = bag_size
        self.N = n_classes
        self.stride_ratio = stride_ratio
        self.features_to_use = features_to_use
        self.dropout_rate = dropout_rate

        # Calculate input and hidden layer sizes for the classifier network
        input_size = len(self.features_to_use) * n_prototypes
        hidden_size = int(input_size * 2.0)

        # Define the layers of the classifier network
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, n_classes)

    def pairwise_distances(self, x, y):
        """
        Calculates the pairwise Euclidean distances between two sets of time series or subsequences.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, seq_len).
            y (torch.Tensor): Tensor of shape (1, num_prototypes, seq_len) representing the prototypes/shapelets.

        Returns:
            torch.Tensor: Pairwise distances tensor of shape (batch_size, num_instances, num_prototypes).
        """
        # Ensure both tensors are on the same device
        x, y = x.to(self.prototypes.device), y.to(self.prototypes.device)

        # Calculate norms for efficient distance computation
        x_norm = (x.norm(dim=2)[:, :, None]).float()
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = (y.norm(dim=2)[:, None])

        # Expand y_t to match the batch size of x
        y_t = torch.cat([y_t] * x.shape[0], dim=0)

        # Compute pairwise distances using the expanded tensors
        dist = x_norm + y_norm - 2.0 * torch.bmm(x.float(), y_t)

        # Clamp distances to ensure non-negativity and avoid numerical issues
        return torch.clamp(dist, 0.0, np.inf).to(self.prototypes.device)

    def cosine_similarity(self, x, y):
        """
        Calculates the cosine similarity between two sets of time series or subsequences.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, seq_len).
            y (torch.Tensor): Tensor of shape (1, num_prototypes, seq_len) representing the prototypes/shapelets.

        Returns:
            torch.Tensor: Cosine similarity tensor of shape (batch_size, num_instances, num_prototypes).
        """
        # Ensure both tensors are on the same device
        x, y = x.to(self.prototypes.device), y.to(self.prototypes.device)

        # Normalize the input tensors along the feature dimension
        x_normalized = F.normalize(x, p=2, dim=2)
        y_normalized = F.normalize(y, p=2, dim=2)

        # Transpose y_normalized for efficient matrix multiplication
        y_t = y_normalized.permute(0, 2, 1).contiguous()

        # Expand y_t to match the batch size of x
        y_t = torch.cat([y_t] * x.shape[0], dim=0)

        # Compute cosine similarity using batch matrix multiplication
        cos_sim = torch.bmm(x_normalized, y_t)

        return cos_sim.to(self.prototypes.device)

    def layer_norm(self, feature):
        """
        Applies layer normalization to the input feature tensor.

        Args:
            feature (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The layer-normalized feature tensor.
        """
        mean = feature.mean(keepdim=True, dim=-1)
        var = ((feature - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + 1e-8).sqrt()
        y = (feature - mean) / std
        return y

    def get_output_from_prototypes(self, batch_inp):
        """
        Calculates features (distances and similarities) between input time series bags and prototypes.

        Args:
            batch_inp (torch.Tensor): Input time series bags of shape (batch_size, num_bags, bag_size, num_features).

        Returns:
            torch.Tensor: Calculated features based on the specified features to use.
        """
        # Calculate pairwise distances and cosine similarities
        dist = self.pairwise_distances(batch_inp, self.prototypes)
        cos_sim = self.cosine_similarity(batch_inp, self.prototypes)

        l_features = []

        # Extract features based on the configuration
        if "min" in self.features_to_use:
            min_dist = self.layer_norm(dist.min(dim=1)[0])
            l_features.append(min_dist)
        if "max" in self.features_to_use:
            max_dist = self.layer_norm(dist.max(dim=1)[0])
            l_features.append(max_dist)
        if "mean" in self.features_to_use:
            mean_dist = self.layer_norm(dist.mean(dim=1))
            l_features.append(mean_dist)
        if "cos" in self.features_to_use:
            cos_sim_feature = self.layer_norm(cos_sim.max(dim=1)[0])
            l_features.append(cos_sim_feature)

        if len(l_features) == 0:
            raise ValueError("No features to use in get_output_from_prototypes. Check the features_to_use configuration.")

        all_features = torch.cat(l_features, dim=1)
        return all_features

    def forward(self, x):
        """
        Defines the forward pass of the ShapeletGeneration module.

        Args:
            x (torch.Tensor): Input time series data of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: The output logits of the classifier network, representing the predicted class probabilities.
        """
        # Convert input time series into bags (overlapping subsequences)
        x_bags = convert_to_bags(x, self.bag_size, self.stride_ratio)

        # Ensure x_bags are on the same device as prototypes
        x_bags = x_bags.to(self.prototypes.device)

        # Calculate features from the prototypes and input bags
        all_features = self.get_output_from_prototypes(x_bags)

        # Pass the features through the classifier network
        out = self.linear1(all_features)
        out = self.selu1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        out = self.output_layer(out)

        return out
