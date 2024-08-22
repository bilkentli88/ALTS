import torch
from skorch import NeuralNetClassifier

class ShapeletRegularizedNet(NeuralNetClassifier):
    """
    A custom Skorch NeuralNetClassifier that incorporates additional regularization terms specific to shapelet-based models.

    This class extends the Skorch NeuralNetClassifier to include three types of regularization:
    - L2 regularization on the learned shapelets (prototypes).
    - L2 regularization on the weights of the linear layers.
    - Fused Lasso regularization on the shapelets to encourage smoothness.

    Args:
        lambda_prototypes (float, optional): Regularization strength for the L2 regularization on shapelets. Defaults to 0.10.
        lambda_linear_params (float, optional): Regularization strength for the L2 regularization on linear layer parameters. Defaults to 0.20.
        lambda_fused_lasso (float, optional): Regularization strength for the Fused Lasso regularization on shapelets. Defaults to 0.10.
        *args: Positional arguments passed to the Skorch NeuralNetClassifier.
        **kwargs: Keyword arguments passed to the Skorch NeuralNetClassifier.
    """
    def __init__(self, *args, lambda_prototypes=0.10,
                 lambda_linear_params=0.20,
                 lambda_fused_lasso=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_prototypes = float(lambda_prototypes)
        self.lambda_linear_params = float(lambda_linear_params)
        self.lambda_fused_lasso = float(lambda_fused_lasso)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """
        Calculates the total loss, which includes the standard classification loss and additional regularization terms.

        Args:
            y_pred (torch.Tensor): The predicted output from the model.
            y_true (torch.Tensor): The ground truth labels.
            X (torch.Tensor, optional): The input data. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The total loss, including classification loss and regularization terms.

        Raises:
            ValueError: If NaN or Inf values are detected in the total loss calculation.
        """
        # Calculate the softmax cross-entropy loss
        loss_softmax = super().get_loss(y_pred, y_true, X=X, training=training)

        # Get the device where y_pred resides (CPU/GPU)
        device = y_pred.device

        # L2 regularization on the prototypes (shapelets)
        loss_prototypes = torch.norm(self.module_.prototypes.to(device), p=2)

        # L2 regularization on the linear layer parameters
        loss_weight_reg = torch.tensor(0.0).to(device)
        for param in self.module_.linear1.parameters():
            loss_weight_reg += param.norm(p=2).sum()

        # Fused Lasso regularization to enforce smoothness on the shapelets
        fused_lasso_reg = torch.sum(torch.abs(self.module_.prototypes.to(device)[:, 1:] - self.module_.prototypes.to(device)[:, :-1]))

        # Ensure regularization parameters are on the correct device and are tensors
        lambda_prototypes = torch.tensor(self.lambda_prototypes, dtype=torch.float32, device=device)
        lambda_fused_lasso = torch.tensor(self.lambda_fused_lasso, dtype=torch.float32, device=device)
        lambda_linear_params = torch.tensor(self.lambda_linear_params, dtype=torch.float32, device=device)

        # Compute the total loss by combining all components
        total_loss = (loss_softmax + lambda_prototypes * loss_prototypes +
                      lambda_linear_params * loss_weight_reg + lambda_fused_lasso * fused_lasso_reg)

        # Check for NaN or Inf values in the total loss and raise an error if detected
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError("NaN or Inf detected in total loss calculation")

        return total_loss
