import os
import torch
import optimize_shapelets
from set_seed import set_seed

# Set the seed for reproducibility
set_seed(88)

def train_and_save_the_model(dataset_name, search_type, n_trials, search_max_epoch):
    """
    Performs hyperparameter optimization, saves the best hyperparameters, trains, and saves the model.

    Args:
        dataset_name (str): The name of the dataset.
        search_type (str): The type of hyperparameter search to use.
        n_trials (int): The number of trials for hyperparameter optimization.
        search_max_epoch (int): The maximum number of epochs to use in the search phase.
    """
    # Create the "Results" directory if it doesn't exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    # Construct the filename for storing search results
    search_filename = optimize_shapelets.get_filename_output_for_search(
        search_type=search_type,
        dataset_name=dataset_name,
        n_trials=n_trials,
        max_epoch=search_max_epoch
    )

    # Check if results already exist for this dataset
    if os.path.exists(search_filename):
        print(f"Results for {dataset_name} already exist. Skipping optimization...")
        return  # Skip further processing if results already exist

    # If results don't exist, perform hyperparameter search and save them
    print(f"Running hyperparameter optimization for dataset: {dataset_name}")
    result_dict = optimize_shapelets.find_result_for_one(
        search_type=search_type,
        dataset_name=dataset_name,
        n_trials=n_trials,
        search_max_epoch=search_max_epoch
    )
    print(f"Saved hyperparameters for {dataset_name} in {search_filename}")

    # Train and save the model
    print(f"Training and saving the model for dataset: {dataset_name}")

    # Use the best hyperparameters from the result_dict directly
    nn_shapelet_generator = optimize_shapelets.ShapeletNN(
        n_prototypes=result_dict['n_prototypes'],
        bag_size=result_dict['bag_size'],
        n_classes=result_dict['n_classes'],
        stride_ratio=optimize_shapelets.STRIDE_RATIO,
        features_to_use_str=optimize_shapelets.FEATURES_TO_USE_STR
    ).to(optimize_shapelets.device)

    net = optimize_shapelets.get_regularized_classifier(
        nn_shapelet_generator=nn_shapelet_generator,
        max_epochs=search_max_epoch,
        use_early_stopping=True,
        lambda_prototypes=result_dict['lambda_prototypes'],
        lambda_fused_lasso=result_dict['lambda_fused_lasso']
    )

    # Load the dataset and ensure y_train is in the correct format
    X_train, y_train, _, _ = optimize_shapelets.load_dataset(dataset_name)

    # Ensure that y_train is a 1D array of labels
    if y_train.ndim > 1:
        y_train = y_train.argmax(axis=1)

    # Fit the model
    net.fit(X_train, y_train)

    # Save the trained model directly using Skorch's save_params method
    model_save_path = os.path.join('Results', f'{dataset_name}_best_model.pth')
    net.save_params(f_params=model_save_path)
    print(f"Trained model for {dataset_name} saved at {model_save_path}")

    # Save the hyperparameters for the model
    hyperparams = {
        'n_prototypes': result_dict['n_prototypes'],
        'bag_size': result_dict['bag_size'],
        'lambda_prototypes': result_dict['lambda_prototypes'],
        'lambda_fused_lasso': result_dict['lambda_fused_lasso'],
        'n_classes': result_dict['n_classes'],
        'stride_ratio': optimize_shapelets.STRIDE_RATIO,
        'features_to_use_str': optimize_shapelets.FEATURES_TO_USE_STR,  # This line ensures consistency
        'max_epochs': search_max_epoch,
        'learning_rate': net.optimizer_.param_groups[0]['lr'],  # Extract learning rate from optimizer
        'dropout_rate': nn_shapelet_generator.dropout_rate  # Extract dropout rate from the instantiated model
    }

    hyperparams_save_path = os.path.join('Results', f'{dataset_name}_hyperparams.pth')
    torch.save(hyperparams, hyperparams_save_path)
    print(f"Hyperparameters for {dataset_name} saved at {hyperparams_save_path}")


if __name__ == "__main__":
    # Example usage for a single dataset
    dataset_name = "BirdChicken"  # Or any other dataset name

    train_and_save_the_model(
        dataset_name=dataset_name,
        search_type="OptunaSearch",
        n_trials=100,
        search_max_epoch=1000
    )
