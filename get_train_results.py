import sys
import os
import get_dataset_names
import optimize_shapelets
from set_seed import set_seed

# Set the seed for reproducibility
set_seed(88)

if __name__ == "__main__":
    """
    Main execution block of the script.
    Retrieves the list of datasets, performs hyperparameter optimization, and saves the results.
    """

    # Retrieve the list of datasets to work on based on command-line arguments
    dataset_list = get_dataset_names.get_dataset_list_from_arguments(sys.argv)

    # Print the list of datasets to be processed
    print("Datasets to work on:")
    print(dataset_list)

    # Define the type of hyperparameter search to use
    search_type = "OptunaSearch"

    # Create the "Results" directory if it doesn't exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    # Iterate over each dataset in the list
    for dataset_name in dataset_list:
        # Construct the filename for storing search results
        search_filename = optimize_shapelets.get_filename_output_for_search(
            search_type=search_type,
            dataset_name=dataset_name,
            n_trials=100,
            max_epoch=1000
        )

        # Check if results already exist for this dataset, skip if they do
        if os.path.exists(search_filename):
            print(f"Results for {dataset_name} already exist. Skipping...")
            continue

        # Find the best hyperparameters and save results for the current dataset using the specified search type
        model = optimize_shapelets.find_result_for_one(
            search_type=search_type,
            dataset_name=dataset_name,
            n_trials=100,  # Number of Optuna trials
            search_max_epoch=1000  # Maximum epochs for each Optuna trial
        )

        # Save the trained model
        if model is not None:
            model_save_path = os.path.join('Results', f'{dataset_name}_best_model.pth')
            optimize_shapelets.save_trained_model(model, model_save_path)
            print(f"Trained model for {dataset_name} saved at {model_save_path}")
        else:
            print(f"Error: Model for {dataset_name} was not trained successfully.")
