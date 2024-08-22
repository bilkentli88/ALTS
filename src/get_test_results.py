import sys
import optimize_shapelets
import get_dataset_names

if __name__ == "__main__":
    """
    Main execution block of the script. Retrieves the list of datasets and evaluates the best hyperparameters on the test set.
    """

    # Retrieve the list of datasets to work on based on command-line arguments
    dataset_list = get_dataset_names.get_dataset_list_from_arguments(sys.argv)

    # Iterate over each dataset in the list
    for dataset_name in dataset_list:
        # Evaluate the best hyperparameters found using Optuna search on the test set and save the results
        optimize_shapelets.get_test_results_for_one_dataset(
            search_type="OptunaSearch",  # Type of search used to find the best hyperparameters
            dataset_name=dataset_name,
            n_trials=100,  # Number of trials used in the search phase
            search_max_epoch=1000,  # Maximum epochs used during the training phase
            test_result_max_epoch=1000  # Maximum epochs to use for evaluating the best model on the test set
        )
