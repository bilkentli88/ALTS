import os
import numpy as np
import pandas as pd
import torch
import optuna
import logging
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from skorch.callbacks import EarlyStopping, EpochScoring
from generate_shapelets import ShapeletGeneration as ShapeletNN
from regularize_shapelets import ShapeletRegularizedNet
from process_datasets import augment_data, load_dataset
from optuna import create_study
from optuna.samplers import TPESampler
from set_seed import set_seed
from collections import Counter


# Set up device, seed, and logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(88)  # Set the seed for reproducibility
optuna.logging.set_verbosity(optuna.logging.WARN)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.model_selection._split')

# Constants
STRIDE_RATIO = 0.01
CV_COUNT = 5
FEATURES_TO_USE_STR = "min,max,mean,cos"
LEARNING_RATE = 0.0001


class TrainAccuracy(EpochScoring):
    """
    Custom callback to track training accuracy during model training.

    Args:
        name (str): The name of the scoring metric (default is 'train_acc').
        lower_is_better (bool): Indicates whether lower scores are better (default is False).
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(self, name='train_acc', lower_is_better=False, **kwargs):
        super().__init__(scoring='accuracy', on_train=True, name=name, lower_is_better=lower_is_better, **kwargs)


def get_list_bag_sizes(dataset_name):
    """
    Generates a list of potential bag sizes (candidate shapelet lengths) based on the length of time series
    in the specified dataset and a range of proportions.

    Args:
        dataset_name (str): The name of the dataset for which to generate bag sizes.

    Returns:
        list: A list of potential bag sizes.
    """
    train_file = os.path.join("Datasets", f"{dataset_name}_TRAIN")
    train_data = np.loadtxt(train_file)
    time_series_size = train_data.shape[1] - 1
    proportions = np.arange(0.02, 0.41, 0.01)
    bag_sizes = [int(time_series_size * prop) for prop in proportions if
                 1 <= int(time_series_size * prop) <= time_series_size]
    return list(set(bag_sizes))


def get_regularized_classifier(nn_shapelet_generator, max_epochs, use_early_stopping, lambda_prototypes,
                               lambda_fused_lasso):
    """
    Creates a Skorch neural network classifier with regularization for shapelet learning.

    Args:
        nn_shapelet_generator (ShapeletGeneration): The shapelet generation neural network module.
        max_epochs (int): The maximum number of training epochs.
        use_early_stopping (bool): Whether to use early stopping during training.
        lambda_prototypes (float): Regularization strength for L2 regularization on shapelets.
        lambda_fused_lasso (float): Regularization strength for Fused Lasso regularization on shapelets.

    Returns:
        skorch.NeuralNetClassifier: The configured Skorch classifier.
    """
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStopping(patience=10))

    net = ShapeletRegularizedNet(
        module=nn_shapelet_generator,
        max_epochs=max_epochs,
        lr=LEARNING_RATE,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        callbacks=callbacks,
        device=device,
        lambda_prototypes=lambda_prototypes,
        lambda_fused_lasso=lambda_fused_lasso
    )

    return net


def evaluate_model_performance(dataset_name, bag_size, n_prototypes, max_epoch, phase, lambda_fused_lasso,
                               lambda_prototypes, augment):
    """
    Evaluates the performance of the shapelet-based model on a given dataset using cross-validation.

    Args:
        dataset_name (str): The name of the dataset.
        bag_size (int): The size of the bags (subsequences) used for shapelet discovery.
        n_prototypes (int): The number of shapelets to discover.
        max_epoch (int): The maximum number of training epochs.
        phase (str): Indicates whether it's the "training" or "test" phase.
        lambda_fused_lasso (float): Regularization strength for Fused Lasso.
        lambda_prototypes (float): Regularization strength for L2 regularization.
        augment (bool): Whether to augment the training data.

    Returns:
        float: The average test accuracy across cross-validation folds.
    """
    train, y_train, test, y_test = load_dataset(dataset_name)

    if augment:
        train, y_train = augment_data(train.cpu().numpy(), y_train.cpu().numpy())
        train = torch.from_numpy(train).float().to(device)
        y_train = torch.from_numpy(y_train).int().to(device)

    y_train = y_train if isinstance(y_train,
                                    np.ndarray) else y_train.cpu().numpy()  # Convert tensor to numpy array if needed
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    nn_shapelet_generator = ShapeletNN(
        n_prototypes=n_prototypes,
        bag_size=bag_size,
        n_classes=n_classes,
        stride_ratio=STRIDE_RATIO,
        features_to_use_str=FEATURES_TO_USE_STR
    ).to(device)

    use_early_stopping = (phase == "training")
    net = get_regularized_classifier(nn_shapelet_generator, max_epoch,
                                     use_early_stopping=use_early_stopping,
                                     lambda_prototypes=lambda_prototypes,
                                     lambda_fused_lasso=lambda_fused_lasso)

    class_counts = Counter(y_train_labels)
    min_class_count = min(class_counts.values())
    n_splits = min(CV_COUNT, min_class_count)  # Ensure n_splits doesn't exceed the smallest class count
    skf = StratifiedKFold(n_splits=n_splits)

    test_accuracies = []

    for fold, (train_index, test_index) in enumerate(skf.split(train, y_train_labels)):
        logger.info(f"Fold {fold + 1}/{CV_COUNT} for evaluation")

        X_train, X_test = train[train_index], train[test_index]
        y_train_fold, y_test_fold = y_train_labels[train_index], y_train_labels[test_index]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float().to(device)
        else:
            X_train = X_train.float().to(device)

        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float().to(device)
        else:
            X_test = X_test.float().to(device)

        net.fit(X_train, y_train_fold)
        y_predict = net.predict(X_test)
        y_predict_labels = np.argmax(y_predict, axis=1) if y_predict.ndim > 1 else y_predict
        test_accuracies.append(accuracy_score(y_test_fold, y_predict_labels))

    avg_test_accuracy = np.mean(test_accuracies)
    logger.info(f"Evaluation completed with average test accuracy: {avg_test_accuracy}")

    return avg_test_accuracy

def optuna_objective(trial):
    """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.Trial): An Optuna trial object.

        Returns:
            float: The average accuracy across cross-validation folds for the current trial's hyperparameters.
        """
    dataset_name = trial.study.user_attrs['dataset_name']
    trial_number = trial.number  # Get the current trial number
    logger.info(f"Starting trial {trial_number + 1} for dataset: {dataset_name}")  # Display trial number (1-based)

    augment = trial.suggest_categorical("augment", [False, True])
    train, y_train, _, _ = load_dataset(dataset_name)

    y_train = y_train if isinstance(y_train, np.ndarray) else y_train.cpu().numpy()
    y_train_labels = np.argmax(y_train, axis = 1)
    n_classes = y_train.shape[1]

    nn_shapelet_generator = ShapeletNN(
        n_prototypes=trial.suggest_int("n_prototypes", 3, 121),
        bag_size=trial.suggest_int("bag_size", min(get_list_bag_sizes(dataset_name)),
                                   max(get_list_bag_sizes(dataset_name))),
        n_classes=n_classes,
        stride_ratio=STRIDE_RATIO,
        features_to_use_str=FEATURES_TO_USE_STR
    ).to(device)

    net = get_regularized_classifier(nn_shapelet_generator, max_epochs=1000, use_early_stopping=True,
                                     lambda_prototypes=trial.suggest_float("lambda_prototypes", 1e-8, 0.1, log=True),
                                     lambda_fused_lasso=trial.suggest_float("lambda_fused_lasso", 1e-8, 0.1, log=True))

    skf = StratifiedKFold(n_splits=CV_COUNT)
    accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(train, y_train_labels)):
        X_train, X_val = train[train_index].cpu().numpy(), train[val_index].cpu().numpy()
        y_train_fold, y_val_fold = y_train_labels[train_index], y_train_labels[val_index]

        X_train = torch.from_numpy(X_train).float().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)

        if augment:
            X_train, y_train_fold = augment_data(X_train.cpu().numpy(), y_train_fold)
            X_train = torch.from_numpy(X_train).float().to(device)
            y_train_fold = torch.from_numpy(y_train_fold).long().to(device)

        net.fit(X_train, y_train_fold)
        y_pred = net.predict(X_val)

        y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        accuracies.append(accuracy_score(y_val_fold, y_pred_labels))

    avg_accuracy = np.mean(accuracies)

    # Remove 'augment' from trial.params before logging
    trial_params = {k: v for k, v in trial.params.items() if k != 'augment'}
    logger.info(f"Trial {trial_number + 1} completed with accuracy: {avg_accuracy} and parameters: {trial_params}")

    trial.set_user_attr("y_train_labels", y_train_labels)
    return avg_accuracy

def find_best_hyper_params_optuna_search(dataset_name, n_trials=100):
    """
    Finds the best hyperparameters for the shapelet-based model using Optuna hyperparameter optimization.

    Args:
        dataset_name (str): The name of the dataset.
        n_trials (int, optional): The number of trials for Optuna optimization (default is 100).

    Returns:
        dict: A dictionary containing the best hyperparameters and the corresponding training accuracy.
    """
    study = create_study(sampler=TPESampler(), direction="maximize")
    study.set_user_attr("dataset_name", dataset_name)
    study.optimize(optuna_objective, n_trials=n_trials)
    best_trial = study.best_trial

    result = {
        "dataset_name": dataset_name,
        "search_type": "OptunaSearch",
        "features_to_use": FEATURES_TO_USE_STR,
        "bag_size": best_trial.params["bag_size"],
        "n_classes": len(np.unique(best_trial.user_attrs["y_train_labels"])),
        "n_prototypes": best_trial.params["n_prototypes"],
        "stride_ratio": STRIDE_RATIO,
        "lambda_fused_lasso": best_trial.params["lambda_fused_lasso"],
        "lambda_prototypes": best_trial.params["lambda_prototypes"],
        "n_iter": n_trials,
        "cv_count": CV_COUNT,
        "train_accuracy": best_trial.value
    }
    return result


def get_test_results_for_one_dataset(search_type, dataset_name, n_trials, search_max_epoch, test_result_max_epoch):
    """
    Gets the test results for a single dataset based on the best hyperparameters found during the search phase.

    Args:
        search_type (str): The type of search used to find the best hyperparameters.
        dataset_name (str): The name of the dataset.
        n_trials (int): The number of trials used in the search phase.
        search_max_epoch (int): The maximum number of epochs used in the search phase.
        test_result_max_epoch (int): The maximum number of epochs to use for evaluating the best model on the test set.

    Returns:
        None: The function saves the test results to a CSV file and doesn't return any value.
    """
    search_filename = get_filename_output_for_search(search_type, dataset_name, n_trials, search_max_epoch)
    output_best_results_filename = get_filename_output_for_best_results(search_type, dataset_name, n_trials,
                                                                        search_max_epoch, test_result_max_epoch)

    if os.path.exists(output_best_results_filename):
        df_results = pd.read_csv(output_best_results_filename)

        if not df_results.empty and dataset_name in df_results["dataset_name"].values:
            print(f"Results exist for {dataset_name} dataset, skipping...")
            return
    else:
        df_results = pd.DataFrame()

    if not os.path.exists(search_filename):
        print(f"{search_filename} does not exist, SKIPPING")
        return

    df_search = pd.read_csv(search_filename)

    if df_search.empty or len(df_search) < 1:
        print(f"{search_filename} is empty or does not have enough data, SKIPPING")
        return

    train_accuracy = df_search['train_accuracy'].iloc[0] if 'train_accuracy' in df_search.columns else None

    df = pd.read_csv(search_filename)
    print(f"RUNNING for {dataset_name}")
    bag_size = int(df["bag_size"][0])
    n_prototypes = df["n_prototypes"][0]
    lambda_fused_lasso = df["lambda_fused_lasso"][0]
    lambda_prototypes = df["lambda_prototypes"][0]
    augment = False  # Ensure augment is set to False during evaluation

    test_accuracy = evaluate_model_performance(
        dataset_name=dataset_name,
        bag_size=bag_size,
        n_prototypes=n_prototypes,
        max_epoch=test_result_max_epoch,
        phase="test",
        lambda_fused_lasso=lambda_fused_lasso,
        lambda_prototypes=lambda_prototypes,
        augment=augment
    )

    print(dataset_name, train_accuracy, test_accuracy)

    d = {
        "dataset_name": dataset_name,
        "bag_size": bag_size,
        "n_prototypes": n_prototypes,
        "stride_ratio": STRIDE_RATIO,
        "max_epoch": test_result_max_epoch,
        "features_to_use_str": FEATURES_TO_USE_STR,
        "lambda_fused_lasso": lambda_fused_lasso,
        "lambda_prototypes": lambda_prototypes,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

    if not os.path.exists('Results'):
        os.makedirs('Results')

    df_results = pd.concat([df_results, pd.DataFrame([d])], ignore_index=True)
    df_results["train_accuracy"] = df_results["train_accuracy"].astype(float).map('{:.5f}'.format)
    df_results["test_accuracy"] = df_results["test_accuracy"].astype(float).map('{:.5f}'.format)

    df_results.to_csv(output_best_results_filename, index=False)
    print(f"Saved results for {dataset_name} in {output_best_results_filename}")


def get_filename_output_for_search(search_type, dataset_name, n_trials, max_epoch):
    """
    Generates the filename for storing the search results based on the search type, dataset name, number of trials and maximum epochs.

    Args:
        search_type (str): The type of search used to find the best hyperparameters.
        dataset_name (str): The name of the dataset.
        n_trials (int): The number of trials used in the search phase.
        max_epoch (int): The maximum number of epochs used in the search phase.

    Returns:
        str: The generated filename for the search results.
    """
    return f"Results/{search_type}_params_{dataset_name}_n_trials_{n_trials}_max_epoch_{max_epoch}.csv"


def get_filename_output_for_best_results(search_type, dataset_name, n_trials, search_max_epoch,
                                         best_result_max_epoch):
    """
    Generates the filename for storing the best results based on the search type, dataset name, number of trials,
    search maximum epochs, and best result maximum epochs.

    Args:
        search_type (str): The type of search used to find the best hyperparameters.
        dataset_name (str): The name of the dataset.
        n_trials (int): The number of trials used in the search phase.
        search_max_epoch (int): The maximum number of epochs used in the search phase.
        best_result_max_epoch (int): The maximum number of epochs to use for evaluating the best model on the test set.

    Returns:
        str: The generated filename for the best results.
    """
    return f"Results/Test_results_according_to_{search_type}_n_trials_{n_trials}_search_max_epoch_{search_max_epoch}_result_max_epoch_{best_result_max_epoch}.csv"


dispatcher = {"OptunaSearch": find_best_hyper_params_optuna_search}


def find_result_for_one(search_type, dataset_name, n_trials, search_max_epoch):
    """
    Finds and saves the results for a single dataset using the specified search type.

    Args:
        search_type (str): The type of search used to find the best hyperparameters.
        dataset_name (str): The name of the dataset.
        n_trials (int, optional): The number of trials for Optuna optimization (default is 100).
        search_max_epoch (int, optional): The maximum number of epochs used in the search phase (default is 1000).

    Returns:
        None. The function saves the results to a CSV file and doesn't return any value.
    """
    output_filename = get_filename_output_for_search(search_type, dataset_name, n_trials, search_max_epoch)

    if os.path.isfile(output_filename):
        print(f"dataset: {dataset_name} exists in {output_filename}")
    else:
        print(f"RUNNING for dataset: {dataset_name}")
        result_dict = dispatcher[search_type](dataset_name, n_trials=n_trials)
        result_dict["search_type"] = search_type

        df = pd.DataFrame(result_dict, index=[0])
        df.to_csv(output_filename, index=False)
        print(f"Saved dataset: {dataset_name} in {output_filename}")
