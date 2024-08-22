def get_dataset_list_from_arguments(sys_argv):
    """
    Retrieves a list of dataset names based on arguments passed to the script.

    Args:
        sys_argv: A list containing arguments passed to the script. This can include command-line arguments
                  when run from a terminal, or arguments passed by an editor environment.

    Returns:
        A list of dataset names based on the provided arguments.

    Functionality:
    * Handles different argument scenarios:
        * No arguments: Returns the default dataset list `list1`.
        * One argument:
            * If the argument contains a colon (':'), interprets it as a range specification
              (e.g., "0:5:2" selects datasets from index 0 to 4 with a step of 2).
            * If the argument is numeric, interprets it as a single dataset index.
            * Otherwise, assumes the argument is a dataset name.
        * Two arguments: Interprets the arguments as the start and end indices of a range.
    * Utilizes predefined dataset lists for convenience.
    * Supports execution in both command-line and editor environments.
    """

    # Default dataset list
    dataset_list = list1

    if len(sys_argv) == 2:  # One argument provided
        arg = sys_argv[1]

        if ":" in arg:  # Range specification
            arg_arr = arg.split(":")
            if len(arg_arr) == 3:
                increment = int(arg_arr[2])
            else:
                increment = 1
            start = int(arg_arr[0])
            end = int(arg_arr[1])
            dataset_list = dataset_list[start:end:increment]
        elif arg.isnumeric():  # Single dataset index
            index = int(arg)
            dataset_list = dataset_list[index:index + 1]
        else:  # Dataset name
            index = dataset_list.index(arg)
            dataset_list = dataset_list[index:index + 1]

    if len(sys_argv) == 3:  # Two arguments provided (start and end indices)
        start = int(sys_argv[1])
        end = int(sys_argv[2])
        dataset_list = dataset_list[start:end]

    return dataset_list

# Predefined dataset lists
list1 = ["Trace"]  # Default dataset list

list3 = ["Coffee", "GunPoint", "ECG200"]

listNew = ["Beef", "Car", "CBF", "Computers", "CricketX", "CricketY", "CricketZ",
           "DistalPhalanxOutlineCorrect", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW",
           "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish"]

list_2_class = ["BirdChicken", "Coffee", "DistalPhalanxOutlineCorrect", "Earthquakes", "ECG200",
                "ECGFiveDays", "GunPoint", "Ham", "HandOutlines", "Herring",
                "ItalyPowerDemand", "Lightning2", "MiddlePhalanxOutlineCorrect",
                "MoteStrain", "PhalangesOutlinesCorrect", "ProximalPhalanxOutlineCorrect",
                "ShapeletSim", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "Strawberry",
                "ToeSegmentation1", "ToeSegmentation2", "TwoLeadECG", "Wafer",
                "Wine", "WormsTwoClass", "Yoga"]

list_image_datasets = ["DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
                       "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
                       "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish",
                       "HandOutlines", "Herring", "MiddlePhalanxOutlineAgeGroup",
                       "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "OSULeaf",
                       "PhalangesOutlinesCorrect", "ProximalPhalanxOutlineAgeGroup",
                       "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "ShapesAll",
                       "SwedishLeaf", "Symbols", "WordSynonyms", "Yoga"]

list_learning_shapelet_datasets = ["Adiac", "Beef", "BirdChicken", "ChlorineConcentration", "Coffee",
                                   "DiatomSizeReduction", "ECGFiveDays", "FaceFour", "ItalyPowerDemand",
                                   "Lightning7", "MedicalImages", "MoteStrain", "SonyAIBORobotSurface1",
                                   "SonyAIBORobotSurface2", "Symbols", "Trace", "TwoLeadECG", "CricketX",
                                   "CricketY", "CricketZ", "Lightning2", "Mallat", "Meat",
                                   "NonInvasiveFatalECGThorax1", "NonInvasiveFatalECGThorax2",
                                   "OliveOil", "ScreenType", "SmallKitchenAppliances", "StarlightCurves",
                                   "Worms", "WormsTwoClass", "Yoga"]

list_all_84 = ["Adiac", "ArrowHead",  "BeetleFly", "BirdChicken", "Car",
               "CBF", "ChlorineConcentration", "CinCECGtorso", "Coffee",
               "Computers", "CricketX", "CricketY", "CricketZ",
               "DiatomSizeReduction", "DistalPhalanxOutlineCorrect",
               "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW",
               "Earthquakes",  "ECG5000", "ECGFiveDays", "ElectricDevices",
               "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA",
               "GunPoint","ItalyPowerDemand", "LargeKitchenAppliances",
               "Lightning7", "Mallat", "Meat", "MedicalImages",
               "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup",
               "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECGThorax1",
               "NonInvasiveFatalECGThorax2", "OliveOil", "OSULeaf",
               "PhalangesOutlinesCorrect", "Plane", "ProximalPhalanxOutlineCorrect",
               "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "RefrigerationDevices",
               "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
               "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarlightCurves", "Strawberry",
               "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2",
               "Trace", "TwoLeadECG", "TwoPatterns",
               "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "UWaveGestureLibraryAll",
               "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass"]