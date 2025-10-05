from typing import List, Any, Dict
import sklearn
import pandas as pd
import numpy as np
import logging
import joblib
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# each algorithm will be set up as its own data model object (a class instance)
# separating data, the configuration of the model and the model fitting and evaluation
class ModelObject:
    """
    Idea: Separate the model object from the data and the fitting process.
    1) Create a model object with the model configuration and the data
    2) Fit the model to the data
    3) Evaluate the model with a metric function
    4) Extract all the metadata for the model
    5) Store the model object, the data and the metadata in a dictionary or a
    """
    def __init__(
        self,
        ):
        self.labels = None
        self.fitted_model = None
        self.score = None
        self.model_saving_info = None

    # The idea of the two following class methods is that we cant place them in the __init__
    # because we want to be able to create an empty model object and then set the model
    # and the data later because we also want to use the class to load already fitted models
    # from a file and then use the class methods to predict and evaluate
    # and extract the metadata
    # So we need to be able to set the model and the data after the object is created
    # and not only at the time of creation
    def set_classifier_model_object(
        self,
        classifier_model_object: sklearn.base.BaseEstimator,
        ) -> None:
        """Setting the classifier model object.

        Args:
            classifier_model_object (sklearn.base.BaseEstimator): The classifier model object to set.
        Examples:
            from sklearn.cluster import KMeans
            from model import ModelObject
            kmeans = KMeans(n_clusters=3, random_state=42)
            model_object_instance = ModelObject()
            model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
        """
        self.classifier_model_object = classifier_model_object


    def set_data(
        self,
        data: pd.DataFrame
        ) -> None:
        """Setting the data for the model.

        Args:
            data (pd.DataFrame): The data to set.
        Examples:
            import pandas as pd
            from model import ModelObject
            data = pd.DataFrame(index=pd.date_range(start="2000-01-01", end="2023-01-01", freq="M"),
                                data={
                                    "feature1": range(276),
                                    "feature2": range(276, 0, -1),
                                    "feature3": [x % 5 for x in range(276)],
                                })
            model_object_instance = ModelObject()
            model_object_instance.set_data(data=data)
        """
        self.data = data


    def fit(
        self
        ):
        """Fit the model to the data.
        """
        logging.info(f"Fitting model: {self.classifier_model_object.__class__.__name__}")
        self.fitted_model = self.classifier_model_object.fit(self.data)
        self.time_last_fitted = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


    def predict(
        self,
        new_data: pd.DataFrame = None
        ) -> List[Any]:
        """Predict the labels for the data.

        Args:
            new_data (pd.DataFrame, optional): The new data to predict. Defaults to None.

        Examples:
            import pandas as pd
            from model import ModelObject
            data = pd.DataFrame(index=pd.date_range(start="2000-01-01", end="2023-01-01", freq="M"),
                                data={
                                    "feature1": range(276),
                                    "feature2": range(276, 0, -1),
                                    "feature3": [x % 5 for x in range(276)],
                                })
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            model_object_instance = ModelObject()
            model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
            model_object_instance.set_data(data=data)
            model_object_instance.fit()

        Raises:
            ValueError: If the model is not fitted.

        Returns:
            List[Any]: The predicted labels.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")
        logging.info(f"Predicting with model: {self.classifier_model_object.__class__.__name__}")
        if new_data is not None:
            if "predict" in dir(self.fitted_model):
                self.labels = self.fitted_model.predict(new_data)
            elif "fit_predict" in dir(self.fitted_model):
                self.labels = self.fitted_model.fit_predict(new_data)
        else:
            if "predict" in dir(self.fitted_model):
                self.labels = self.fitted_model.predict(self.data)
            elif "fit_predict" in dir(self.fitted_model):
                self.labels = self.fitted_model.fit_predict(self.data)
        return self.labels


    def evaluate(
        self,
        metric_function: callable
        ) -> float:
        """Evaluate the model using a metric function.

        Args:
            metric_function (callable): The metric function to use for evaluation.

        Examples:
            from sklearn.metrics import silhouette_score
            from model import ModelObject
            data = pd.DataFrame(index=pd.date_range(start="2000-01-01", end="2023-01-01", freq="M"),
                                data={
                                    "feature1": range(276),
                                    "feature2": range(276, 0, -1),
                                    "feature3": [x % 5 for x in range(276)],
                                })
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            model_object_instance = ModelObject()
            model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
            model_object_instance.set_data(data=data)
            model_object_instance.fit()
            model_object_instance.predict()
            evaluation_score = model_object_instance.evaluate(metric_function=silhouette_score)

        Raises:
            ValueError: If the model is not fitted.

        Returns:
            float: The evaluation score.
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before evaluation.")
        try:
            self.score = metric_function(self.data, self.labels)
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            self.score = None
        logging.info(f"Evaluation score: {self.score}")
        return self.score


    def save_model(
        self,
        file_format: str=".pkl",
        file_path: str=None,
        model_file_name: str=None
        ):
        """Save the model to a file.

        Args:
            file_format (str, optional): The file format to use for saving. Defaults to ".pkl".
            file_path (str, optional): The file path to save the model. Defaults to None.
            model_file_name (str, optional): The model file name. Defaults to None.

        Examples:
            from model import ModelObject
            model_object_instance = ModelObject()
            model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
            model_object_instance.set_data(data=data)
            model_object_instance.fit()
            model_object_instance.save_model(file_format=".pkl", file_path="/path/to/save", model_file_name="kmeans_model")
        """
        logging.info(f"Saving model to {file_path}/{model_file_name} in format {file_format}")
        joblib.dump(
            value=self.fitted_model,
            filename=fr"{file_path}/{model_file_name}{file_format}"
            )
        model_saving_info = {
            "file_format": file_format,
            "file_path": file_path,
            "model_file_name": model_file_name,
            "model_file_saved": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "full_model_path": fr"{file_path}/{model_file_name}{file_format}"
        }
        self.model_saving_info = model_saving_info


    def load_model(
        self,
        file_format: str=".pkl",
        file_path: str=None,
        model_file_name: str=None
        ):
        """Load a model from a file.

        Args:
            file_format (str, optional): The file format to use for loading. Defaults to ".pkl".
            file_path (str, optional): The file path to load the model from. Defaults to None.
            model_file_name (str, optional): The pure model file name,
            without the file extension, that is added by the parameter file_format. Defaults to None.

        Examples:
            from model import ModelObject
            model_object_instance = ModelObject()
            model_object_instance.load_model(file_format=".pkl", file_path="/path/to/save", model_file_name="kmeans_model")

        Returns:
            _type_: _description_
        """
        logging.info(f"Loading model from {file_path}/{model_file_name} in format {file_format}")
        self.fitted_model = joblib.load(filename=fr"{file_path}/{model_file_name}{file_format}")
        self.classifier_model_object = self.fitted_model
        self.labels = self.fitted_model.labels_ if hasattr(self.fitted_model, 'labels_') else None
        return self.fitted_model


    def _flatten_model_info(
        self,
        info: Dict[str, Any]
        ) -> pd.DataFrame:
        """Flatten the model information into a DataFrame.

        Args:
            info (Dict[str, Any]): The model information to flatten.

        Examples:
            from model import ModelObject
            model_object_instance = ModelObject()
            flat_info_df = model_object_instance._flatten_model_info(info=model_info)

        Returns:
            pd.DataFrame: The flattened model information.
        """
        flat_info = {}
        for k, v in info.items():
            if isinstance(v, (np.ndarray, list)):
                flat_info[k] = str(v)
            elif isinstance(v, dict):
                flat_info[k] = str(v)
            else:
                flat_info[k] = v
        flat_info_df = pd.DataFrame([flat_info])
        return flat_info_df


    def get_full_model_info(
        self,
        save: bool=False,
        load: bool=False,
        return_info_dict: bool=True,
        return_info_df: bool=False,
        file_format: str=".txt", # change to json() format
        file_path: str=None,
        file_name: str=None
        ):
        """Get the full model information.

        Args:
            save (bool, optional): Whether to save the model information. Defaults to False.
            load (bool, optional): Whether to load the model information. Defaults to False.
            return_info_dict (bool, optional): Whether to return the information as a dictionary. Defaults to True.
            return_info_df (bool, optional): Whether to return the information as a DataFrame. Defaults to False.
            file_format (str, optional): The file format to use for saving or loading. Defaults to ".txt".
            file_name (str, optional): The file name to use for saving or loading. Defaults to None.

        Examples:
            from model import ModelObject
            model_object_instance = ModelObject()
            model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
            model_object_instance.set_data(data=data)
            model_object_instance.fit()
            model_object_instance.evaluate(metric_function=silhouette_score)
            model_info = model_object_instance.get_full_model_info(save=True, file_format=".txt", file_path="/path/to/save", file_name="kmeans_model")
            model_info_dict = model_object_instance.get_full_model_info(return_info_dict=True, return_info_df=False)
            model_info_df = model_object_instance.get_full_model_info(return_info_dict=False, return_info_df=True)
            model_info_both = model_object_instance.get_full_model_info(return_info_dict=True, return_info_df=True)

        Returns:
            _type_: _description_
        """
        # extract all the metadata for the model
        model_type = self.classifier_model_object.__class__.__name__
        feature_names_in = self.classifier_model_object.feature_names_in_ if hasattr(self.classifier_model_object, 'feature_names_in_') else None
        model_params = self.classifier_model_object.get_params() if hasattr(self.classifier_model_object, 'get_params') else None
        n_iter = self.classifier_model_object.n_iter_ if hasattr(self.classifier_model_object, 'n_iter_') else None
        cluster_centers = self.classifier_model_object.cluster_centers_ if hasattr(self.classifier_model_object, 'cluster_centers_') else None
        labels = self.classifier_model_object.labels_ if hasattr(self.classifier_model_object, 'labels_') else None
        time_last_fitted = self.time_last_fitted if hasattr(self, 'time_last_fitted') else None
        model_info = {
            "model_type": model_type,
            "feature_names_in": feature_names_in,
            "model_params": model_params,
            "n_iter": n_iter,
            "cluster_centers": cluster_centers,
            "time_last_fitted": time_last_fitted,
            "predicted_labels": labels
        }

        model_info["evaluation_score"] = self.score
        if self.model_saving_info is not None:
            model_info.update(self.model_saving_info)

        if save and file_path is not None and file_name is not None:
            model_info["model_info_saved"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Saving model info to {file_path}/{file_name} in format {file_format}")
            try:
                with open(fr"{file_path}/{file_name}{file_format}", "w") as f:
                    for key, value in model_info.items():
                        f.write(f"{key}: {value}\n")
            except Exception as e:
                logging.error(f"Error saving model info: {e}")

        elif load and file_path is not None and file_name is not None:
            logging.info(f"Loading model info from {file_path}/{file_name} in format {file_format}")
            try:
                model_info = {}
                with open(fr"{file_path}/{file_name}{file_format}", "r") as f:
                    for line in f:
                        key, value = line.strip().split(": ", 1)
                        model_info[key] = value
                model_info.update(model_info)
            except Exception as e:
                logging.error(f"Error loading model info: {e}")

        if return_info_dict:
            return model_info
        elif return_info_df:
            flat_info_df = self._flatten_model_info(info=model_info)
            return flat_info_df
        elif return_info_dict and return_info_df:
            flat_info_df = self._flatten_model_info(info=model_info)
            return model_info, flat_info_df
        
def flatten_model_info(
    info: Dict[str, Any]
    ) -> pd.DataFrame:
    """Flatten the model information into a DataFrame.

    Args:
        info (Dict[str, Any]): The model information to flatten.

    Examples:
        from model import ModelObject
        model_object_instance = ModelObject()
        flat_info_df = model_object_instance._flatten_model_info(info=model_info)

    Returns:
        pd.DataFrame: The flattened model information.
    """
    flat_info = {}
    for k, v in info.items():
        if isinstance(v, (np.ndarray, list)):
            flat_info[k] = str(v)
        elif isinstance(v, dict):
            flat_info[k] = str(v)
        else:
            flat_info[k] = v
    flat_info_df = pd.DataFrame([flat_info])
    return flat_info_df

def load_full_model_info(
    file_format: str=".txt",
    file_path: str=None,
    file_name: str=None,
    return_info_dict: bool=True,
    return_info_df: bool=False,
    ) -> Dict[str, Any]:
    """Load the full model information from a file.

    Args:
        file_format (str, optional): The file format to use for loading. Defaults to ".txt".
        file_path (str, optional): The file path to load the model information from. Defaults to None.
        file_name (str, optional): The pure model file name,
            without the file extension, that is added by the parameter file_format. Defaults to None.

    Examples:
        from model import load_full_model_info
        model_info = load_full_model_info(file_format=".txt", file_path="/path/to/save", file_name="kmeans_model")

    Returns:
        Dict[str, Any]: The loaded model information.
    """
    logging.info(f"Loading model info from {file_path}/{file_name} in format {file_format}")
    try:
        model_info = {}
        with open(fr"{file_path}/{file_name}{file_format}", "r") as f:
            for line in f:
                if ": " not in line:
                    continue  # skip malformed lines
                key, value = line.strip().split(": ", 1)
                model_info[key] = value
        model_info.update(model_info)

        if return_info_df:
            flat_info_df = flatten_model_info(info=model_info)
            return flat_info_df
        elif return_info_dict:
            return model_info
        elif return_info_dict and return_info_df:
            flat_info_df = flatten_model_info(info=model_info)
            return model_info, flat_info_df
    except Exception as e:
        logging.error(f"Error loading model info: {e}")
        return {}


# Set up Benchmark models to identify the relevant exchange rate regimes
print(os.getcwd())
os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code")
print(os.getcwd())

from data_graphing.data_grapher import DataGraphing

data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)
# Now load the exchange rate data from BIS - daily data
country_keys_mapping = {
    "US": "United States",
}
exchange_rates_df = data_loading_instance.get_bis_exchange_rate_data(
    country_keys_mapping=country_keys_mapping,
    exchange_rate_type_list=[
        "Nominal effective exchange rate - daily - narrow basket",
        ]).rename(columns={"US_Nominal effective exchange rate - daily - narrow basket": "US_NEER"})
exchange_rates_df = exchange_rates_df.dropna()
















# For each model type create a data model object instance 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift

kmeans = KMeans(n_clusters=3, random_state=42)
agg = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=1.5, min_samples=8)
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
ms = MeanShift()



# Create the single ModelObjects
data = pd.DataFrame(index=pd.date_range(start="2000-01-01", end="2023-01-01", freq="M"),
                    data={
                        "feature1": range(276),
                        "feature2": range(276, 0, -1),
                        "feature3": [x % 5 for x in range(276)],
                    })
# models_objects_list = [ModelObject() for model in models_list]
# Fit each model and store the labels
model_object_instance = ModelObject()
model_object_instance.set_classifier_model_object(classifier_model_object=kmeans)
model_object_instance.set_data(data=data)
model_object_instance.fit()
predicted_labels = model_object_instance.predict()
from sklearn.metrics import silhouette_score
evaluation_score = model_object_instance.evaluate(metric_function=silhouette_score)
model_object_instance.save_model(file_format=r".pkl", file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models", model_file_name=r"kmeans_model")
model_object_instance.get_full_model_info(save=True, file_format=r".txt", file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models", file_name=r"kmeans_model")
full_model_info = model_object_instance.get_full_model_info()
# Load a model from a saved file
new_model_object_instance = ModelObject()
new_model_object_instance.load_model(file_format=r".pkl", file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models", model_file_name=r"kmeans_model")
dir(new_model_object_instance.fitted_model)
full_model_info_new = new_model_object_instance.get_full_model_info()
# Load an alreaday saved full_model_info file
new_model_object_instance.get_full_model_info(
    save=False,
    load=True,
    return_info_dict=True,
    return_info_df=False,
    file_format=".txt",
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models",
    file_name="KMeans_model")


# Loop for multiple models
models_list = [kmeans, agg, dbscan, spectral, ms]
for model in models_list:
    model_object_instance = ModelObject() # Initialize a new instance for each model
    model_object_instance.set_classifier_model_object(classifier_model_object=model) # Set the model
    model_object_instance.set_data(data=data) # Set the data
    model_object_instance.fit() # Fit the model
    predicted_labels = model_object_instance.predict() # Predict the labels
    evaluation_score = model_object_instance.evaluate(metric_function=silhouette_score) # Evaluate the model
    # Save the model and the model info with dynamic names based on the model class name
    model_object_instance.save_model(file_format=r".pkl", file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models", model_file_name=f"{model.__class__.__name__}_model")
    # Get the full model info
    model_object_instance.get_full_model_info(save=True, file_format=r".txt", file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models", file_name=f"{model.__class__.__name__}_model")

# For a proper evaluation now, loading all full_info files and compare the evaluation scores
import os
full_model_info_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models"
all_model_info_files = [file for file in os.listdir(full_model_info_path) if file.endswith(".txt")]
# Delete the file extension for loading
all_model_info_files = [os.path.splitext(file)[0] for file in all_model_info_files]
# Load all model info files into a list of dataframes

all_model_info_dfs = [load_full_model_info(
    file_format=".txt",
    file_path=full_model_info_path,
    file_name=file,
    return_info_dict=False,
    return_info_df=True,
    ) for file in all_model_info_files]
all_models_comp_df = pd.concat(all_model_info_dfs).reset_index(drop=True)
# all_models_comp_df.to_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models/all_models_comparison.xlsx", index=False)
