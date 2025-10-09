from typing import List, Any, Dict
import sklearn
import pandas as pd
import numpy as np
import logging
import joblib
import statsmodels
import json
import datetime as dt
import inspect
import os
import warnings
import ast
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models"
FIGURES_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"
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
    ######################## Internal helper methods #########################
    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved, i.e.
                        i.e. /Users/Robert_Hennings/Uni/Master/Seminar/reports/figures
                    !!!!DO NOT INCLUDE THE FILENAME like: /Users/Robert_Hennings/Uni/Master/Seminar/reports/figures/figure.pdf
        """
        folder_name = path.split("/")[-1]
        path = "/".join(path.split("/")[:-1])
        # FileNotFoundError()
        # os.path.isdir()
        if folder_name not in os.listdir(path):
            logging.info(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            logging.info(f"Folder: {folder_name} created in path: {path}")


    def __get_init_params(
        self,
        model
        ) -> Dict[str, Any]:
        sig = inspect.signature(model.__class__.__init__)
        param_names = [p for p in sig.parameters if p != 'self']
        params = {}
        for name in param_names:
            if hasattr(model, name):
                params[name] = getattr(model, name)
            else:
                # Try to get from __dict__ if not a direct attribute
                params[name] = model.__dict__.get(name, None)
        return params


    def set_model_object(
        self,
        model_object: sklearn.base.BaseEstimator,
        ) -> None:
        """Setting the model object.

        Args:
            classifier_model_object (sklearn.base.BaseEstimator): The classifier model object to set.
        Examples:
            from sklearn.cluster import KMeans
            from model import ModelObject
            kmeans = KMeans(n_clusters=3, random_state=42)
            model_object_instance = ModelObject()
            model_object_instance.set_model_object(model_object=kmeans)
        """
        self.model_object = model_object
        # Also save the initialization parameters of the model
        self.model_init_params = self.__get_init_params(model_object)


    def set_data(
        self,
        training_data: pd.DataFrame,
        testing_data: pd.DataFrame, 
        ) -> None:
        """Setting the data for the model.

        Args:
            training_data (pd.DataFrame): The training data to set.
            testing_data (pd.DataFrame): The testing data to set.
        Examples:
            import pandas as pd
            from model import ModelObject
            training_data = pd.DataFrame(index=pd.date_range(start="2000-01-01", end="2023-01-01", freq="M"),
                                data={
                                    "feature1": range(276),
                                    "feature2": range(276, 0, -1),
                                    "feature3": [x % 5 for x in range(276)],
                                })
            testing_data = pd.DataFrame(index=pd.date_range(start="2023-02-01", end="2024-01-01", freq="M"),
                                data={
                                    "feature1": range(276, 0, -1),
                                    "feature2": range(276),
                                    "feature3": [x % 5 for x in range(276)],
                                })
            model_object_instance = ModelObject()
            model_object_instance.set_data(training_data=training_data, testing_data=testing_data)
        """
        self.training_data = training_data
        self.testing_data = testing_data
        # Also store the timestamps for training and testing data
        self.training_data_dates = training_data.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        self.testing_data_dates = testing_data.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Compare the dates of training and testing data
        # Statsmodels-style model: already has data inside
        if hasattr(self.model_object, 'fit') and 'endog' in self.model_init_params:
            model_training_data_dates = self.model_object.data.orig_endog.index.tolist()
            model_training_data_dates = [pd.Timestamp(date).strftime("%Y-%m-%d %H:%M:%S") for date in model_training_data_dates]
            if model_training_data_dates == self.training_data_dates and model_training_data_dates == self.testing_data_dates:
                logging.info("Model data dates match training data dates.")
            else:
                logging.warning("Model data dates do not match training data dates.")
                logging.warning(f"Model data dates: {model_training_data_dates}")
                logging.warning(f"Training data dates: {self.training_data_dates}")
                self.training_data_dates = model_training_data_dates
            # Also extract the testing data from the model and check
            if self.training_data.equals(self.testing_data):
                self.testing_data_dates = self.training_data_dates


    def fit(
        self,
        **kwargs
        ) -> None:
        """Fit the model to the data."""
        logging.info(f"Fitting model: {self.model_object.__class__.__name__}")

        # Statsmodels-style model: already has data inside
        if hasattr(self.model_object, 'fit') and 'endog' in self.model_init_params:
            valid_fit_params = inspect.signature(self.model_object.fit).parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fit_params}
            fit_mod = self.model_object.fit(**filtered_kwargs)
        # Sklearn-style model: data must be passed in
        elif hasattr(self.model_object, 'fit'):
            valid_fit_params = inspect.signature(self.model_object.fit).parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fit_params}
            fit_mod = self.model_object.fit(self.training_data, **filtered_kwargs)

        else:
            raise TypeError(f"Model {self.model_object.__class__.__name__} does not have a fit() method.")

        self.fitted_model = fit_mod
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
        logging.info(f"Predicting with model: {self.model_object.__class__.__name__}")
        if new_data is not None:
            testing_start_date = new_data.index.min()
            testing_end_date = new_data.index.max()
            if "predict" in dir(self.fitted_model):
                try:
                    self.labels = self.fitted_model.predict(new_data)
                except Exception as e:
                    logging.info(f"Error during prediction: {e}")
                    if "start" in inspect.signature(self.fitted_model.predict).parameters:
                        self.labels = self.fitted_model.predict(start=testing_start_date, end=testing_end_date)
            elif "fit_predict" in dir(self.fitted_model):
                try:
                    self.labels = self.fitted_model.fit_predict(new_data)
                except Exception as e:
                    logging.info(f"Error during prediction: {e}")
                    if "start" in inspect.signature(self.fitted_model.fit_predict).parameters:
                        self.labels = self.fitted_model.fit_predict(start=testing_start_date, end=testing_end_date)
        else:
            if "predict" in dir(self.fitted_model):
                try:
                    # For sklearn models
                    self.labels = self.fitted_model.predict(self.testing_data)
                except Exception as e:
                    logging.info(f"Error during prediction: {e}")
                    # For statsmodels MarkovRegression results object
                    if "start" in inspect.signature(self.fitted_model.predict).parameters:
                        # Use index values for start/end
                        start = self.testing_data.index.min()
                        end = self.testing_data.index.max()
                        self.labels = self.fitted_model.predict(start=0, end=len(self.testing_data)-1)
                        if self.model_object.__class__.__name__ == "MarkovRegression":
                            # Convert the probabilities to regimes based on a threshold
                            smoothed_probabilities = model_object_instance.fitted_model.smoothed_marginal_probabilities
                            msm_regime = smoothed_probabilities[1]
                            regime = (msm_regime >= MARKOV_REGIME_PROB).astype(int)
                            self.labels = regime.tolist()
                    else:
                        self.labels = self.fitted_model.predict()
            elif "fit_predict" in dir(self.fitted_model):  
                try:
                    self.labels = self.fitted_model.fit_predict(self.testing_data)
                except Exception as e:
                    logging.info(f"Error during prediction: {e}")
                    self.labels = self.fitted_model.fit_predict()
        if type(self.labels) == np.ndarray:
            self.labels = self.labels.tolist()
        elif type(self.labels) == pd.Series:
            # also transform to a simple list
            self.labels = self.labels.tolist()
            # self.labels = self.labels.to_frame(name="predicted_labels")
        return self.labels


    def evaluate(
        self,
        metric_function_list: List[callable]
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
            score_dict = {}
            for metric_function in metric_function_list:
                logging.info(f"Evaluating model with metric: {metric_function.__name__}")
                score = metric_function(self.testing_data, self.labels)
                score_dict[metric_function.__name__] = score
            self.score_dict = score_dict
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            self.score_dict = None
        logging.info(f"Evaluation score: {self.score_dict}")
        return self.score_dict


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
        logging.info(f"Saving model to {file_path}/{model_file_name}{file_format}")
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
        file_format_model: str=".pkl",
        file_path_model: str=None,
        model_file_name: str=None,
        return_info_dict: bool=False,
        return_info_df: bool=False,
        file_format_model_info: str=".json",
        file_path_model_info: str=None,
        model_file_name_info: str=None
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
        self = new_model_object_instance
        logging.info(f"Loading model from {file_path_model}/{model_file_name}{file_format_model}")
        self.fitted_model = joblib.load(filename=fr"{file_path_model}/{model_file_name}{file_format_model}")
        self.model_object = self.fitted_model

        full_model_info_loaded_dict, full_model_info_loaded_dict_df = self.get_full_model_info(
            load=True,
            return_info_dict=True,
            return_info_df=True,
            file_format=file_format_model_info,
            file_path=file_path_model_info,
            file_name=model_file_name_info
            )
        # Now we could loop through the dict and set the attributes of the class instance
        for key, value in full_model_info_loaded_dict.items():
            setattr(self, key, value)
        if return_info_dict:
            return self.fitted_model, full_model_info_loaded_dict, full_model_info_loaded_dict_df
        if return_info_df:
            return self.fitted_model, full_model_info_loaded_dict_df


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
        if save:
            # extract all the metadata for the model
            model_type = self.model_object.__class__.__name__
            feature_names_in = self.model_object.feature_names_in_ if hasattr(self.model_object, 'feature_names_in_') else self.training_data.columns.tolist()
            # save also the training and testing data
            training_data = self.training_data if hasattr(self, 'training_data') else None
            testing_data = self.testing_data if hasattr(self, 'testing_data') else None
            model_params = self.model_object.get_params() if hasattr(self.model_object, 'get_params') else None
            n_iter = self.model_object.n_iter_ if hasattr(self.model_object, 'n_iter_') else None
            cluster_centers = self.model_object.cluster_centers_ if hasattr(self.model_object, 'cluster_centers_') else None
            labels = self.labels if hasattr(self, 'labels') else None
            time_last_fitted = self.time_last_fitted if hasattr(self, 'time_last_fitted') else None
            testing_data_dates = self.testing_data_dates if hasattr(self, 'testing_data_dates') else None
            training_data_dates = self.training_data_dates if hasattr(self, 'training_data_dates') else None
            model_info = {
                "model_type": model_type,
                "feature_names_in": feature_names_in,
                "model_params": model_params,
                "n_iter": n_iter,
                "cluster_centers": cluster_centers,
                "time_last_fitted": time_last_fitted,
                "predicted_labels": labels,
                "training_data": training_data,
                "testing_data": testing_data,
                "testing_data_dates": testing_data_dates,
                "training_data_dates": training_data_dates,
            }
            # Also append the initialization parameters of the model
            if hasattr(self, 'model_init_params'):
                for key, value in self.model_init_params.items():
                    model_info[key] = value

            model_info["evaluation_score"] = self.score_dict if hasattr(self, 'score_dict') else None
            if self.model_saving_info is not None:
                model_info.update(self.model_saving_info)
        if save and file_path is not None and file_name is not None:
            model_info["model_info_saved"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Saving model info to {file_path}/{file_name}{file_format}")

            if file_format == ".txt":
                try:
                    with open(fr"{file_path}/{file_name}{file_format}", "w") as f:
                        for key, value in model_info.items():
                            f.write(f"{key}: {value}\n")
                except Exception as e:
                    logging.error(f"Error saving model info: {e}")
            elif file_format == ".json":
                # Transform numpy.ndarray to list for JSON serialization
                for key, value in model_info.items():
                    if isinstance(value, np.ndarray):
                        model_info[key] = value.tolist()
                # Also transform the pd.DataFrame objects to string representation
                for key, value in model_info.items():
                    if isinstance(value, pd.DataFrame):
                        model_info[key] = value.to_dict(orient="list")
                try:
                    with open(fr"{file_path}/{file_name}{file_format}", "w") as f:
                        json.dump(model_info, f, indent=2)
                except Exception as e:
                    logging.error(f"Error saving model info: {e}")

        elif load and file_path is not None and file_name is not None:
            logging.info(f"Loading model info from {file_path}/{file_name}{file_format}")
            if file_format == ".txt":
                try:
                    model_info = {}
                    with open(fr"{file_path}/{file_name}{file_format}", "r") as f:
                        for line in f:
                            key, value = line.strip().split(": ", 1)
                            model_info[key] = value
                    model_info.update(model_info)
                except Exception as e:
                    logging.error(f"Error loading model info: {e}")
            elif file_format == ".json":
                try:
                    with open(fr"{file_path}/{file_name}{file_format}", "r") as f:
                        model_info = json.load(f)
                    model_info.update(model_info)
                    # Transfer the testing and training data back to pd.DataFrame
                    if "training_data" in model_info and isinstance(model_info["training_data"], dict):
                        model_info["training_data"] = pd.DataFrame(model_info["training_data"])
                    if "testing_data" in model_info and isinstance(model_info["testing_data"], dict):
                        model_info["testing_data"] = pd.DataFrame(model_info["testing_data"])
                except Exception as e:
                    logging.error(f"Error loading model info: {e}")

        flat_info_df = self._flatten_model_info(info=model_info)

        if return_info_dict and return_info_df:
            return model_info, flat_info_df
        elif return_info_df:
            return flat_info_df
        elif return_info_dict:
            return model_info


    def save_model_summary(
        self,
        model_summary: str,
        save_txt: bool=False,
        save_csv: bool=False,
        save_latex: bool=False,
        save_html: bool=False,
        save_json: bool=False,
        file_path: str=None,
        file_name: str=None,
        ) -> None:
        self.__check_path_existence(path=file_path)
        full_path_save = f"{file_path}/{file_name}"
        if save_txt:
            full_path_save_txt = f"{full_path_save}.txt"
            try:
                with open(file=full_path_save_txt, mode="w") as f:
                    f.write(model_summary.as_text())
                logging.info(f"Exported DataFrame to {full_path_save_txt}")
            except Exception as e:
                logging.error(f"Error exporting to .txt: {e}")
        if save_csv:
            full_path_save_csv = f"{full_path_save}.csv"
            try:
                with open(file=full_path_save_csv, mode="w") as f:
                    f.write(model_summary.as_csv())
                logging.info(f"Exported DataFrame to {full_path_save_csv}")
            except Exception as e:
                logging.error(f"Error exporting to .csv: {e}")

        if save_latex:
            full_path_save_latex = f"{full_path_save}.tex"
            try:
                with open(file=full_path_save_latex, mode="w") as f:
                    f.write(model_summary.as_latex())
                logging.info(f"Exported DataFrame to {full_path_save_latex}")
            except Exception as e:
                logging.error(f"Error exporting to .tex: {e}")

        if save_html:
            full_path_save_html = f"{full_path_save}.html"
            try:
                with open(file=full_path_save_html, mode="w") as f:
                    f.write(model_summary.as_html())
                logging.info(f"Exported DataFrame to {full_path_save_html}")
            except Exception as e:
                logging.error(f"Error exporting to .html: {e}")

        if save_json:
            full_path_save_json = f"{full_path_save}.json"
            try:
                with open(file=full_path_save_json, mode="w") as f:
                    json.dump(model_summary.as_csv(), f, indent=2)
                logging.info(f"Exported DataFrame to {full_path_save_json}")
            except Exception as e:
                logging.error(f"Error exporting to .json: {e}")


#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Benchmark Models for Regime Identification
#----------------------------------------------------------------------------------------
print(os.getcwd())
os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code")
print(os.getcwd())

from data_loading.data_loader import DataLoading
from data_graphing.data_grapher import DataGraphing

# Import util functions for model evaluation and comparisons
from utils.evaluation import get_model_metadata_df,\
    extract_predicted_labels_from_metadata_df,\
    get_recoded_predicted_labels_df, \
    get_regime_counts_df, \
    get_overlapping_regimes_df, \
    get_periods_overlaying_df
# Import custom written model evaluation scores
from utils.evaluation_metrics import compute_rcm

data_graphing_instance = DataGraphing()
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
# Recreate the Markov-Switching model from the paper: The impact of oil shocks on exchange rates: A Markov-switching approach
# Two regimes assumed: High volatility and low volatility regime, page 14
# Dependent variable: first difference of the log real exchange rate for country i, page 14
# The Markov-switching models for exchange rates were estimated using the fMarkovSwitching package in R (Perlin, 2008).
# The models were estimated with two states, state dependent regression coefficients and state dependent volatility for
# the error process. Exchange rates are known to exhibit volatility clustering which is why we allow volatility to vary across regimes. Models were estimated using two different as- sumptions about the error term (normal, Student-t).

# Prepare the data for the Markov Switching model
exchange_rates_df["log_US_NEER"] = np.log(exchange_rates_df["US_NEER"])
exchange_rates_df["log_US_NEER_diff"] = exchange_rates_df["log_US_NEER"].diff()
exchange_rates_df = exchange_rates_df.dropna()
window = 30
exchange_rates_vola_df = exchange_rates_df["log_US_NEER_diff"].rolling(window=window).std().dropna().to_frame(name="rolling_vola")
# A first benchmark approach will be to only use the univariate rolling volatility of the exchange rate changes
# as the variable to identify the regimes
# Later we will also include exogeneous variables like oil price volatility and gas price volatility
# to see if the regimes change
# Also note that the Markov Switching model is only capable of doing IN-SAMPLE Forecasts
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import MiniBatchKMeans
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import silhouette_score

# At first we assume three regimes: One low volatility regime and two high volatility regimes
n_regimes = 2
MARKOV_REGIME_PROB = 0.5
# Now the work flow looks like follows:
# We initialize a new instance of the ModelObject class for each model with desired settings
kmeans = KMeans(n_clusters=n_regimes, random_state=42)
agg = AgglomerativeClustering(n_clusters=n_regimes)
dbscan = DBSCAN(eps=1.5, min_samples=8)
# spectral = SpectralClustering(n_clusters=n_regimes, affinity='nearest_neighbors', random_state=42)
ms = MeanShift()
msm = MarkovRegression(
    endog=exchange_rates_vola_df,
    k_regimes=n_regimes,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
)
gmm = GaussianMixture(n_components=n_regimes, random_state=42)
birch = Birch(n_clusters=n_regimes)
affinity = AffinityPropagation()
optics = OPTICS()
minibatch_kmeans = MiniBatchKMeans(n_clusters=n_regimes, random_state=42)
# In the below dict, additional fit kwargs for each model class can be specified
# that will be passed to the fit() method of the respective model class
fit_kwargs_dict = {
    "KMeans": {"n_init": 10},
    "AgglomerativeClustering": {},
    "DBSCAN": {},
    "MeanShift": {},
    "MarkovRegression": {"em_iter": 10, "search_reps": 20},
    "MarkovAutoregression": {"em_iter": 10, "search_reps": 20},
    # Add more models as needed
}
# We store them in a list that we will loop through
models_list = [kmeans, agg, dbscan, ms, msm, gmm, birch, affinity, optics, minibatch_kmeans]
for model in models_list:
    # 1) Initialize a new instance for each model class
    model_object_instance = ModelObject() # Initialize a new instance for each model
    # 2) Set the model
    model_object_instance.set_model_object(model_object=model)
    # 3) Set the data - Here we only want an In-sample comparison
    # therefore train and test data are the same
    model_object_instance.set_data(
        training_data=exchange_rates_vola_df,
        testing_data=exchange_rates_vola_df
    )
    # 4) Fit the model
    # based on the model class name extract additional parameters for the fit
    model_name = model.__class__.__name__
    fit_kwargs = fit_kwargs_dict.get(model_name, {})
    model_object_instance.fit(**fit_kwargs)
    # 5) Predict the labels - In sample forecast
    predicted_labels = model_object_instance.predict()
    # 6) Evaluate the model - pick the desired score to evaluate
    # Here we could also think of providing a list with multiple functions at once
    evaluation_score = model_object_instance.evaluate(metric_function_list=[silhouette_score])
    # Save the model and the model info with dynamic names based on the model class name
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{model.__class__.__name__}_{timestamp}"
    # First save the fitted model object itself to a .pkl file - we can also read it in later
    model_object_instance.save_model(
        file_format=r".pkl",
        file_path=MODELS_PATH,
        model_file_name=file_name
        )
    # Second save all the related metadata/information about the model - from here the relevant metadata will be pulled
    model_object_instance.get_full_model_info(
        save=True,
        return_info_dict=False,
        file_format=r".json",
        file_path=MODELS_PATH,
        file_name=file_name
        )
# For a proper evaluation now, loading all full_info.json files and compare the evaluation scores
all_models_comp_df = get_model_metadata_df(
    full_model_info_path=MODELS_PATH,
    )
all_models_comp_df[["model_type", "silhouette_score", "feature_names_in"]].sort_values(by="silhouette_score", ascending=False)
# all_models_comp_df.to_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models/all_models_comparison_1.xlsx", index=False)
predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
# Now also map the regimes across the models correctly, in that we assume the high vola regime is encoded as 1 and the low vola regime is a 0
# KMeans is already correctly encoded
# AgglomerativeClustering needs to be encoded
# MeanShift is also correctly encoded
# MarkovRegression needs to be encoded
predicted_labels_df = get_recoded_predicted_labels_df(
    predicted_labels_df=predicted_labels_df,
    label_mapping_dict={0: 1, 1: 0},
    column_names_list=["AgglomerativeClustering", "MarkovRegression"]
    )
# Compare the fitted in sample regimes
regime_counts_df = get_regime_counts_df(
    predicted_labels_df=predicted_labels_df
    )
# Also identify the time periods that are overlapping in the classification of the algorithms and count their share
predicted_labels_overlap_df = get_overlapping_regimes_df(
    predicted_labels_df=predicted_labels_df
    )
# Notice: Overlap is defined as two models having the exact same regime encoding so 1 == 1
# It might be that the regimes are just encoded differently, e.g. 0 == 1 and 1 == 0 for two models
# This is not captured in the above overlap calculation

# Plot them all together in a plotly graph, together with the target data
# Plot the regimes on a secondary y-axis
predicted_labels_df.index = exchange_rates_vola_df.index

# Now also load in the table of crisis periods and overlay these time periods as well
# to see if regime changes have been picked up
graphing_df = pd.concat(
    [exchange_rates_vola_df, predicted_labels_df],
    axis=1
)


with open("/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods_dict.json", "r") as f:
    crisis_periods_dict = json.load(f)

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(graphing_df.columns)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(graphing_df.columns.tolist())}


fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=graphing_df,
    crisis_periods_dict=crisis_periods_dict,
    variables=["rolling_vola"],
    secondary_y_variables=graphing_df.columns[1:].tolist(),
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {graphing_df.index[0].year} - {graphing_df.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility",
    color_mapping_dict=color_mapping,
    num_years_interval_x_axis=5,
    showlegend=True,
    save_fig=False,
    file_name="exchange_rate_oil_raw_vola_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
# evaluate numerically if the models were able to identify the crisis periods
# by checking how many of the crisis period data points were classified as high volatility regime
# by each model
crisis_periods_df = pd.DataFrame(crisis_periods_dict).T.reset_index().rename(columns={"index": "Crisis", "start": "Start-date", "end": "End-date"})
overlay_df = get_periods_overlaying_df(
    crisis_periods_df=crisis_periods_df,
    predicted_labels_df=predicted_labels_df,
    predicted_labels_df_column_names_list=predicted_labels_df.columns[1:].tolist(),
)
# Load a specific model/model object back - Here a MarkovRegression Model
new_model_object_instance = ModelObject()
fitted_model, model_info_dict, model_info_df = new_model_object_instance.load_model(
    file_format_model=".pkl",
    file_path_model=MODELS_PATH,
    model_file_name="MarkovRegression_2025-10-08_22-26-05",
    return_info_dict=True,
    return_info_df=True,
    file_format_model_info=".json",
    file_path_model_info=MODELS_PATH,
    model_file_name_info="MarkovRegression_2025-10-08_22-26-05"
    )
fitted_model.predict(start=0, end=len(model_info_dict.get("testing_data_dates"))-1)
# Extract the regime periods
model_info_dict.keys()
len(model_info_dict.get("testing_data_dates"))
model_info_dict.get("training_data")
model_info_dict.get("testing_data")
len(model_info_dict.get("predicted_labels"))
new_model_object_instance.fitted_model.summary()
#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Models with exogenous Variables for Regime Identification
#----------------------------------------------------------------------------------------
# Now try to also include exogeneous variables like WTI Oil Price vola and Henry Hub gas Vola and see if something changes
series_dict_mapping = {
    'WTI Oil': 'DCOILWTICO',
    "Nat Gas": "DHHNGSP",
}

start_date = exchange_rates_vola_df.index.min().strftime('%Y-%m-%d')
end_date = exchange_rates_vola_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_us_df = pd.concat(list(data_dict.values()), axis=1).dropna()
data_us_full_info_table = pd.DataFrame(data_full_info_dict).T



# Relevel the data to have the same dimensions
data_us_df = data_us_df.reindex(exchange_rates_vola_df.index).dropna()
data_us_df["log_WTI_Oil"] = np.log(data_us_df["WTI Oil"])
data_us_df["log_Nat_Gas"] = np.log(data_us_df["Nat Gas"])
data_us_df["log_WTI_Oil_diff"] = data_us_df["log_WTI_Oil"].diff()
data_us_df["log_Nat_Gas_diff"] = data_us_df["log_Nat_Gas"].diff()
data_us_df = data_us_df.dropna()
# Calculate rolling volatilities
window = 30
data_us_df["rolling_vola_oil"] = data_us_df["log_WTI_Oil_diff"].rolling(window=window).std().dropna()
data_us_df["rolling_vola_gas"] = data_us_df["log_Nat_Gas_diff"].rolling(window=window).std().dropna()
data = pd.concat([exchange_rates_vola_df, data_us_df], axis=1).dropna()

# Now apply the very same loop from earlier, but now with the exogeneous variable included
# We initialize a new instance of the ModelObject class for each model with desired settings
# We store them in a list that we will loop through
msm = MarkovRegression(
    endog=data[["rolling_vola"]],
    exog=data[['rolling_vola_oil', 'rolling_vola_gas']],
    k_regimes=n_regimes,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
) # <- remember to always create a new msm object if new data is used because else the dimensions will not align due to different data
models_list = [kmeans, agg, dbscan, ms, msm]
for model in models_list:
    # 1) Initialize a new instance for each model class
    model_object_instance = ModelObject() # Initialize a new instance for each model
    # 2) Set the model
    model_object_instance.set_model_object(model_object=model)
    # 3) Set the data - Here we only want an In-sample comparison
    # therefore train and test data are the same
    model_object_instance.set_data(
        training_data=data[["rolling_vola", "rolling_vola_oil", "rolling_vola_gas"]],
        testing_data=data[["rolling_vola", "rolling_vola_oil", "rolling_vola_gas"]]
    )
    # 4) Fit the model
    model_name = model.__class__.__name__
    fit_kwargs = fit_kwargs_dict.get(model_name, {})
    model_object_instance.fit(**fit_kwargs)
    # 5) Predict the labels - In sample forecast
    predicted_labels = model_object_instance.predict()
    # 6) Evaluate the model - pick the desired score to evaluate
    # Here we could also think of providing a list with multiple functions at once
    evaluation_score = model_object_instance.evaluate(metric_function_list=[silhouette_score])
    # Save the model and the model info with dynamic names based on the model class name
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{model.__class__.__name__}_{timestamp}"
    # First save the fitted model object itself
    model_object_instance.save_model(
        file_format=r".pkl",
        file_path=MODELS_PATH,
        model_file_name=file_name
        )
    # Second save all the related metadata/information about the model
    model_object_instance.get_full_model_info(
        save=True,
        return_info_dict=False,
        file_format=r".json",
        file_path=MODELS_PATH,
        file_name=file_name
        )
#################################################################################
# Again reevaluate
# For a proper evaluation now, loading all full_info files and compare the evaluation scores
all_models_comp_df = get_model_metadata_df(
    full_model_info_path=MODELS_PATH,
    )
all_models_comp_df[["model_type", "silhouette_score", "feature_names_in"]].sort_values(by="silhouette_score", ascending=False)
predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
# Plotting the results as a plotly bar plot
fig = go.Figure()
for model in all_models_comp_df["model_type"].unique():
    model_data = all_models_comp_df[all_models_comp_df["model_type"] == model]
    fig.add_trace(go.Bar(
        x=model_data["feature_names_in"],
        y=model_data["silhouette_score"],
        name=model
    ))
fig.update_layout(
    title="Model Comparison",
    xaxis_title="Feature Names",
    yaxis_title="Silhouette Score",
    barmode="group"
)
fig.show(renderer="browser")

#################################################################################
model_object_instance = ModelObject()
msm = MarkovRegression(
    endog=data["rolling_vola"],
    exog=data[['rolling_vola_oil']],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=False,
)
model_object_instance.set_model_object(model_object=msm)
model_object_instance.set_data(
    training_data=data[['rolling_vola']], # see comment above
    testing_data=data[['rolling_vola']] # see comment above
    )
model_object_instance.fit(em_iter=10, search_reps=20)
predicted_labels = model_object_instance.predict()
model_object_instance.evaluate(metric_function_list=[silhouette_score])
model_object_instance.fitted_model.summary()
# Get the smoothed probabilities for each regime
smoothed_probabilities = model_object_instance.fitted_model.smoothed_marginal_probabilities
data["msm_regime"] = smoothed_probabilities[1]
data["regime"] = (data["msm_regime"] >= 0.5).astype(int)
# Data Points per regime
data["regime"].value_counts()
rcm_value = compute_rcm(S=2, smoothed_probs=data["msm_regime"])
print(f"Regime Classification Measure (RCM): {rcm_value}")

regime_0_periods = data[data["regime"] == 0].index
regime_1_periods = data[data["regime"] == 1].index
print(f"Regime 0 periods: {regime_0_periods}")
print(f"Regime 1 periods: {regime_1_periods}")

# Plot the regimes together with the exchange rate changes as plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["rolling_vola"],
                         mode='lines', name='Rolling Volatility'))
fig.add_trace(go.Scatter(x=data.index, y=data["msm_regime"],
                         mode='lines', name='MSM Regime Probability', yaxis='y2'))
fig.update_layout(
    title='Markov Switching Model Regimes with Rolling Volatility',
    xaxis_title='Date',
    yaxis_title='Log US NEER Diff',
    yaxis2=dict(
        title='MSM Regime Probability',
        overlaying='y',
        side='right'    
    ),
    legend=dict(x=0, y=1)
)
fig.show(renderer="browser")
#################################################################################
model_object_instance = ModelObject()
model_object_instance.set_model_object(model_object=kmeans)
model_object_instance.set_data(
    training_data=data[['rolling_vola']], # see comment above
    testing_data=data[['rolling_vola']] # see comment above
    )
model_object_instance.fit()
dir(model_object_instance.fitted_model)
len(model_object_instance.fitted_model.labels_)
# Try to replicate the example from the Lecture Notes
# Page 3/16 Modeling Nonlinearities I: Markov-Switching
# A model incorporating GARCH effects in the return equation has been introduced by Engle, Lilien and Robins (1987)
# The so-called GARCH-M model:
# s_t - s_t-1 = beta_0 + beta_1 * (i_t-1 - i*_t-1) + beta_2 * h_t + u_t
# u_t = epsilon_t * sqrt(h_t)
# h_t = c + alpha * epsilon_t-1^2 + beta * h_t-1
# Of course, we expect beta_0 = 0, beta_1 = 1 and beta_2 > 0!
# The parameter estimates are calculated by numerical maximization of
# the log likelihood
# Estimate the model
import statsmodels.api as sm

msm_garch = sm.tsa.MarkovAutoregression(
    endog=exchange_rates_df["log_US_NEER_diff"],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
    order=1,
    switching_ar=True,
    garch_order=(1, 1)
)
msm_garch_fitted = msm_garch.fit(em_iter=10, search_reps=20)
print(msm_garch_fitted.summary())
# Get the smoothed probabilities for each regime
exchange_rates_df["msm_garch_regime"] = msm_garch_fitted.smoothed_marginal_probabilities[1]
# Plot the smoothed probabilities
exchange_rates_df["msm_garch_regime"]
# Evaluate the RCM for the Markov Switching GARCH model
rcm_garch_value = compute_rcm(S=2, smoothed_probs=exchange_rates_df["msm_garch_regime"])
print(f"Regime Classification Measure (RCM) for GARCH model: {rcm_garch_value}")
# Plot the regimes together with the exchange rate changes as plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=exchange_rates_df.index, y=exchange_rates_df["log_US_NEER_diff"],
                         mode='lines', name='Log US NEER Diff'))
fig.add_trace(go.Scatter(x=exchange_rates_df.index, y=exchange_rates_df["msm_garch_regime"],
                         mode='lines', name='MSM GARCH Regime Probability', yaxis='y2'))
fig.update_layout(
    title='Markov Switching GARCH Model Regimes',
    xaxis_title='Date',
    yaxis_title='Log US NEER Diff',
    yaxis2=dict(
        title='MSM GARCH Regime Probability',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1)
)
fig.show(renderer="browser")