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
                        self.labels = self.fitted_model.predict()
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
    if file_path is not None and file_name is not None:
        if file_format == ".txt":
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
        elif file_format == ".json":
            try:
                with open(fr"{file_path}/{file_name}{file_format}", "r") as f:
                    model_info = json.load(f)
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

#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Benchmark Models for Regime Identification
#----------------------------------------------------------------------------------------
print(os.getcwd())
os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code")
print(os.getcwd())

from data_loading.data_loader import DataLoading

data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)
# The regime classification measure (RCM) of Ang and Bekaert (2002) is used
# to determine the accuracy of the Markov-switching models. This statistic is computed using the following formula:
# The RCM is computed as the average of the product of smoothed probabilities p~; where S is the number of
# regimes (states, S). The switching variable follows a Bernoulli distribution and as a result, the RCM provides an
# estimate of the variance. The RCM statistic ranges be- tween 0 (perfect regime classification) and 100
# (failure to detect any re- gime classification) with lower values of the RCM preferable to higher values of the RCM.
# Thus to ensure significantly different regimes, it is important that a model's RCM is close to zero and its
# smoothed proba- bility indicator be close to 1.
def compute_rcm(S: int, smoothed_probs: pd.Series, threshold: float=0.5) -> float:
    """Compute the Regime Classification Measure (RCM) for a series of smoothed probabilities.

    Args:
        S (int): The number of regimes.
        smoothed_probs (pd.Series): The smoothed probabilities for the regimes.
        threshold (float, optional): The threshold to classify regimes. Defaults to 0.5.

    Examples:
        from model import compute_rcm
        rcm = compute_rcm(S=2, smoothed_probs=exchange_rates_df["msm_regime"])

    Returns:
        float: The RCM value.
    """
    regime_classification = (smoothed_probs >= threshold).astype(int)
    rcm = 100 * S**2 * (1 - (1 / len(smoothed_probs)) * np.sum(regime_classification * smoothed_probs + (1 - regime_classification) * (1 - smoothed_probs)))
    return rcm
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
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import silhouette_score

# At first we assume three regimes: One low volatility regime and two high volatility regimes
n_regimes = 2
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
models_list = [kmeans, agg, dbscan, ms, msm]
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
# For a proper evaluation now, loading all full_info files and compare the evaluation scores
def get_model_metadata_df(
    full_model_info_path: str,
    file_format: str=".json"
    ) -> pd.DataFrame:
    # If some entries are strings, convert them to dicts
    def safe_dict(x):
        if isinstance(x, dict):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    all_model_info_files = [file for file in os.listdir(full_model_info_path) if file.endswith(file_format)]
    logging.info(f"Found {len(all_model_info_files)} model info files in {full_model_info_path}")
    # Delete the file extension for loading
    all_model_info_files = [os.path.splitext(file)[0] for file in all_model_info_files]
    # Load all model info files into a list of dataframes
    all_model_info_dfs = [load_full_model_info(
        file_format=file_format,
        file_path=full_model_info_path,
        file_name=file,
        return_info_dict=False,
        return_info_df=True,
        ) for file in all_model_info_files]
    all_models_comp_df = pd.concat(all_model_info_dfs).reset_index(drop=True)
    all_models_comp_df["evaluation_score_dict"] = all_models_comp_df["evaluation_score"].apply(safe_dict)
    expanded_scores = all_models_comp_df["evaluation_score_dict"].apply(pd.Series)
    all_models_comp_df = pd.concat([all_models_comp_df, expanded_scores], axis=1)
    return all_models_comp_df

all_models_comp_df = get_model_metadata_df(
    full_model_info_path=MODELS_PATH,
    )
all_models_comp_df["testing_data_dates"]
# metadata_df = all_models_comp_df
# Apply conversion
all_models_comp_df[["model_type", "silhouette_score"]].sort_values(by="silhouette_score", ascending=False)
# all_models_comp_df.to_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models/all_models_comparison_1.xlsx", index=False)

# Extract all the predicted regimes and compare them in a plot
# When there are multiple models that were trained on different lengths of data sets then
# ther resulting predicted labels will have different lengths
# Therefore we need to align them first

def extract_predicted_labels_from_metadata_df(
    metadata_df: pd.DataFrame,
    column_name_predicted_labels: str="predicted_labels"
    ) -> pd.DataFrame:
    predicted_labels_df = pd.DataFrame()
    # Set the initial and longest index of all for later merging correctly
    len_model_dates = [len(ast.literal_eval(metadata_df["testing_data_dates"][i])) for i in range(len(metadata_df.index))]
    # get the index of the max
    max_length_index = np.argmax(len_model_dates)
    predicted_labels_df_index = ast.literal_eval(metadata_df["testing_data_dates"][max_length_index])
    predicted_labels_df.index = [pd.Timestamp(date) for date in predicted_labels_df_index]
    for i in range(len(metadata_df.index)):
        model = metadata_df["model_type"][i]
        if model in predicted_labels_df.columns:
            model = f"{model}_{i}"
        predicted_labels_string = metadata_df[column_name_predicted_labels][i]
        predicted_labels = ast.literal_eval(predicted_labels_string)
        # also extract the testing_data_dates for a correct matching of the predicted_lables data
        testing_data_dates_string = metadata_df["testing_data_dates"][i]
        testing_data_dates = ast.literal_eval(testing_data_dates_string)
        testing_data_dates = [pd.Timestamp(date) for date in testing_data_dates]
        # create a small DataFrame for a correct merging
        model_df = pd.DataFrame(index=testing_data_dates, data=predicted_labels, columns=[model])
        try:
            predicted_labels_df = predicted_labels_df.merge(
                right=model_df,
                left_index=True,
                right_index=True
            )
        except Exception as e:
            logging.error(f"Error adding predicted labels for model {model}: {e} because of differing length: {len(predicted_labels)}")
            continue
    return predicted_labels_df

predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
# Transform the values of the Markov Model
smoothed_probabilities = model_object_instance.fitted_model.smoothed_marginal_probabilities
msm_regime = smoothed_probabilities[1]
regime = (msm_regime >= 0.5).astype(int)
predicted_labels_df["MarkovRegression"] = regime.tolist()
# Now also map the regimes across the models correctly, in that we assume the high vola regime is encoded as 1 and the low vola regime is a 0
# KMeans is already correctly encoded
# AgglomerativeClustering needs to be encoded
# MeanShift is also correctly encoded
# MarkovRegression needs to be encoded
models_recoding_list = ["AgglomerativeClustering", "MarkovRegression"]
for model in models_recoding_list:
    if model in predicted_labels_df.columns:
        # Recode the regimes
        predicted_labels_df[model] = predicted_labels_df[model].map({0: 1, 1: 0})
# DBSCAN has some -1 values for noise points

# Compare the fitted in sample regimes
# Count the number of data points per regime
# Create a dictionary to hold counts for each model
regime_counts = {}
for column in predicted_labels_df.columns:
    counts = predicted_labels_df[column].value_counts()
    # Ensure both regimes (0 and 1) are present
    counts = counts.reindex([0, 1], fill_value=0)
    regime_counts[column] = counts
# Convert to DataFrame
regime_counts_df = pd.DataFrame(regime_counts)
regime_counts_df.index.name = "Regime"
print(regime_counts_df)
# Also identify the time periods that are overlapping in the classification of the algorithms and count their share
predicted_labels_overlap_df = pd.DataFrame()
for i in range(len(predicted_labels_df.columns)):
    for j in range(i + 1, len(predicted_labels_df.columns)):
        model_i = predicted_labels_df.columns[i]
        model_j = predicted_labels_df.columns[j]
        overlap = (predicted_labels_df[model_i] == predicted_labels_df[model_j]).sum()
        overlap_percentage = overlap / len(predicted_labels_df) * 100
        print(f"Overlap between {model_i} and {model_j}: {overlap} data points ({overlap_percentage:.2f}%)")
        predicted_labels_overlap_df = pd.concat([predicted_labels_overlap_df, pd.DataFrame({
            "Model 1": [model_i],
            "Model 2": [model_j],
            "Overlap (absolute number of obs.)": [overlap],
            "Overlap (percentage of total obs.)": [round(overlap_percentage, 2)]
        })], ignore_index=True)
    print("\n")
# Notice: Overlap is defined as two models having the exact same regime encoding so 1 == 1
# It might be that the regimes are just encoded differently, e.g. 0 == 1 and 1 == 0 for two models
# This is not captured in the above overlap calculation

# Plot them all together in a plotly graph, together with the target data
# Plot the regimes on a secondary y-axis
predicted_labels_df.index = exchange_rates_vola_df.index
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=exchange_rates_vola_df.index, y=exchange_rates_vola_df["rolling_vola"],
                         mode='lines', name='Rolling Volatility'))
for column in predicted_labels_df.columns:
    fig.add_trace(go.Scatter(x=predicted_labels_df.index, y=predicted_labels_df[column],
                             mode='lines', name=column, yaxis='y2'))
fig.update_layout(
    title='Clustering Model Regimes with Rolling Volatility',
    xaxis_title='Date',
    yaxis_title='Log US NEER Diff',
    yaxis2=dict(
        title='Regime',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1)
)
fig.show(renderer="browser")
# Now also load in the table of crisis periods and overlay these time periods as well
# to see if regime changes have been picked up
crisis_periods_df = pd.read_json(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods.json")
crisis_periods_df["Start-date"] = pd.to_datetime(crisis_periods_df["Start-date"])
crisis_periods_df["End-date"] = pd.to_datetime(crisis_periods_df["End-date"])
# Restric the crisis periods to the time frame of the exchange rate data
crisis_periods_df = crisis_periods_df[
    (crisis_periods_df["End-date"] >= exchange_rates_vola_df.index.min()) &
    (crisis_periods_df["Start-date"] <= exchange_rates_vola_df.index.max())
    ].reset_index(drop=True)
# Add these periods as shaded areas in the plotly graph
for i, row in crisis_periods_df.iterrows():
    fig.add_vrect(
        x0=row["Start-date"], x1=row["End-date"],
        fillcolor="LightSalmon", opacity=0.5,
        layer="below", line_width=0,
        annotation_text=row["Event-Type"], annotation_position="top left"
    )
fig.show(renderer="browser")

# evaluate numerically if the models were able to identify the crisis periods
# by checking how many of the crisis period data points were classified as high volatility regime
# by each model
for column in predicted_labels_df.columns:
    print(f"Model: {column}")
    total_crisis_points = 0
    high_vol_crisis_points = 0
    for i, row in crisis_periods_df.iterrows():
        mask = (predicted_labels_df.index >= row["Start-date"]) & (predicted_labels_df.index <= row["End-date"])
        crisis_points = predicted_labels_df.loc[mask, column]
        total_crisis_points += len(crisis_points)
        if len(crisis_points) > 0:
            # Assuming high volatility regime is encoded as '1'
            high_vol_crisis_points += (crisis_points == 1).sum()
    if total_crisis_points > 0:
        high_vol_percentage = (high_vol_crisis_points / total_crisis_points) * 100
    else:
        high_vol_percentage = 0
    print(f"Total crisis data points: {total_crisis_points}")
    print(f"High volatility regime during crisis periods: {high_vol_crisis_points} data points ({high_vol_percentage:.2f}%)")
    print("\n")



# Load a specific model/model object back - Here a MarkovRegression Model
new_model_object_instance = ModelObject()
fitted_model, model_info_dict, model_info_df = new_model_object_instance.load_model(
    file_format_model=".pkl",
    file_path_model=MODELS_PATH,
    model_file_name="MarkovRegression_2025-10-07_15-00-36",
    return_info_dict=True,
    return_info_df=True,
    file_format_model_info=".json",
    file_path_model_info=MODELS_PATH,
    model_file_name_info="MarkovRegression_2025-10-07_15-00-36"
    )
# Extract the regime periods
model_info_dict.keys()
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
data_us_df["log_WTI_Oil_diff"] = data_us_df["log_WTI_Oil"].diff()
data_us_df = data_us_df["log_WTI_Oil_diff"].rolling(window=window).std().dropna().to_frame(name="rolling_vola_oil")
data = pd.concat([exchange_rates_vola_df, data_us_df], axis=1).dropna()

# Now apply the very same loop from earlier, but now with the exogeneous variable included
# We initialize a new instance of the ModelObject class for each model with desired settings
# We store them in a list that we will loop through
models_list = [kmeans, agg, dbscan, ms, msm]
for model in models_list:
    # 1) Initialize a new instance for each model class
    model_object_instance = ModelObject() # Initialize a new instance for each model
    # 2) Set the model
    model_object_instance.set_model_object(model_object=model)
    # 3) Set the data - Here we only want an In-sample comparison
    # therefore train and test data are the same
    model_object_instance.set_data(
        training_data=data[["rolling_vola", "rolling_vola_oil"]],
        testing_data=data[["rolling_vola", "rolling_vola_oil"]]
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
all_models_comp_df["evaluation_score_dict"] = all_models_comp_df["evaluation_score"].apply(safe_dict)
all_models_comp_df["silhouette_score"] = all_models_comp_df["evaluation_score_dict"].apply(extract_score)
all_models_comp_df[["model_type", "silhouette_score", "feature_names_in"]].sort_values(by="silhouette_score", ascending=False)
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