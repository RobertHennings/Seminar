import logging
import os
import inspect
from typing import Dict, Any, List
import sklearn
import pandas as pd
import numpy as np
import joblib
import json

MARKOV_REGIME_PROB = 0.5
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
                        start = self.testing_data.index[0]
                        end = self.testing_data.index[-1]
                        self.labels = self.fitted_model.predict(start=start, end=end)
                        if self.model_object.__class__.__name__ == "MarkovRegression":
                            # Convert the probabilities to regimes based on a threshold
                            # Get the smoothed probabilities for all regimes
                            smoothed_probabilities = self.fitted_model.smoothed_marginal_probabilities
                            # Each column in smoothed_probabilities is a regime
                            # Assign each time point to the regime with the highest probability
                            regime = smoothed_probabilities.idxmax(axis=1)
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
