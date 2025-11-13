from typing import List, Dict, Any, Iterable
import os
import logging
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from skfolio.model_selection import CombinatorialPurgedCV
# Set up warnings filter
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimiser:
    """
    A basic class performing GridSearch for a variety of given algorithms and datasets.
    """
    def __init__(
        self,
        data_list: List[pd.DataFrame],
        param_grids_dict: Dict[str, Dict[str, List[Any]]],
        model_class_dict: Dict[str, Any],
        scoring_dict: Dict[str, Any],
        ) -> None:
        """The class variables shared by all class methods.

        Args:
            data_list (List[pd.DataFrame]): The data-sets to be used for model optimisation.
            param_grids_dict (Dict[str, Dict[str, List[Any]]]): Expects that the Dict Keys are
                                                                exactly named as the model
                                                                algorithm name class, i.e. "KMeans".
                                                                The model algorithm value is then
                                                                itself again a dict holding the value
                                                                ranges that shall be tried out for GridSearch.
            model_class_dict (Dict[str, Any]): The Dict matching the model class name to the bare model class instance.
            scoring_dict (Dict[str, Any]): The Dict holding the scoring metrics to be used for model evaluation.
        """
        self.data_list = data_list
        self.param_grids_dict = param_grids_dict
        self.model_class_dict = model_class_dict
        self.scoring_dict = scoring_dict


    def __check_path_existence(
        self,
        path: str
        ) -> None:
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

    # In order to speed up the GridSearchCV process, parallelizing is advised
    def _grid_search_worker(
        self,
        model_name: str,
        cls: type,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        scoring_dict: Dict[str, Any],
        refit: str,
        return_train_score: bool,
        cv,
        pre_dispatch,
        error_score,
        data_index: int,
        per_task_n_jobs: int = 1,
        ):
        """Top-level worker for one (model_name, data) GridSearchCV run.
        This must be a module-level function so it is picklable for ProcessPoolExecutor.
        """
        try:
            est = cls()
            # use per_task_n_jobs=1 to avoid nested joblib parallelism unless you know what you're doing
            clf = GridSearchCV(
                estimator=est,
                param_grid=param_grid,
                scoring=scoring_dict,
                refit=refit,
                return_train_score=return_train_score,
                cv=cv,
                n_jobs=per_task_n_jobs,
                pre_dispatch=pre_dispatch,
                error_score=error_score,
            )
            clf.fit(data)
            clf_results = clf.cv_results_
            clf_results["model_name"] = [model_name] * len(clf_results["params"])
            clf_results["data_index"] = [data_index] * len(clf_results["params"])
            clf_results["feature_names_in"] = [data.columns.tolist()] * len(clf_results["params"])
            return clf_results
        except Exception as e:
            logging.error(f"Worker failed for model {model_name}, data {data_index}: {e}")
            return None


    def run_grid_search_cv(
        self,
        grid_search_refit_scorer: str="silhouette",
        grid_search_return_train_score: bool=True,
        grid_search_cv: int | BaseCrossValidator | Iterable | None = None,
        grid_search_n_jobs_outer: int | None = None,
        grid_search_per_task_n_jobs: int | None = 1,
        grid_search_pre_dispatch: str | int = "2*n_jobs",
        grid_search_error_score: float | str = np.nan,
        ) -> List[Dict[str, Any]]:
        """Parallelised outer loop using ProcessPoolExecutor.
        - grid_search_n_jobs_outer controls how many (model,data) tasks run in parallel.
        - grid_search_per_task_n_jobs controls n_jobs inside each GridSearchCV (set to 1 to avoid nested parallelism).
        """
        results = []
        tasks = []
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare tasks (model_name, cls, data, ...)
        for model_name, cls in self.model_class_dict.items():
            for i, data in enumerate(self.data_list):
                param_grid = self.param_grids_dict.get(model_name)
                if param_grid is None:
                    logging.warning(f"No param_grid for model {model_name}, skipping.")
                    continue
                tasks.append((model_name, cls, data, param_grid, self.scoring_dict,
                            grid_search_refit_scorer, grid_search_return_train_score,
                            grid_search_cv, grid_search_pre_dispatch, grid_search_error_score, i, grid_search_per_task_n_jobs))

        # Run tasks in a process pool
        max_workers = grid_search_n_jobs_outer or min(len(tasks), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            future_to_task = {
                exe.submit(self._grid_search_worker, *task): task for task in tasks
            }
            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    logging.error(f"Task failed {task[0]} data {task[10]}: {e}")
                    continue

        return results
    # def run_grid_search_cv(
    #     self,
    #     grid_search_refit_scorer: str="silhouette",
    #     grid_search_return_train_score: bool=True,
    #     grid_search_cv: int | BaseCrossValidator | Iterable | None = None,
    #     grid_search_n_jobs: int | None = None,
    #     grid_search_pre_dispatch: str | int = "2*n_jobs",
    #     grid_search_error_score: float | str = np.nan,
    #     ) -> List[Dict[str, Any]]:
    #     """The main class method performing the GridSearchCV for all given algorithms,
    #        for all given datasets, with the specified scoring metrics and cross-validation
    #        strategy and the provided value ranges for each parameter of a model class.

    #     Args:
    #         grid_search_refit_scorer (str, optional): Dict key specifying what algorithm to use for refitting.
    #                                                   Must be included as key in the class variable scoring_dict.
    #                                                   Defaults to "silhouette".
    #         grid_search_return_train_score (bool, optional): Whether to return the training score. Defaults to True.
    #         grid_search_cv (int | BaseCrossValidator | Iterable | None, optional): The cross-validation strategy to use. Defaults to None.
    #         grid_search_n_jobs (int | None, optional): The number of jobs to run in parallel. Defaults to None.
    #         grid_search_pre_dispatch (str | int, optional): Controls the number of jobs that are dispatched during parallel execution. Defaults to "2*n_jobs".
    #         grid_search_error_score (float | str, optional): The error score to assign to a failed fit. Defaults to np.nan.

    #     Returns:
    #         List[Dict[str, Any]]: A list of dictionaries containing the results of the grid search.
    #     """
    #     results = []
    #     for model_name, cls in self.model_class_dict.items():
    #         cls_instance = cls()
    #         logging.info(f"Model: {model_name}, Class: {cls}")
    #         for i, data in enumerate(self.data_list):
    #             try:
    #                 logging.info(f"Fitting Model: {model_name}, Class: {cls}, Data: {i}")
    #                 clf = GridSearchCV(
    #                     estimator=cls_instance,
    #                     param_grid=self.param_grids_dict[model_name],
    #                     scoring=self.scoring_dict,
    #                     refit=grid_search_refit_scorer,
    #                     return_train_score=grid_search_return_train_score,
    #                     cv=grid_search_cv,
    #                     n_jobs=grid_search_n_jobs,
    #                     pre_dispatch=grid_search_pre_dispatch,  
    #                     error_score=grid_search_error_score
    #                     )
    #                 clf.fit(data)
    #                 clf_results = clf.cv_results_
    #                 # Add additional data info
    #                 clf_results["model_name"] = [model_name] * len(clf_results["params"])
    #                 clf_results["data_index"] = [i] * len(clf_results["params"])
    #                 clf_results["feature_names_in"] = [data.columns.tolist()] * len(clf_results["params"])
    #                 results.append(clf_results)
    #             except Exception as e:
    #                 logging.error(f"Model {model_name} failed to fit: {e}")
    #                 continue
    #     return results


    def save_grid_search_results(
        self,
        results: List[Dict[str, Any]],
        file_path: str=None,
        file_name: str=None,
        ) -> None:
        """Save GridSearchCV results as JSON, converting numpy arrays to lists.

        Args:
            results (List[Dict[str, Any]]): The results of the grid search.
            file_path (str, optional): The directory to save the JSON file. Defaults to None.
            file_name (str, optional): The name of the JSON file. Defaults to None.
        """
        self.__check_path_existence(path=file_path)
        full_path_save_json = f"{file_path}/{file_name}.json"

        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    elif isinstance(value, np.integer):
                        serializable_result[key] = int(value)
                    elif isinstance(value, np.floating):
                        serializable_result[key] = float(value)
                    elif isinstance(value, list):
                        # Handle lists that might contain numpy objects
                        serializable_result[key] = [
                            item.tolist() if isinstance(item, np.ndarray) else
                            int(item) if isinstance(item, np.integer) else
                            float(item) if isinstance(item, np.floating) else
                            item for item in value
                        ]
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)

            # Save to JSON
            with open(file=full_path_save_json, mode="w") as f:
                json.dump(serializable_results, f, indent=2)
            logging.info(f"Exported results to {full_path_save_json}")

        except Exception as e:
            logging.error(f"Error exporting to .json: {e}")


    def load_grid_search_results(
        self,
        file_path: str=None,
        file_name: str=None,
        ) -> List[Dict[str, Any]]:
        """Load GridSearchCV results from JSON.

        Args:
            file_path (str, optional): The file path to the results file. Defaults to None.
            file_name (str, optional): The file name of the results file. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the results of the grid search.
        """
        full_path_load_json = f"{file_path}/{file_name}.json"
        try:
            with open(file=full_path_load_json, mode="r") as f:
                results = json.load(f)
            logging.info(f"Loaded results from {full_path_load_json}")
            return results
        except Exception as e:
            logging.error(f"Error loading from .json: {e}")
            return []
