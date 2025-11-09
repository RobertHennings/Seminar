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
    def __init__(
        self,
        data_list: List[pd.DataFrame],
        param_grids_dict: Dict[str, Dict[str, List[Any]]],
        model_class_dict: Dict[str, Any],
        scoring_dict: Dict[str, Any],
        ) -> None:
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


    def run_grid_search_cv(
        self,
        grid_search_refit_scorer: str="silhouette",
        grid_search_return_train_score: bool=True,
        grid_search_cv: int | BaseCrossValidator | Iterable | None = None,
        grid_search_n_jobs: int | None = None,
        grid_search_pre_dispatch: str | int = "2*n_jobs",
        grid_search_error_score: float | str = np.nan,
        ) -> List[Dict[str, Any]]:
        results = []
        for model_name, cls in self.model_class_dict.items():
            cls_instance = cls()
            logging.info(f"Model: {model_name}, Class: {cls}")
            for i, data in enumerate(self.data_list):
                try:
                    logging.info(f"Fitting Model: {model_name}, Class: {cls}, Data: {i}")
                    clf = GridSearchCV(
                        estimator=cls_instance,
                        param_grid=self.param_grids_dict[model_name],
                        scoring=self.scoring_dict,
                        refit=grid_search_refit_scorer,
                        return_train_score=grid_search_return_train_score,
                        cv=grid_search_cv,
                        n_jobs=grid_search_n_jobs,
                        pre_dispatch=grid_search_pre_dispatch,  
                        error_score=grid_search_error_score
                        )
                    clf.fit(data)
                    clf_results = clf.cv_results_
                    # Add additional data info
                    clf_results["model_name"] = [model_name] * len(clf_results["params"])
                    clf_results["data_index"] = [i] * len(clf_results["params"])
                    clf_results["feature_names_in"] = [data.columns.tolist()] * len(clf_results["params"])
                    results.append(clf_results)
                except Exception as e:
                    logging.error(f"Model {model_name} failed to fit: {e}")
                    continue
        return results


    def save_grid_search_results(
        self,
        results: List[Dict[str, Any]],
        file_path: str=None,
        file_name: str=None,
        ) -> None:
        """Save GridSearchCV results as JSON, converting numpy arrays to lists."""
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
        """Load GridSearchCV results from JSON."""
        full_path_load_json = f"{file_path}/{file_name}.json"
        try:
            with open(file=full_path_load_json, mode="r") as f:
                results = json.load(f)
            logging.info(f"Loaded results from {full_path_load_json}")
            return results
        except Exception as e:
            logging.error(f"Error loading from .json: {e}")
            return []

class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(
        self,
        n_splits=5,
        max_train_size=None,
        test_size=None,
        gap=0
        ) -> None:
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X,
        y=None,
        groups=None
        ) -> Iterable:
        """Generate indices to split data into training and test set."""
        n_samples = len(X)

        # Calculate sizes
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # Calculate test indices
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            test_indices = indices[test_start:test_end]

            # Calculate train indices (up to gap before test)
            train_end = test_start - self.gap
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            if train_end <= train_start:
                continue

            train_indices = indices[train_start:train_end]

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None
        ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class SkfolioWrapper(BaseCrossValidator):
    def __init__(
        self,
        n_folds: int=5,
        n_test_folds: int=2,
        embargo_size: int=30,
        purged_size: float=0.05
        ) -> None:
        self.skfolio_cv = CombinatorialPurgedCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            embargo_size=embargo_size,
            purged_size=purged_size
        )
        self.n_folds = n_folds

    def split(
        self,
        X,
        y=None,
        groups=None
        ):
        # Convert skfolio's output to sklearn format
        try:
            # This might need adjustment based on skfolio's actual API
            for train_idx, test_idx in self.skfolio_cv.split(X):
                yield train_idx, test_idx
        except Exception as e:
            # Fallback to simple time series split
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            for train_idx, test_idx in tscv.split(X):
                yield train_idx, test_idx

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None
        ):
        return self.n_folds