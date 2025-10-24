from typing import Dict, List, Any
import os
import pandas as pd
import numpy as np
import ast
import json
import logging
from datetime import datetime
import re
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    model_type_counts = metadata_df["model_type"].value_counts().to_dict()
    model_type_seen = {}
    columns = []
    for i in range(len(metadata_df.index)):
        try:
            model = metadata_df["model_type"][i]
            time_last_fitted = metadata_df["time_last_fitted"][i]
            # Create a unique model name
            model_name = f"{model}_{time_last_fitted}"
            # Track how many times we've seen this model type
            model_type_seen[model] = model_type_seen.get(model, 0) + 1
            # If there are multiple occurrences, append a suffix
            # if model_type_counts[model] > 1:
            #     model_name = f"{model}_{model_type_seen[model]}"
            # else:
            #     model_name = model
            columns.append(model_name)
            predicted_labels_string = metadata_df[column_name_predicted_labels][i]
            # Check if the string is a list of only NaNs
            if re.fullmatch(r"\[ *(nan, *)*nan *\]", predicted_labels_string.replace(" ", "")):
                logging.warning(f"Predicted labels at index {i} are all NaN, skipping.")
                continue
            predicted_labels = ast.literal_eval(predicted_labels_string)
            # Optionally, check if all elements are NaN after eval
            if isinstance(predicted_labels, list) and all(pd.isna(x) for x in predicted_labels):
                logging.warning(f"Predicted labels at index {i} are all NaN after eval, skipping.")
                continue
            testing_data_dates_string = metadata_df["testing_data_dates"][i]
            testing_data_dates = ast.literal_eval(testing_data_dates_string)
            testing_data_dates = [pd.Timestamp(date) for date in testing_data_dates]
        except Exception as e:
            logging.error(f"Error parsing predicted labels or testing data dates for model at index {i}: {e}")
        try:
            model_df = pd.DataFrame(index=testing_data_dates, data=predicted_labels, columns=[model_name])
            predicted_labels_df = predicted_labels_df.merge(
                right=model_df,
                left_index=True,
                right_index=True,
                how="left"
            )
        except Exception as e:
            logging.error(f"Error adding predicted labels for model {model_name}: {e} because of differing length at index {i}")
            continue
    return predicted_labels_df


def get_recoded_predicted_labels_df(
    predicted_labels_df: pd.DataFrame,
    label_mapping_dict: Dict[int, int],
    column_names_list: List[str]
    ) -> pd.DataFrame:
    for model in column_names_list:
        if model in predicted_labels_df.columns:
            # Recode the regimes
            logging.info(f"Recoding predicted labels for model: {model}")
            predicted_labels_df[model] = predicted_labels_df[model].map(label_mapping_dict)
    return predicted_labels_df


# Count the number of data points per regime
# Create a dictionary to hold counts for each model
def get_regime_counts_df(
    predicted_labels_df: pd.DataFrame
    ) -> pd.DataFrame:
    regime_counts = {}
    for column in predicted_labels_df.columns:
        counts = predicted_labels_df[column].value_counts()
        # Ensure both regimes (0 and 1) are present
        counts = counts.reindex([0, 1], fill_value=0)
        regime_counts[column] = counts
    # Convert to DataFrame
    regime_counts_df = pd.DataFrame(regime_counts)
    regime_counts_df.index.name = "Regime"
    return regime_counts_df


def get_overlapping_regimes_df(
    predicted_labels_df: pd.DataFrame
    ) -> pd.DataFrame:
    predicted_labels_overlap_df = pd.DataFrame()
    for i in range(len(predicted_labels_df.columns)):
        for j in range(i + 1, len(predicted_labels_df.columns)):
            model_i = predicted_labels_df.columns[i]
            model_j = predicted_labels_df.columns[j]
            overlap = (predicted_labels_df[model_i] == predicted_labels_df[model_j]).sum()
            overlap_percentage = overlap / len(predicted_labels_df) * 100
            predicted_labels_overlap_df = pd.concat([predicted_labels_overlap_df, pd.DataFrame({
                "Model 1": [model_i],
                "Model 2": [model_j],
                "Overlap (absolute number of obs.)": [overlap],
                "Overlap (percentage of total obs.)": [round(overlap_percentage, 2)]
            })], ignore_index=True)
    return predicted_labels_overlap_df


def get_periods_overlaying_df(
    crisis_periods_df: pd.DataFrame,
    predicted_labels_df: pd.DataFrame,
    predicted_labels_df_column_names_list: List[str],
    ) -> pd.DataFrame:
    overlay_df = pd.DataFrame()
    for column in predicted_labels_df_column_names_list:
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
        overlay_df = pd.concat([
            overlay_df,
            pd.DataFrame({
                "Model": [column],
                "Total Crisis Data Points": [total_crisis_points],
                "High Volatility Crisis Points": [high_vol_crisis_points],
                "High Volatility Percentage": [high_vol_percentage]
            })
        ], ignore_index=True)
    return overlay_df


def adf_test(
    data: pd.Series,
    title: str = "Augmented Dickey-Fuller Test",
    regression_type: str = 'c',  # 'c' for constant, 'ct' for constant and trend, 'nc' for no constant
    autolag: str = 'AIC',  # 'AIC', 'BIC', 't-stat', None
    maxlag: int = None,
    return_regression_summary: bool = False,
    variable: str = None,
    significance_level: float = 0.05
    ) -> pd.DataFrame:
    """
    Performs the Augmented Dickey-Fuller (ADF) test on a time series and prints the results.

    Args:
        data (pd.Series): Time series data to test for stationarity.
        title (str): Title for the output.
        regression_type (str): Type of regression ('c', 'ct', 'ctt', 'n').
        autolag (str): Method to use for lag selection ('AIC', 'BIC', 't-stat', None).
        maxlag (int): Maximum number of lags to consider.
        variable (str): Name of the variable being tested.
        significance_level (float): Significance level for the test.
    Returns:
        pd.DataFrame: DataFrame containing the ADF test results.
    """
    if data.empty:
        raise ValueError("The provided Series is empty.")

    from statsmodels.tsa.stattools import adfuller
    from datetime import datetime
    data = data[variable].dropna()
    adf_stat, p_val, crit_vals, result = adfuller(
        x=data,
        maxlag=maxlag,
        regression=regression_type,
        autolag=autolag,
        store=True,
        regresults=False
        )
    regression_summary = result.resols.summary()
    print(f"{title}\n")
    print(f"ADF Statistic: {adf_stat}")
    print(f"p-value: {p_val}")
    print("Critical Values:")
    for key, value in crit_vals.items():
        print(f"   {key}: {value}")
    if p_val < significance_level:
        print(f"The null hypothesis can be rejected at the {significance_level*100}% significance level. The series is stationary.")
    else:
        print(f"The null hypothesis cannot be rejected at the {significance_level*100}% significance level. The series is non-stationary.")

    result_df = pd.DataFrame({
        'ADF Statistic': [adf_stat],
        'p-value': [p_val],
        '1% Critical Value': [crit_vals['1%']],
        '5% Critical Value': [crit_vals['5%']],
        '10% Critical Value': [crit_vals['10%']],
        'H0': [result.H0],
        'HA': [result.HA],
        'Used Lag:': [result.usedlag],
        'Max Lag:': [result.maxlag],
        'Start Time:': [f"{data.index.min().strftime('%d-%m-%Y')}"],
        'End Time:': [f"{data.index.min().strftime('%d-%m-%Y')}"],
        'Regression Type': [regression_type],
        'Observations:': [result.nobs],
        'Data': [data],
        'Variable': [variable],
        'Tested at': [f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"]
        })
    if return_regression_summary:    
        return result_df, regression_summary
    else:
        return result_df


def granger_causality_test(
    data: pd.DataFrame,
    variable_x: str,
    variable_y: str,
    max_lag: int = 10,
    significance_level: float = 0.05,
    test_type_p_value: str = 'ssr_ftest'
    ) -> None:
    """
    Performs the Granger Causality test between two time series and prints the results.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        variable_x (str): The name of the first variable (cause).
        variable_y (str): The name of the second variable (effect).
        max_lag (int): Maximum number of lags to test.
        significance_level (float): Significance level for the test.
        test_type_p_value (str): Type of test statistic to use for p-value ('ssr_ftest', 'ssr_chi2test', etc.).

    Returns:
        None
    """
    if data.empty:
        raise ValueError("The provided DataFrame is empty.")
    if variable_x not in data.columns or variable_y not in data.columns:
        raise ValueError(f"One or both variables '{variable_x}', '{variable_y}' are not in the DataFrame columns.")

    from statsmodels.tsa.stattools import grangercausalitytests
    print(f"Granger Causality Test between {variable_x} and {variable_y}\n")
    data = data[[variable_y, variable_x]].dropna()
    test_result = grangercausalitytests(data, maxlag=max_lag, verbose=True)
    granger_test_df_list = []
    for lag in range(1, max_lag + 1):
        lag_data = test_result[lag]
        lag_data = lag_data[0]
        granger_test_df = pd.DataFrame()
        for key in lag_data.keys():
            lag_diagnostics = pd.DataFrame(
                lag_data.get(key),
                ).T
            if lag_diagnostics.shape[1] == 3:
                columns = ['Test-Statistic', 'p-value', 'df_num']
            else:
                columns = ['Test-Statistic', 'p-value', 'df_denom', 'df_num']
            lag_diagnostics.columns = columns
            lag_diagnostics["Metric"] = key
            granger_test_df = pd.concat(
                [granger_test_df,
                lag_diagnostics],
                axis=0)
        granger_test_df["Lag"] = lag
        granger_test_df_list.append(granger_test_df)
        p_value = test_result[lag][0][test_type_p_value][1]
        if p_value < significance_level:
            print(f"Lag {lag}: Reject null hypothesis at {significance_level*100}% significance level. {variable_x} Granger-causes {variable_y}.")
        else:
            print(f"Lag {lag}: Cannot reject null hypothesis at {significance_level*100}% significance level. No Granger causality from {variable_x} to {variable_y}.")
    granger_test_df_complete = pd.concat(granger_test_df_list, axis=0).reset_index(drop=True)
    granger_test_df_complete['Start Time'] = f"{data.index.min().strftime('%d-%m-%Y')}"
    granger_test_df_complete['End Time'] = f"{data.index.max().strftime('%d-%m-%Y')}"
    granger_test_df_complete['Observations'] = data.shape[0]
    granger_test_df_complete["Variable X"] = variable_x
    granger_test_df_complete["Variable Y"] = variable_y
    # granger_test_df_complete["Data Variable X"] = [data[variable_x]]
    # granger_test_df_complete["Data Variable Y"] = [data[variable_y]]
    granger_test_df_complete['Test'] = f"{variable_x} causes {variable_y}"
    granger_test_df_complete['Tested at'] = f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"

    return test_result, granger_test_df_complete


def cointegration_test(
    data: pd.DataFrame,
    variable_x: str,
    variable_y: str,
    significance_level: float = 0.05,
    trend: str = 'c',
    method: str = 'aeg',
    maxlag: int = None,
    ) -> pd.DataFrame:
    """
    Performs the Engle-Granger cointegration test between two time series and prints the results.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        variable_x (str): The name of the first variable.
        variable_y (str): The name of the second variable.
        significance_level (float): Significance level for the test.

    Returns:
        None
    """
    if data.empty:
        raise ValueError("The provided DataFrame is empty.")
    if variable_x not in data.columns or variable_y not in data.columns:
        raise ValueError(f"One or both variables '{variable_x}', '{variable_y}' are not in the DataFrame columns.")

    from statsmodels.tsa.stattools import coint
    data = data[[variable_x, variable_y]].dropna()
    cointegration_test_result = coint(
        data[variable_x],
        data[variable_y],
        trend=trend,
        method=method,
        maxlag=maxlag
        )
    print(f"Cointegration Test between {variable_x} and {variable_y}\n")
    print(f"Cointegration Score: {cointegration_test_result[0]}")
    print(f"p-value: {cointegration_test_result[1]}")
    if cointegration_test_result[1] < significance_level:
        print(f"The null hypothesis can be rejected at the {significance_level*100}% significance level. The series are cointegrated.")
    else:
        print(f"The null hypothesis cannot be rejected at the {significance_level*100}% significance level. The series are not cointegrated.")
    cointegration_df = pd.DataFrame({
        'Cointegration Score': [cointegration_test_result[0]],
        'p-value': [cointegration_test_result[1]],
        'Start Time': [f"{data.index.min().strftime('%d-%m-%Y')}"],
        'End Time': [f"{data.index.max().strftime('%d-%m-%Y')}"],
        'Observations': [data.shape[0]],
        'Trend': [trend],
        'Method': [method],
        'Max Lag': [maxlag],
        'Variable X': [variable_x],
        'Variable Y': [variable_y],
        "Data": [data],
        'Test': [f'{variable_x} and {variable_y}'],
        'Tested at': [f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"]
        })
    return cointegration_df


def test_data_for_normality(
    data: pd.DataFrame,
    variables: List[str],
    significance_level: float = 0.05,
    test_shapiro_wilks: bool = True,
    test_anderson_darling: bool = True,
    test_kolmogorov_smirnov: bool = True,
    test_dagostino_k2: bool = True
    ) -> pd.DataFrame:
    normality_test_results = pd.DataFrame(columns=["Variable", "Test", "Statistic", "p-value"])
    for variable in variables:
        # logging.info(f"Testing normality for variable: {variable}")
        if test_shapiro_wilks:
            # Shapiro-Wilk Test
            # The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
            stat, p = shapiro(data[variable])
            test_df = pd.DataFrame({"Variable": variable, "Test": "Shapiro-Wilk", "Statistic": stat, "p-value": p}, index=[0])
            normality_test_results = pd.concat([normality_test_results, test_df], ignore_index=True)

        if test_anderson_darling:
            # Anderson-Darling Test
            # The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population that follows a particular distribution.
            # Default is norm.
            result = anderson(data[variable])
            test_df = pd.DataFrame({"Variable": variable, "Test": "Anderson-Darling", "Statistic": result.statistic, "p-value": result.critical_values[2]}, index=[0])
            normality_test_results = pd.concat([normality_test_results, test_df], ignore_index=True)

        if test_kolmogorov_smirnov:
            # Kolmogorov-Smirnov Test
            # The one-sample test compares the underlying distribution F(x) of a sample against a given distribution G(x).
            stat, p = kstest(data[variable], 'norm')
            test_df = pd.DataFrame({"Variable": variable, "Test": "Kolmogorov-Smirnov", "Statistic": stat, "p-value": p}, index=[0])
            normality_test_results = pd.concat([normality_test_results, test_df], ignore_index=True)

        if test_dagostino_k2:
            # D'Agostino's K^2 Test
            # This function tests the null hypothesis that a sample comes from a normal distribution.
            stat, p = normaltest(data[variable])
            test_df = pd.DataFrame({"Variable": variable, "Test": "D'Agostino's K^2", "Statistic": stat, "p-value": p}, index=[0])
            normality_test_results = pd.concat([normality_test_results, test_df], ignore_index=True)

    normality_test_results["Significance-level"] = significance_level
    normality_test_results[f"p-value < {significance_level}"] = normality_test_results["p-value"] < significance_level
    normality_test_results["Result"] = np.where(normality_test_results["p-value"] < significance_level, "Not-Normal", "Normal")
    return normality_test_results


def extract_best_params(
    results_list: List[Dict[str, Any]],
    score_name: str = "silhouette"
    ) -> List[Dict[str, Any]]:
    """Extract best parameters from GridSearchCV results."""
    best_params_list = []

    for i, result in enumerate(results_list):
        results_df = pd.DataFrame(result)
        # Find best parameters
        rank_col = f'rank_test_{score_name}'
        mean_score_col = f'mean_test_{score_name}'

        if rank_col in results_df.columns:
            best_row = results_df[results_df[rank_col] == 1].iloc[0]
        else:
            # Fallback to highest score
            best_idx = results_df[mean_score_col].idxmax()
            best_row = results_df.loc[best_idx]

        best_params_list.append({
            'data_index': i,
            'best_params': best_row['params'],
            'best_score': best_row[mean_score_col],
            'score_std': best_row[f'std_test_{score_name}'],
            'feature_names_in': best_row['feature_names_in'],
            'model_name': best_row['model_name']
        })

    return best_params_list