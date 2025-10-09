from typing import Dict, List, Any
import os
import pandas as pd
import numpy as np
import ast
import json
import logging

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
        model = metadata_df["model_type"][i]
        # Track how many times we've seen this model type
        model_type_seen[model] = model_type_seen.get(model, 0) + 1
        # If there are multiple occurrences, append a suffix
        if model_type_counts[model] > 1:
            model_name = f"{model}_{model_type_seen[model]}"
        else:
            model_name = model
        columns.append(model_name)
        predicted_labels_string = metadata_df[column_name_predicted_labels][i]
        predicted_labels = ast.literal_eval(predicted_labels_string)
        testing_data_dates_string = metadata_df["testing_data_dates"][i]
        testing_data_dates = ast.literal_eval(testing_data_dates_string)
        testing_data_dates = [pd.Timestamp(date) for date in testing_data_dates]
        try:
            model_df = pd.DataFrame(index=testing_data_dates, data=predicted_labels, columns=[model_name])
            predicted_labels_df = predicted_labels_df.merge(
                right=model_df,
                left_index=True,
                right_index=True,
                how="left"
            )
        except Exception as e:
            logging.error(f"Error adding predicted labels for model {model_name}: {e} because of differing length: {len(predicted_labels)} at index {i}")
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