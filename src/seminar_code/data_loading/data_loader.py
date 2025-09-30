from typing import Dict, List, Tuple
import os
import zipfile
import io
import logging
import json
from ecbdata import ecbdata
import datetime as dt
import pandas as pd
from fredapi import Fred
import yfinance as yf
import requests

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import config settings with static global variables
# os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/data_loading")
try:
    from . import config as cfg
except:
    import config as cfg

SAVE_PATH = cfg.SAVE_PATH
PARAMS_MAPPING_DICT = cfg.PARAMS_MAPPING_DICT
STORAGE_OPTIONS_DICT = cfg.STORAGE_OPTIONS_DICT
REQUEST_TIMEOUT = cfg.REQUEST_TIMEOUT
ERROR_CODES_DICT = cfg.ERROR_CODES_DICT
SUCCESS_CODES_DICT = cfg.SUCCESS_CODES_DICT
BIS_EXCHANGE_RATE_MAPPING_DICT = cfg.BIS_EXCHANGE_RATE_MAPPING_DICT
BIS_DATA_BASE_URL = cfg.BIS_DATA_BASE_URL

class DataLoading(object):
    """The main purpose of this class is to provide a one stop shop for
       loading all the necessary data from the various sources used.
    """
    def __init__(
        self,
        credential_path: str=None,
        credential_file_name: str=None,
        request_timeout: int=REQUEST_TIMEOUT,
        proxies: Dict[str, str]=None,
        verify: bool=None,
        error_codes_dict: Dict[int, Dict[str, str]]=ERROR_CODES_DICT,
        success_codes_dict: Dict[int, Dict[str, str]]=SUCCESS_CODES_DICT,
        bis_exchange_rate_mapping_dict: Dict[str, str]=BIS_EXCHANGE_RATE_MAPPING_DICT,
        bis_data_base_url: str=BIS_DATA_BASE_URL,
        ):
        self.credential_path = credential_path
        self.credential_file_name = credential_file_name
        self.request_timeout = request_timeout
        self.error_codes_dict = error_codes_dict
        self.success_codes_dict = success_codes_dict
        self.proxies = proxies
        self.verify = verify
        self.bis_exchange_rate_mapping_dict = bis_exchange_rate_mapping_dict
        self.bis_data_base_url = bis_data_base_url
        # For some APIs credentials are needed, these are loaded here from a local file
        if (self.credential_path and self.credential_file_name) is not None:
            # trigger credential loading
            credentials = self.__load_credentials()
            self.credentials = credentials
            # initialize the different API Clients
            fred_api_key = credentials.get("FRED_API", None)
            us_bureau_api_key = credentials.get("US_Bureau_of_Labor_Statistics", None)

            if fred_api_key is not None:
                self.fred = Fred(api_key=fred_api_key)
            else:
                logging.info(f"Fred_API not defined in credentials file: {self.credential_file_name} in path: {self.credential_path}")
    ######################### Internal helper methods #########################
    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved
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


    def __load_credentials(
        self
        ):
        # First check if a secrets file is already present at the provided path
        if self.credential_file_name is not None and self.credential_path is not None:
            self.__check_path_existence(path=self.credential_path)
            if self.credential_file_name in os.listdir(self.credential_path):
                file_path_full = f"{self.credential_path}/{self.credential_file_name}"
                with open(file_path_full, encoding="utf-8") as json_file:
                    credentials = json.load(json_file)
                logging.info(f"Credentials loaded from file: {self.credential_file_name} in path: {self.credential_path}")
                return credentials
            else:
                raise KeyError(f"{self.credential_file_name} not found in path: {self.credential_path}")
        else:
            logging.info(f"No credentials provided, missing the file path: {self.credential_path} and/or the file name: {self.credential_file_name}")


    ### Further robustness checks
    def __resilient_request(
        self,
        response: requests.models.Response,
        ):
        """Internal helper method - serves as generous requests
           checker using the custom defined error and sucess code dicts
           for general runtime robustness

        Args:
            response (requests.models.Response): generous API response

        Raises:
            Exception: _description_

        """
        status_code = response.status_code
        response_url = response.url
        status_code_message = [
            dict_.get(status_code).get("message")
            for dict_ in [self.error_codes_dict, self.success_codes_dict]
            if status_code in dict_.keys()]
        # If status code is not present in the defined dicts
        if status_code_message == []:
            logging.info(f"Status code: {status_code} not defined")
        else: # if status code is definded in the dicts
            # get the defined message for the status code
            status_code_message = f"{"".join(status_code_message)} for URL: {response_url}"
            # get the defined return (type) for the status code
            status_code_return = [
                dict_.get(status_code).get("return_")
                for dict_ in [self.error_codes_dict, self.success_codes_dict]
                if status_code in dict_.keys()]

            if status_code_return is not None:
                logging.info(status_code_message)
            else:
                raise Exception("Error")


    ######################### FRED Data #########################
    def __normalize_freq_fred(
        self,
        freq: str
        ) -> str:
        """Normalize frequency strings for FRED API requests. Since we often want
           to load multiple series with different frequencies, we need to
           normalize them to a common set of frequency indicators (i.e. resample them).
           In order to directly see what is the common lowest frequency, we normalize
           the frequency strings to a common set of indicators.

        Args:
            freq (str): The frequency string to normalize.

        Returns:
            str: The normalized frequency string.
        """
        if freq is None:
            return None
        # Take only the first letter for common cases (D, W, M, Q, A, etc.)
        # For more complex cases, you may want to use regex or more logic
        if freq.startswith('QS') or freq.startswith('Q'):
            return 'Q'
        if freq.startswith('MS') or freq.startswith('M'):
            return 'M'
        if freq.startswith('AS') or freq.startswith('A'):
            return 'A'
        if freq.startswith('W'):
            return 'W'
        if freq.startswith('D') or freq.startswith('B'):
            return 'D'
        return freq  # fallback


    def get_fred_data(
        self,
        series_dict_mapping: Dict[str, str],
        start_date: str,
        end_date: str
        ) -> Tuple[Dict[str, pd.Series], Dict[str, str], Dict[str, str], Dict[str, str], pd.DataFrame]:
        """Fetch data from FRED API for multiple series at once.

        Args:
            series_dict_mapping (Dict[str, str]): Mapping of series names to their IDs.
            start_date (str): Start date for the data retrieval.
            end_date (str): End date for the data retrieval.

        Returns:
            Tuple[Dict[str, pd.Series], Dict[str, str], Dict[str, str], Dict[str, str], pd.DataFrame]: A tuple containing the data series, full info, frequency info, and a DataFrame of the combined data.
        """
        data_dict = {}
        data_full_info_dict = {}
        data_freq_dict = {}
        for name, sid in series_dict_mapping.items():
            logging.info(f"Fetching data for variable {name} with series id: ({sid}) for start date: {start_date} and end date: {end_date}")
            data_series = self.fred.get_series(
                series_id=sid,
                observation_start=start_date,
                observation_end=end_date).rename(name)
            info_dict = self.fred.get_series_info(series_id=sid)
            data_freq_dict[name] = info_dict.loc["frequency_short"]
            info_dict["url"] = f"https://fred.stlouisfed.org/series/{sid}"
            info_dict["Last Accessed"] = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
            data_full_info_dict[name] = info_dict
            actual_start_date = data_series.index.min()
            actual_end_date = data_series.index.max()
            if actual_start_date > pd.Timestamp(start_date):
                logging.info(f"Actual start date for loaded data variable: {name} with series id: ({sid}): {actual_start_date} instead of {start_date}")
            if actual_end_date < pd.Timestamp(end_date):
                logging.info(f"Actual end date for loaded data variable: {name} with series id: ({sid}): {actual_end_date} instead of {end_date}")

            try:
                frequency = pd.infer_freq(data_series.index)
            except Exception as e:
                logger.warning(f"Could not infer frequency for {name} due to error: {e}")
                frequency = None
            if frequency is not None:
                logging.info(f"Inferred frequency for {name} is {frequency}")
                data_freq_dict[name] = frequency
            else:
                logger.warning(f"Could not infer frequency for {name}, filling missing dates with forward fill.")
            data_dict[name] = data_series

        data_freq_dict_list = list(data_freq_dict.values())
        freq_order = ['D', 'W', 'M', 'Q', 'A']
        normalized_freqs = [self.__normalize_freq_fred(f) for f in data_freq_dict_list if f is not None]
        lowest_freq = max(normalized_freqs, key=lambda x: freq_order.index(x))
        logging.info(f"Lowest (slowest) frequency: {lowest_freq}")
        return data_dict, data_full_info_dict, lowest_freq


    ######################### World in Data - Oil and Gas Consumption/Production #########################
    def get_oil_gas_prod_con_data(
        self,
        series_dict_mapping: Dict[str, str],
        params_mapping_dict: Dict[str, List[str]]=PARAMS_MAPPING_DICT,
        storage_options_dict: Dict[str, str]=STORAGE_OPTIONS_DICT
        ) -> Tuple[Dict[str, pd.Series], Dict[str, str], Dict[str, str], Dict[str, str], pd.DataFrame]:
        """Fetch oil and gas production/consumption data from World in Data.

        Args:
            series_dict_mapping (Dict[str, str]): Mapping of series names to their URLs.
            params_mapping_dict (Dict[str, List[str]], optional): Mapping of parameter names to their values. Defaults to PARAMS_MAPPING_DICT.
            storage_options_dict (Dict[str, str], optional): Storage options for the data retrieval. Defaults to STORAGE_OPTIONS_DICT.

        Returns:
            Tuple[Dict[str, pd.Series], Dict[str, str], Dict[str, str], Dict[str, str], pd.DataFrame]: A tuple containing the data series, full info, frequency info, and a DataFrame of the combined data.
        """
        data_dict = {}
        data_full_info_dict = {}
        data_freq_dict = {}
        for name, url in series_dict_mapping.items():
            data_url = f"{url}{params_mapping_dict.get('data', '')}"
            metadata_url = f"{url}{params_mapping_dict.get('metadata', '')}"
            logging.info(f"Fetching data for variable {name} with url: ({data_url}).")
            data_df = pd.read_csv(
                filepath_or_buffer=data_url,
                storage_options=storage_options_dict)
            data_dict[name] = data_df
            info_dict = requests.get(url=metadata_url, timeout=self.request_timeout).json()
            info_dict = info_dict.get("columns")
            keys_list = list(info_dict.keys())
            info_dict = info_dict.get(keys_list[0])
            info_dict["url"] = data_url
            info_dict["Last Accessed"] = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
            data_full_info_dict[name] = info_dict

        return data_dict, data_full_info_dict


    def get_cftc_commitment_of_traders(
        self,
        start_date: str,
        end_date: str,
        base_url: str=r"https://www.cftc.gov/files/dea/history/",
        report_type: str="Futures-and-Options Combined Reports",
        save_files_locally: bool=False,
        save_path: str=None,
        file_ending_dict: Dict[str, str]=None,
        ) -> pd.DataFrame:
        """
        Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm#:~:text=(Text%2C%20Excel)-,Futures%2Dand%2DOptions%20Combined%20Reports:,the%20new%20format%20by%20year.

        Args:
            base_url (str): _description_
            start_date (str): _description_
            end_date (str): _description_
            report_type (str, optional): _description_. Defaults to "Futures-and-Options Combined Reports".
            save_files_locally (bool, optional): _description_. Defaults to False.
            save_path (str, optional): _description_. Defaults to None.
            file_ending_dict (Dict[str, str], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_

        Examples:

        """
        year_list_files = pd.date_range(start=start_date, end=end_date, freq="YS").year.to_list()
        file_ending_dict = {
            "Futures-and-Options Combined Reports": "deahistfo",
            "Commodity Index Trader Supplement": "",
            "Traders in Financial Futures ; Futures-and-Options Combined Reports": "",
            "Traders in Financial Futures ; Futures Only Reports": "",
            "Disaggregated Futures-and-Options Combined Reports": "",
            "Disaggregated Futures Only Reports": ""
        }
        file_ending = file_ending_dict.get(report_type, None)
        if file_ending is not None:
            url_list = [f"{base_url}{file_ending}_{year}.zip" for year in year_list_files]
        else:
            raise ValueError(f"Report type '{report_type}' not supported.")
        combined_data = []
        # https://www.cftc.gov/files/dea/history/deahistfo_2004.zip
        # url = "https://www.cftc.gov/files/dea/history/deahistfo_2005.zip"
        # url = url_list[0]
        # Retrieve the zip files from the urls and save them locally
        for url in url_list:
            # logging.info(f"Downloading file from URL: {url}")
            print(f"Downloading file from URL: {url}")
            response = requests.get(url, timeout=self.request_timeout)
            if response.status_code == 404:
                try:
                    # logging.info(f"File for year {url.split('_')[-1].split('.')[0]} failed, trying different naming convention.")
                    print(f"File for year {url.split('_')[-1].split('.')[0]} failed, trying different naming convention.")
                    url_changed = "".join(url.split('_'))
                    print(f"Trying URL: {url_changed}")
                    response = requests.get(url_changed, timeout=self.request_timeout)
                    if response.status_code == 200:
                        print(f"File for year {url.split('_')[-1].split('.')[0]} successfully loaded with changed URL")
                except:
                    print(f"File for year not found")
            # self.__connection_handler(response=response)
            # self.__resilient_request(response=response)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_name in z.namelist():
                    # logging.info(f"Extracting file: {file_name}")
                    print(f"Extracting file: {file_name}")
                    with z.open(file_name) as file:
                        # Assuming the file is a CSV, adjust if it's another format
                        df = pd.read_csv(file)
                        combined_data.append(df)

        # Combine all data into a single DataFrame
        if combined_data != []:
            final_df = pd.concat(combined_data, ignore_index=True).reset_index(drop=True)
            final_df["Market and Exchange Names"] = final_df["Market and Exchange Names"].str.strip()
        else:
            logging.warning("No data was extracted from the zip files.")
            final_df = pd.DataFrame()
        if save_files_locally and save_path is not None:
            self.__check_path_existence(path=save_path)
            file_name = f"{report_type}_{start_date}_to_{end_date}.csv"
            file_path_full = f"{save_path}/{file_name}"
            final_df.to_csv(file_path_full, index=False)
            logging.info(f"Data saved locally at: {file_path_full}")
        return final_df


    def get_yahoo_data(
        self,
        ticker: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        auto_adjust: bool=False,
        interval: str="1d"
        ) -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance.

        Args:
            ticker (List[str]): List of stock tickers to fetch data for.
            start_date (pd.Timestamp): Start date for the data retrieval.
            end_date (pd.Timestamp): End date for the data retrieval.
            auto_adjust (bool, optional): Whether to automatically adjust prices for splits and dividends. Defaults to False.
            interval (str, optional): Data interval frequency. Defaults to "1d".

        Returns:
            pd.DataFrame: Historical market data for the specified tickers and date range.
        """
        tickers_connected = " ".join(ticker)
        print("FUNCTION: get_yahoo_data")
        data = yf.download(
            tickers_connected,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            interval=interval,
            timeout=self.request_timeout,
            )
        return data


    def filter_yahoo_data(
        self,
        yahoo_df: pd.DataFrame,
        columns: List[str]
        ) -> pd.DataFrame:
        """Filter the Yahoo Finance DataFrame to keep only the specified columns.

        Args:
            yahoo_df (pd.DataFrame): The original Yahoo Finance DataFrame.
            columns (List[str]): List of columns to keep.

        Returns:
            pd.DataFrame: A filtered DataFrame with only the specified columns.
        """
        logging.info("FUNCTION: filter_yahoo_data")
        return yahoo_df[columns]


    def get_ecb_data(
        self,
        series_dict_mapping: Dict[str, str],
        start_date: str,
        end_date: str,
        detail: str=None,
        updatedafter: str=None,
        firstnobservations: int=None,
        lastnobservations: int=None,
        includehistory: bool=False
        ) -> Dict[str, pd.Series]:
        """Fetch historical data from the ECB.

        Args:
            series_dict_mapping (Dict[str, str]): Mapping of variable names to series IDs.
            start_date (str): Start date for the data retrieval.
            end_date (str): End date for the data retrieval.
            detail (str, optional): Level of detail for the data. Defaults to None.
            updatedafter (str, optional): Fetch data updated after this date. Defaults to None.
            firstnobservations (int, optional): Fetch only the first n observations. Defaults to None.
            lastnobservations (int, optional): Fetch only the last n observations. Defaults to None.
            includehistory (bool, optional): Whether to include historical data. Defaults to False.

        Returns:
            Dict[str, pd.Series]: A dictionary mapping variable names to their historical data series.
        """
        data_dict = {}
        for name, sid in series_dict_mapping.items():
            logging.info(f"Fetching data for variable {name} with series id: ({sid}) for start date: {start_date} and end date: {end_date}")
            data_dict[name] = ecbdata.get_series(
                series_key=sid,
                start=start_date,
                end=end_date,
                detail=detail,
                updatedafter=updatedafter,
                firstnobservations=firstnobservations,
                lastnobservations=lastnobservations,
                includehistory=includehistory
                )
        return data_dict


    def get_bis_exchange_rate_data(
        self,
        country_keys_mapping: Dict[str, str],
        dataflow: str="WS_EER",
        exchange_rate_type_list: List[str]=["Real effective exchange rate - monthly - broad basket"],
        ) -> pd.DataFrame:
        """Fetch exchange rate data from the BIS API.

        Args:
            country_keys_mapping (Dict[str, str]): Mapping of country names to their keys.
            dataflow (str, optional): The data flow to use. Defaults to "WS_EER".
            exchange_rate_type (List[str], optional): List of exchange rate types to fetch. Defaults to ["Real effective exchange rate - monthly - broad basket"].
            keep_columns (List[str], optional): List of columns to keep in the final DataFrame. Defaults to ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"].

        Returns:
            pd.DataFrame: A DataFrame containing the fetched exchange rate data.
        """
        # Construct the single urls
        converted_rate_types = [self.bis_exchange_rate_mapping_dict.get(rate_type, None) for rate_type in exchange_rate_type_list]
        urls = [f"{self.bis_data_base_url}data/dataflow/BIS/{dataflow}/1.0/{rate_type}{country}?format=csv" for country in list(country_keys_mapping.keys()) for rate_type in converted_rate_types if rate_type is not None]
        exchange_rate_type_long_list = [f"{country}_{rate}" for country in list(country_keys_mapping.keys()) for rate in exchange_rate_type_list if rate is not None]

        master_df = pd.DataFrame()
        for i in range(len(urls)):
            country_url = urls[i]
            column_name = exchange_rate_type_long_list[i]
            logging.info(f"Fetching data from URL: {country_url}")
            country_df = pd.read_csv(country_url)
            country_df_pivoted = country_df.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
            country_df_pivoted.columns = [column_name]
            
            if master_df.empty:
                master_df = country_df_pivoted
            else:
                master_df = pd.concat([master_df, country_df_pivoted], ignore_index=False, axis=1)

        master_df.index = pd.to_datetime(master_df.index, format="%Y-%m")
        master_df = master_df.sort_index()
        return master_df

        