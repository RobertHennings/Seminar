from typing import List, Dict
import json
import os
import pandas as pd
import zipfile
import io
import requests
import datetime as dt
os.chdir(r"/Users/Robert_Hennings/Uni/Master/MasterThesis/src/library_name/data_generation")
import config as config
BASE_URL = "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip"
ERROR_CODES_DICT = cfg.ERROR_CODES_DICT
SUCCESS_CODES_DICT = cfg.SUCCESS_CODES_DICT
REQUEST_TIMEOUT = 90
COUNTRY_NAMES = ["Germany"]


class EUWholesalePricesLoader:
    def __init__(
            self,
            base_url: str=BASE_URL,
            country_names: List[str]=COUNTRY_NAMES,
            error_codes_dict: Dict[str, Dict[str, str]]=ERROR_CODES_DICT,
            success_codes_dict: Dict[str, Dict[str, str]]=SUCCESS_CODES_DICT,
            request_timeout: int=REQUEST_TIMEOUT,
            ):
        self.base_url = base_url
        self.country_names = country_names
        self.error_codes_dict = error_codes_dict
        self.success_codes_dict = success_codes_dict
        self.request_timeout = request_timeout


    # #################### internal helper methods ####################
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
            print(f"Status code: {status_code} not defined")
        else: # if status code is definded in the dicts
            # get the defined message for the status code
            status_code_message = f"{"".join(status_code_message)} for URL: {response_url}"
            # get the defined return (type) for the status code
            status_code_return = [
                dict_.get(status_code).get("return_")
                for dict_ in [self.error_codes_dict, self.success_codes_dict]
                if status_code in dict_.keys()]

            if status_code_return is not None:
                print(status_code_message)
            else:
                raise Exception("Error")


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
            print(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            print(f"Folder: {folder_name} created in path: {path}")


    def load_prices(self) -> Dict[str, pd.DataFrame]:
        # Set up the file list for the given countries
        country_file_list = [f"{country}.csv" for country in self.country_names]
        # Set up the saving dictionary to return, each country as key, resulting pd.DataFrame as value
        response = requests.get(url=self.base_url, timeout=self.request_timeout)
        # self.__resilient_request(response=response)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                file_list = z.namelist()
                print("Files in the ZIP:", file_list)

                # Iterate over the country file list and load data for each country
                self.country_data = {}
                for country_file in country_file_list:
                    country_name = country_file.replace(".csv", "")
                    matching_file = [f for f in file_list if country_file in f]
                    if matching_file:
                        with z.open(matching_file[0]) as f:
                            self.country_data[country_name] = pd.read_csv(f)
                            print(f"{country_file} loaded successfully!")
                    else:
                        print(f"{country_file} not found in the ZIP.")
                return self.country_data
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")


eu_wholesale_prices_loader_instance = EUWholesalePricesLoader()
eu_wholesale_prices_hourly = eu_wholesale_prices_loader_instance.load_prices()
germany_data = eu_wholesale_prices_hourly.get("Germany")
# Display the first few rows of the Germany data
germany_data["Datetime (UTC)"] = pd.to_datetime(germany_data["Datetime (UTC)"])

data_dict = {
    "Data": germany_data.to_dict(orient="records"),
    "Accessed": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

file_name = f"de_wholesale_electricity_price_data_hourly.json"
full_path_save = f"/Users/Robert_Hennings/Uni/Master/MasterThesis/data/raw/{file_name}"
with open(full_path_save, "w") as json_file:
    json.dump(data_dict, json_file, indent=4, ensure_ascii=False)
