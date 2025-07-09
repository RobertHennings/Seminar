from typing import Dict, List
import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
os.chdir(r"/Users/Robert_Hennings/Uni/Master/MasterThesis/src/library_name/data_generation")
import config as cfg

BASE_URL = "https://www.epexspot.com/en/market-results"
ERROR_CODES_DICT = cfg.ERROR_CODES_DICT
SUCCESS_CODES_DICT = cfg.SUCCESS_CODES_DICT
REQUEST_TIMEOUT = 90


class EpexSpotScraper(object):
    def __init__(
            self,
            error_codes_dict: Dict[int, Dict[str, str]]=ERROR_CODES_DICT,
            success_codes_dict: Dict[int, Dict[str, str]]=SUCCESS_CODES_DICT,
            base_url: str=BASE_URL,
            headers: Dict[str, str]=None,
            proxies: Dict[str, str]=None,
            verify: bool=None
            ):
        self.base_url = base_url
        self.error_codes_dict = error_codes_dict
        self.success_codes_dict = success_codes_dict
        self.headers = headers
        self.proxies = proxies
        self.verify = verify
        self.request_timeout = REQUEST_TIMEOUT


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

    def get_epex_spot_data(
            self,
            params: Dict[str, str]=None,
            ) -> requests.models.Response:
        """Generous request method for the EPEX SPOT API

        Args:
            url (str): URL to the EPEX SPOT API endpoint
            params (Dict[str, str], optional): URL parameters. Defaults to None.
            headers (Dict[str, str], optional): Request headers. Defaults to None.
            proxies (Dict[str, str], optional): Proxies for the request. Defaults to None.
            verify (bool, optional): SSL verification. Defaults to None.

        Returns:
            requests.models.Response: Response object from the request
        """
        response = requests.get(
            url=self.base_url,
            params=params,
            headers=self.headers,
            proxies=self.proxies,
            verify=self.verify,
            timeout=self.request_timeout
        )
        self.__resilient_request(response=response)
        return response


    def __generate_custom_time_intervals(self) -> List[str]:
        intervals = []

        # Define the structure of intervals
        for hour in range(24):
            next_hour = hour + 1

            # Hourly interval
            intervals.append(f"{hour:02d} - {next_hour:02d}")

            # Sub-hourly intervals
            intervals.append(f"{hour:02d}:00 - {hour:02d}:30")
            intervals.append(f"{hour:02d}:00 - {hour:02d}:15")
            intervals.append(f"{hour:02d}:15 - {hour:02d}:30")
            intervals.append(f"{hour:02d}:30 - {next_hour:02d}:00")
            intervals.append(f"{hour:02d}:30 - {hour:02d}:45")
            intervals.append(f"{hour:02d}:45 - {next_hour:02d}:00")

        return intervals


    def clean_epex_spot_data(
            self,
            response: requests.models.Response,
            ) -> pd.DataFrame:
        """Generous request method for the EPEX SPOT API

        Args:
            url (str): URL to the EPEX SPOT API endpoint
            params (Dict[str, str], optional): URL parameters. Defaults to None.
            headers (Dict[str, str], optional): Request headers. Defaults to None.
            proxies (Dict[str, str], optional): Proxies for the request. Defaults to None.
            verify (bool, optional): SSL verification. Defaults to None.

        Returns:
            requests.models.Response: Response object from the request
        """
        soup = BeautifulSoup(response.text, "lxml")
        # Find all the table column names
        table_headers = soup.find_all("th")

        # Extract and clean the text from each <th> element
        cleaned_headers = [" ".join(th.text.split()).strip() for th in table_headers]
        cleaned_headers = [header for header in cleaned_headers if header != ""]  # Remove empty strings
        # cleaned_headers = ["Hours"] + cleaned_headers  # Add "Hours" at the beginning

        # Extract the hours from the div with class "fixed-column js-table-times"
        # Find the div containing the hours
        hours_div = soup.find("div", class_="fixed-column js-table-times")
        # Find all <li> elements inside the <ul>
        hours_lis = hours_div.find_all("li")

        # Extract the text from each <a> tag inside <li>
        hours = [li.a.text.strip() for li in hours_lis]

        # Trading Product
        trading_product_element = "h2"
        h2_element = soup.find(trading_product_element, string=lambda t: t and "Continuous" in t and "60min" in t and "DE" in t)
        if h2_element:
            # Clean and extract the bare string
            trading_product_element_bare_string = " ".join(h2_element.text.split()).replace(">", "").strip()
            trading_product_element_bare_string = trading_product_element_bare_string.replace("  ", " ")
        else:
            trading_product_element_bare_string = ""
            print(f"Element: {trading_product_element} not found")

        # Last updated
        last_update_class_ = "last-update"
        last_update_element = soup.find("span", class_=last_update_class_)
        if last_update_element:
            # Clean and extract the bare string
            last_update_bare_string = " ".join(last_update_element.text.split()).strip()
        else:
            last_update_bare_string = ""
            print(f"Class: {last_update_class_} not found")

        # Extract the table rows
        rows = soup.select("tbody tr")
        if rows:
            data = []
            for row in rows:
                cols = [td.text.strip() for td in row.find_all("td")]
                data.append(cols)

            df = pd.DataFrame(data, columns=cleaned_headers)
            time_intervals = self.__generate_custom_time_intervals()
            df.insert(0, "Hours", time_intervals)  # Insert "Hours" column at the beginning
            return df, trading_product_element_bare_string, last_update_bare_string


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.epexspot.com/",
}
epex_spot_scraper_instance = EpexSpotScraper(headers=headers)
country = "DE"
date = "2025-02-02"
product = "60"
params = {
    "market_area": country,
    "auction": "",
    "trading_date": "",
    "delivery_date": date,
    "underlying_year": "",
    "modality": "Continuous",
    "sub_modality": "",
    "technology": "",
    "data_mode": "table",
    "period": "",
    "production_period": "",
    "product": product,
}

raw_epex_spot_data_response = epex_spot_scraper_instance.get_epex_spot_data(params=params)
cleaned_epex_spot_data, trading_product, last_update = epex_spot_scraper_instance.clean_epex_spot_data(response=raw_epex_spot_data_response)
print(cleaned_epex_spot_data)
# Filter for the full hour rows
hour_df = cleaned_epex_spot_data[cleaned_epex_spot_data["Hours"].str.match(r"^\d{2} - \d{2}$")]
print(hour_df)
