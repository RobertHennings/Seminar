import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from datetime import timedelta

os.chdir(r"/Users/Robert_Hennings/Uni/Master/MasterThesis/src/library_name/data_generation")
from ORDERBOOK_GENERATION import OrderBookGenerator
# Load the day ahead prices - will be used as mid point from which to generate the order book
file_path = r"/Users/Robert_Hennings/Uni/Master/MasterThesis/data/raw/de_wholesale_electricity_price_data_hourly.json"
with open(file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)
germany_data = pd.DataFrame(json_data.get("Data"))
start_datetime_str = germany_data["Datetime (UTC)"][0].strftime('%Y-%m-%d %H:%M:%S')
end_datetime_str = germany_data["Datetime (UTC)"][germany_data.shape[0]-1].strftime('%Y-%m-%d %H:%M:%S')
day_ahead_prices = germany_data[["Price (EUR/MWhe)", "Datetime (UTC)"]].set_index("Datetime (UTC)", drop=True)

full_range = pd.date_range(start=start_datetime_str, end=end_datetime_str, freq="H")
missing_hours = full_range.difference(germany_data["Datetime (UTC)"])
print(f"Missing hours: {missing_hours}")

valid_hours = full_range.difference(missing_hours) # since in the real dataset some hours are missing, we need to filter them out
day_ahead_prices = germany_data.query("`Datetime (UTC)` in @valid_hours")
day_ahead_prices = day_ahead_prices['Price (EUR/MWhe)'].tolist()

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
country = "DE"
save_order_book_path = rf"/Users/Robert_Hennings/Uni/Master/MasterThesis/data/raw"
save_order_book_file_name = f"Continous_Intraday_OrderBook_60Minutes_Product_{country}_{now}.json"


order_book_generator_instance = OrderBookGenerator(
    day_ahead_prices=day_ahead_prices,
    valid_hours=valid_hours,
    save_order_book=True,
    save_order_book_path=save_order_book_path,
    save_order_book_file_name=save_order_book_file_name
    )
order_book = order_book_generator_instance.generate_intraday_order_book()
order_book_generator_instance.__save_order_book(order_book=order_book)
