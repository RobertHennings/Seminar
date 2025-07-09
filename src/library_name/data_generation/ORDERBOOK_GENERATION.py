from typing import List, Dict
import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import zipfile
import io
import requests
import pytz
import uuid

class OrderBookGenerator:
    def __init__(
            self,
            day_ahead_prices: List[float],
            valid_hours: List[datetime],
            timezone_offset_hours=1, # CET = UTC+1, adjust for summer time if needed
            tick_size=0.01, # minimum tick size EUR/MW
            volume_increment=0.1, # minimum volume increment MW
            min_order_book_levels=20,  # Minimum levels at all times
            max_order_book_levels=40,  # Maximum levels at all times
            base_volatility=0.02,  # EUR/MWh, base price offset
            save_order_book=False,
            save_order_book_path="order_book_data",
            save_order_book_file_name="order_book_data.json",
            ):
        self.day_ahead_prices = day_ahead_prices
        self.valid_hours = valid_hours
        self.timezone_offset_hours = timezone_offset_hours
        self.tick_size = tick_size
        self.volume_increment = volume_increment
        self.min_order_book_levels = min_order_book_levels
        self.max_order_book_levels = max_order_book_levels
        self.base_volatility = base_volatility
        self.max_volatility = 0.20
        self.save_order_book = save_order_book
        self.save_order_book_path = save_order_book_path
        self.save_order_book_file_name = save_order_book_file_name
        self.results = []

    # #################### internal helper methods ####################
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

    def __save_order_book(
        self,
        order_book: List[Dict]
        ):
        full_path_save = f"{self.save_order_book_path}/{self.save_order_book_file_name}"
        self.__check_path_existence(path=self.save_order_book_path)
        with open(full_path_save, "w") as json_file:
            json.dump(order_book, json_file, indent=2)
        print(f"Order book saved to {full_path_save}")


    # def generate_intraday_order_book(
    #     self,
    #     valid_hours,  # List of valid timestamps
    #     day_ahead_prices,
    #     timezone_offset_hours=1  # CET = UTC+1, adjust for summer time if needed
    #     ) -> List[Dict[str, str]]:
    #     """
    #     Generate synthetic order book data for the German intraday market.

    #     Args:
    #         valid_hours (list of datetime): List of valid timestamps to process.
    #         day_ahead_prices (list of floats): Realized hourly Day-Ahead prices.
    #         timezone_offset_hours (int): Offset from UTC to CET (default 1).

    #     Returns:
    #         list: List of dictionaries representing order book snapshots in JSON format.
    #     """
    #     results = []
    #     tick_size = self.tick_size
    #     volume_increment = self.volume_increment
    #     min_order_book_levels = self.min_order_book_levels  # Minimum levels at all times
    #     max_order_book_levels = self.max_order_book_levels  # Maximum levels at all times

    #     if len(day_ahead_prices) != len(valid_hours):
    #         raise ValueError(
    #             f"Mismatch between number of valid hours ({len(valid_hours)}) and length of day_ahead_prices ({len(day_ahead_prices)})."
    #         )

    #     for hour_idx, delivery_time in enumerate(valid_hours):
    #         percentage_complete = ((hour_idx + 1) / len(valid_hours)) * 100
    #         print(f"Generating order book for hour {hour_idx + 1} of {len(valid_hours)}... ({percentage_complete:.2f}% complete)")
    #         mid_price = day_ahead_prices[hour_idx]

    #         # Trading window for this product
    #         trading_start = (delivery_time - timedelta(days=1)).replace(hour=14 + timezone_offset_hours, minute=0, second=0, microsecond=0)
    #         if trading_start > delivery_time:
    #             trading_start -= timedelta(days=1)
    #         trading_end = delivery_time  # Trading closes at delivery start

    #         # Fraction of trading window elapsed (0 = just opened, 1 = at delivery)
    #         now = trading_end
    #         trading_window_seconds = (trading_end - trading_start).total_seconds()
    #         time_to_delivery_seconds = (trading_end - now).total_seconds()
    #         elapsed_fraction = 1 - (time_to_delivery_seconds / trading_window_seconds)
            
    #         # Number of orders per side: min 20, up to 60 as delivery approaches
    #         expected_levels = min_order_book_levels + int(max_order_book_levels * elapsed_fraction)

    #         # Use a smoother function for the number of orders per side
    #         # For example, a sigmoid function for smooth growth
    #         def smooth_order_growth(elapsed_fraction, min_levels, max_levels):
    #             growth_rate = 10  # Adjust this to control the steepness of the curve
    #             return min_levels + (max_levels - min_levels) / (1 + np.exp(-growth_rate * (elapsed_fraction - 0.5)))

    #         n_orders_per_side = int(smooth_order_growth(elapsed_fraction, min_order_book_levels, min_order_book_levels + max_order_book_levels))

    #         # Volatility increases near delivery (e.g., price offset stddev grows)
    #         base_volatility = self.base_volatility  # EUR/MWh, base price offset
    #         max_volatility = self.max_volatility   # EUR/MWh, max price offset near delivery
    #         volatility = base_volatility + (max_volatility - base_volatility) * elapsed_fraction

    #         bids = []
    #         asks = []
    #         for level in range(1, n_orders_per_side + 1):
    #             # For the best levels, occasionally allow overlap or zero spread
    #             if level == 1 and random.random() < 0.4:  # 40% chance of overlap at best level
    #                 overlap = random.choice([0, -tick_size, -random.uniform(0.01, 0.05)])
    #                 bid_price = round(mid_price + overlap, 2)
    #                 ask_price = round(mid_price + overlap, 2)
    #                 if bid_price < ask_price:
    #                     bid_price, ask_price = ask_price, bid_price
    #             elif level == 1 and random.random() < 0.2:  # 20% chance of zero spread
    #                 bid_price = ask_price = round(mid_price, 2)
    #             else:
    #                 # Price offset increases with volatility and level
    #                 price_offset = (level * tick_size) + abs(random.gauss(0, volatility))
    #                 bid_price = round(mid_price - price_offset, 2)
    #                 ask_price = round(mid_price + price_offset, 2)

    #             # Simulate volume with randomness, higher near best prices
    #             base_volume = max(1000.0 - 50.0 * level + random.uniform(0.0, 1000.0), 100.0)
    #             # More volume as delivery approaches
    #             volume_boost = 1 + 0.5 * elapsed_fraction
    #             bid_volume = round(max(base_volume * volume_boost + (random.random()*100), volume_increment), 1)
    #             ask_volume = round(max(base_volume * volume_boost + (random.random()*100), volume_increment), 1)

    #             # More orders closer to delivery
    #             time_fraction = random.betavariate(2 + 8 * elapsed_fraction, 5 - 3 * elapsed_fraction)
    #             order_time = trading_start + (trading_end - trading_start) * time_fraction

    #             bids.append({
    #                 "price": bid_price,
    #                 "volume": bid_volume,
    #                 "timestamp": order_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    #                 "side": "buy"
    #             })
    #             asks.append({
    #                 "price": ask_price,
    #                 "volume": ask_volume,
    #                 "timestamp": order_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    #                 "side": "sell"
    #             })
    #         # Sort bids and asks
    #         bids_sorted = sorted(bids, key=lambda x: -x["price"])
    #         asks_sorted = sorted(asks, key=lambda x: x["price"])

    #         order_book = {
    #             "delivery_hour_utc": delivery_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    #             "mid_price": mid_price,
    #             "orderbook": {
    #                 "bids": bids_sorted,
    #                 "asks": asks_sorted
    #             }
    #         }
    #         results.append(order_book)

    #     return results
    def generate_intraday_order_book(
            self,
            valid_hours,  # List of valid timestamps (datetime objects)
            day_ahead_prices,
            timezone_offset_hours=1  # CET = UTC+1, adjust for summer time if needed
        ) -> list:
        """
        Generate synthetic order book data for the German intraday market (EPEX SPOT realistic format).

        Args:
            valid_hours (list of datetime): List of valid timestamps to process.
            day_ahead_prices (list of floats): Realized hourly Day-Ahead prices.
            timezone_offset_hours (int): Offset from UTC to CET (default 1).

        Returns:
            list: List of dictionaries representing order book snapshots.
        """
        results = []
        tick_size = self.tick_size
        volume_increment = self.volume_increment
        min_order_book_levels = self.min_order_book_levels
        max_order_book_levels = self.max_order_book_levels

        if len(day_ahead_prices) != len(valid_hours):
            raise ValueError(
                f"Mismatch between number of valid hours ({len(valid_hours)}) and length of day_ahead_prices ({len(day_ahead_prices)})."
            )

        for hour_idx, delivery_time in enumerate(valid_hours):
            percentage_complete = ((hour_idx + 1) / len(valid_hours)) * 100
            print(f"Generating order book for hour {hour_idx + 1} of {len(valid_hours)}... ({percentage_complete:.2f}% complete)")
            mid_price = day_ahead_prices[hour_idx]

            # Trading window for this product
            trading_start = (delivery_time - timedelta(days=1)).replace(hour=14 + timezone_offset_hours, minute=0, second=0, microsecond=0)
            if trading_start > delivery_time:
                trading_start -= timedelta(days=1)
            trading_end = delivery_time  # Trading closes at delivery start

            # Fraction of trading window elapsed (0 = just opened, 1 = at delivery)
            now = trading_end
            trading_window_seconds = (trading_end - trading_start).total_seconds()
            time_to_delivery_seconds = (trading_end - now).total_seconds()
            elapsed_fraction = 1 - (time_to_delivery_seconds / trading_window_seconds)
            
            # Use a smoother function for the number of orders per side
            def smooth_order_growth(elapsed_fraction, min_levels, max_levels):
                growth_rate = 10  # Adjust this to control the steepness of the curve
                return min_levels + (max_levels - min_levels) / (1 + np.exp(-growth_rate * (elapsed_fraction - 0.5)))

            n_orders_per_side = int(smooth_order_growth(elapsed_fraction, min_order_book_levels, min_order_book_levels + max_order_book_levels))

            # Volatility increases near delivery (e.g., price offset stddev grows)
            base_volatility = self.base_volatility  # EUR/MWh, base price offset
            max_volatility = self.max_volatility   # EUR/MWh, max price offset near delivery
            volatility = base_volatility + (max_volatility - base_volatility) * elapsed_fraction

            bids = []
            asks = []
            for level in range(1, n_orders_per_side + 1):
                # For the best levels, occasionally allow overlap or zero spread
                if level == 1 and random.random() < 0.4:  # 40% chance of overlap at best level
                    overlap = random.choice([0, -tick_size, -random.uniform(0.01, 0.05)])
                    bid_price = round(mid_price + overlap, 2)
                    ask_price = round(mid_price + overlap, 2)
                    if bid_price < ask_price:
                        bid_price, ask_price = ask_price, bid_price
                elif level == 1 and random.random() < 0.2:  # 20% chance of zero spread
                    bid_price = ask_price = round(mid_price, 2)
                else:
                    # Price offset increases with volatility and level
                    price_offset = (level * tick_size) + abs(random.gauss(0, volatility))
                    bid_price = round(mid_price - price_offset, 2)
                    ask_price = round(mid_price + price_offset, 2)

                # Simulate volume with randomness, higher near best prices
                base_volume = max(1000.0 - 50.0 * level + random.uniform(0.0, 1000.0), 100.0)
                # More volume as delivery approaches
                volume_boost = 1 + 0.5 * elapsed_fraction
                bid_volume = round(max(base_volume * volume_boost + (random.random()*100), volume_increment), 1)
                ask_volume = round(max(base_volume * volume_boost + (random.random()*100), volume_increment), 1)

                # More orders closer to delivery
                time_fraction = random.betavariate(2 + 8 * elapsed_fraction, 5 - 3 * elapsed_fraction)
                order_time = trading_start + (trading_end - trading_start) * time_fraction
                timestamp_str = order_time.strftime("%Y-%m-%dT%H:%M:%SZ")

                # Generate unique order IDs
                bid_order_id = str(uuid.uuid4())
                ask_order_id = str(uuid.uuid4())

                # Common attributes
                delivery_start = delivery_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                delivery_end = (delivery_time + timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
                product = "DE-AT-LU_15min"
                delivery_area = "DE-AT-LU"
                market_area = "DE-AT-LU"
                currency = "EUR"
                isOTC = random.choice(["N", "Y"]) if random.random() < 0.05 else "N"
                execution_restriction = random.choices(
                    ["NON", "AON", "IOC", "FOK"], weights=[0.85, 0.05, 0.05, 0.05]
                )[0]

                # Bid
                bids.append({
                    "order_id": bid_order_id,
                    "initial_id": bid_order_id,
                    "parent_id": "",
                    "entry_time": timestamp_str,
                    "action_code": "A",
                    "transaction_time": timestamp_str,
                    "validity_time": "",
                    "delivery_start": delivery_start,
                    "delivery_end": delivery_end,
                    "product": product,
                    "delivery_area": delivery_area,
                    "market_area": market_area,
                    "side": "buy",
                    "price": bid_price,
                    "currency": currency,
                    "volume": bid_volume,
                    "isOTC": isOTC,
                    "revision_no": 1,
                    "is_user_defined_block": "N",
                    "execution_restriction": execution_restriction,
                    "timestamp": timestamp_str  # for compatibility with your original structure
                })

                # Ask
                asks.append({
                    "order_id": ask_order_id,
                    "initial_id": ask_order_id,
                    "parent_id": "",
                    "entry_time": timestamp_str,
                    "action_code": "A",
                    "transaction_time": timestamp_str,
                    "validity_time": "",
                    "delivery_start": delivery_start,
                    "delivery_end": delivery_end,
                    "product": product,
                    "delivery_area": delivery_area,
                    "market_area": market_area,
                    "side": "sell",
                    "price": ask_price,
                    "currency": currency,
                    "volume": ask_volume,
                    "isOTC": isOTC,
                    "revision_no": 1,
                    "is_user_defined_block": "N",
                    "execution_restriction": execution_restriction,
                    "timestamp": timestamp_str  # for compatibility with your original structure
                })

            # Sort bids and asks
            bids_sorted = sorted(bids, key=lambda x: -x["price"])
            asks_sorted = sorted(asks, key=lambda x: x["price"])

            order_book = {
                "delivery_hour_utc": delivery_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mid_price": mid_price,
                "orderbook": {
                    "bids": bids_sorted,
                    "asks": asks_sorted
                }
            }
            results.append(order_book)

        return results

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
country = "DE"
save_order_book_path = rf"/Users/Robert_Hennings/Uni/Master/MasterThesis/data/raw"
save_order_book_file_name = f"Continous_Intraday_OrderBook_60Minutes_Product_{country}_{now}.json"

order_book_generator_instance = OrderBookGenerator(
    day_ahead_prices=day_ahead_prices[:10],
    valid_hours=valid_hours[:10],
    save_order_book=True,
    save_order_book_path=save_order_book_path,
    save_order_book_file_name=save_order_book_file_name
    )
order_book = order_book_generator_instance.generate_intraday_order_book(
    day_ahead_prices=day_ahead_prices[:10],
    valid_hours=valid_hours[:10])

def generate_trade_log_from_orderbook(order_book_snapshots):
    """
    Generate a trade log by matching bids and asks in each order book snapshot.

    Args:
        order_book_snapshots (list): Output from generate_intraday_order_book (one per delivery period).

    Returns:
        list of dict: Trade log entries.
    """
    trade_log = []
    trade_id_counter = 1

    for snapshot in order_book_snapshots:
        bids = [dict(bid) for bid in snapshot["orderbook"]["bids"]]  # Copy to allow volume modification
        asks = [dict(ask) for ask in snapshot["orderbook"]["asks"]]
        delivery_hour_utc = snapshot["delivery_hour_utc"]

        # Sort bids descending, asks ascending
        bids_sorted = sorted(bids, key=lambda x: (-x["price"], x["timestamp"]))
        asks_sorted = sorted(asks, key=lambda x: (x["price"], x["timestamp"]))

        bid_idx = 0
        ask_idx = 0

        while bid_idx < len(bids_sorted) and ask_idx < len(asks_sorted):
            bid = bids_sorted[bid_idx]
            ask = asks_sorted[ask_idx]

            # Check for match
            if bid["price"] >= ask["price"]:
                trade_price = ask["price"]  # Market convention: price of passive order (ask)
                trade_quantity = min(bid["volume"], ask["volume"])
                trade_time = max(bid["timestamp"], ask["timestamp"])

                trade_entry = {
                    "trade_id": f"TRADE_{trade_id_counter}",
                    "execution_time": trade_time,
                    "delivery_hour_utc": delivery_hour_utc,
                    "price": trade_price,
                    "quantity": trade_quantity,
                    "buy_order_id": bid["order_id"],
                    "sell_order_id": ask["order_id"],
                    "buy_entry_time": bid["entry_time"],
                    "sell_entry_time": ask["entry_time"],
                    "product": bid["product"],
                    "delivery_area": bid["delivery_area"],
                    "market_area": bid["market_area"],
                    "currency": bid["currency"],
                    "isOTC": bid["isOTC"] if bid["side"] == "buy" else ask["isOTC"],
                    "execution_restriction": bid["execution_restriction"] if bid["side"] == "buy" else ask["execution_restriction"],
                }
                trade_log.append(trade_entry)
                trade_id_counter += 1

                # Subtract traded quantity
                bid["volume"] -= trade_quantity
                ask["volume"] -= trade_quantity

                # Remove fully filled orders
                if bid["volume"] <= 0:
                    bid_idx += 1
                if ask["volume"] <= 0:
                    ask_idx += 1
            else:
                # No more matches possible
                break

    return trade_log


trade_log = generate_trade_log_from_orderbook(order_book)
pd.DataFrame(trade_log).buy_order_id
pd.DataFrame(order_book[2].get("orderbook").get("bids")).query("order_id == '2a0cbd0f-9e66-41b8-8240-f49ae9f9f11e'")

# 0    2a0cbd0f-9e66-41b8-8240-f49ae9f9f11e
# 1    17a5f5c9-3551-4723-a986-ed560a20a4ce
# 2    56b3340c-623b-4ccc-8bd3-9ee19c983075
# 3    9a63f361-6778-4ec8-9300-c6f523fe2c24
# 4    ef1e2cc1-851b-4d4a-abd7-684640870ebb


# order_book_generator_instance.__save_order_book(order_book=order_book)
            