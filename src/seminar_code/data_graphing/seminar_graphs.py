from typing import Dict, List, Tuple
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import json

CAU_COLOR_SCALE = ["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"]
FIGURES_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"
TABLES_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables"
DATA_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/data"
NUM_YEARS_INTERVAL_X_AXIS = 5
SRC_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/src"
SEMINAR_CODE_PATH = rf"{SRC_PATH}/seminar_code"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# import config settings with static global variables
print(os.getcwd())
os.chdir(SEMINAR_CODE_PATH)
print(os.getcwd())

from utils.evaluation import adf_test, \
    granger_causality_test, \
    cointegration_test, \
    test_data_for_normality

from data_loading.data_loader import DataLoading
from data_graphing.data_grapher import DataGraphing

# Instantiate the data loading class
data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)
# Instantiate the data graphing class
data_graphing_instance = DataGraphing()
# Convention for Figure Titles: [Data Frequency] + [Variable Description] + [Country/Region] + [Time Period as only years]
# Convention for Figure x_axis: Time
#----------------------------------------------------------------------------------------
# 00 - Intro - Intro: The PPP puzzle and Commodity Currencies
#----------------------------------------------------------------------------------------
# Also sourcing the real effective exchange rates on a daily basis from BIS
country_keys_mapping = {
    "US": "United States",
    "GB": "United Kingdom",
    "JP": "Japan",
    "CH": "Switzerland",
    "XM": "Euro Area (EA)"
}
exchange_rates_df = data_loading_instance.get_bis_exchange_rate_data(
    country_keys_mapping=country_keys_mapping,
    exchange_rate_type_list=[
        "Real effective exchange rate - monthly - narrow basket",
        ]).rename(columns={
            "US_Real effective exchange rate - monthly - narrow basket": "United States",
            "GB_Real effective exchange rate - monthly - narrow basket": "United Kingdom",
            "JP_Real effective exchange rate - monthly - narrow basket": "Japan",
            "CH_Real effective exchange rate - monthly - narrow basket": "Switzerland",
            "XM_Real effective exchange rate - monthly - narrow basket": "Euro Area (EA)"
        })
effective_exchange_rates_df = pd.DataFrame()
effective_exchange_rates_df["GBPREAL"] = np.log(exchange_rates_df["United Kingdom"]) - np.log(exchange_rates_df["United States"])
effective_exchange_rates_df["JPYREAL"] = np.log(exchange_rates_df["Japan"]) - np.log(exchange_rates_df["United States"])
effective_exchange_rates_df["CHFREAL"] = np.log(exchange_rates_df["Switzerland"]) - np.log(exchange_rates_df["United States"])
effective_exchange_rates_df["EUROREAL"] = np.log(exchange_rates_df["Euro Area (EA)"]) - np.log(exchange_rates_df["United States"])


data = effective_exchange_rates_df.copy()
variables = data.columns.tolist()
secondary_yaxis_variables = []
title = f"Monthly deviations of USD Spot Rate from PPP-values over the time: {data.index[0].year} - {data.index[-1].year}"
x_axis_title = "Time"
y_axis_title = "Deviations of USD Spot Rate from PPP-values (log)"
secondary_yaxis_title = ""
color_mapping = {
    'GBPREAL': "grey",
    'JPYREAL': "black",
    'CHFREAL': "#9b0a7d",
    'EUROREAL': "darkgrey"
}
fig_deviations_from_ppp = data_graphing_instance.get_fig_deviations_ppp(
        data=data,
        variables=variables,
        secondary_y_variables=secondary_yaxis_variables,
        title="",
        secondary_y_axis_title=secondary_yaxis_title,
        color_discrete_sequence=CAU_COLOR_SCALE,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="chap_00_deviations_of_usd_spotrates_from_ppp_values",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_deviations_from_ppp.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil: Global Production and Consumption over time
# ----------------------------------------------------------------------------------------
series_dict_mapping = {
    "Oil Consumption": "https://ourworldindata.org/grapher/oil-consumption-by-country",
    "Oil Production": "https://ourworldindata.org/grapher/oil-production-by-country",
    "Gas Consumption": "https://ourworldindata.org/grapher/gas-consumption-by-country",
    "Gas Production": "https://ourworldindata.org/grapher/gas-production-by-country",
}
data_dict, data_full_info_dict = data_loading_instance.get_oil_gas_prod_con_data(
    series_dict_mapping=series_dict_mapping
    )
data_full_info_df = pd.concat([pd.DataFrame(index=range(len(data_full_info_dict.get(key))), data=data_full_info_dict.get(key)) for key in data_full_info_dict.keys()]).reset_index(drop=True)
# Save the data accordingly
oil_consumption_df = data_dict.get("Oil Consumption")
oil_production_df = data_dict.get("Oil Production").rename(mapper={"oil_production__twh": "oil_production_twh"}, axis=1)
gas_consumption_df = data_dict.get("Gas Consumption")
gas_production_df = data_dict.get("Gas Production").rename(mapper={"gas_production__twh": "gas_production_twh"}, axis=1)
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil Consumption
# ----------------------------------------------------------------------------------------
variables = ["United States", "China", "India", "Japan", "Russia", "Saudi Arabia", "South Korea", "Canada", "Brazil", "European Union (27)"]
secondary_y_variables = ["World"]
# Figure out the common start date for a shared x-axis
start_year_consumption = oil_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year_consumption = oil_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()
start_year_production = oil_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year_production = oil_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()

oil_start_year = max(start_year_consumption, start_year_production)

start_year = oil_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year = oil_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()
title = f"Yearly oil consumption by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Oil consumption (in terawatt-hours)"
secondary_y_axis_title = "World oil consumption (in terawatt-hours)"


custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
world_color = "red"  # Fixed color for the "World" category
# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["World"] = world_color  # Assign the fixed color for "World"

oil_consumption_df["Year"] = pd.to_datetime(oil_consumption_df["Year"], format="%Y")
oil_consumption_df = oil_consumption_df.pivot(index="Year", columns="Entity", values="oil_consumption_twh")
oil_consumption_df = oil_consumption_df[variables + secondary_y_variables]

# See if the overall pricture get better when we restrict the data to only the relative share of the US
oil_consumption_df_usa = oil_consumption_df[["United States", "World"]]
# Calculate the relative share of the US in world oil consumption
oil_consumption_df_usa["United States Share of World"] = oil_consumption_df_usa["United States"] / oil_consumption_df_usa["World"]

fig_oil_consumption = data_graphing_instance.get_fig_consumption_production_oil_gas(
        data=oil_consumption_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_yearly_oil_consumption_by_country",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_oil_consumption.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil Production
# ----------------------------------------------------------------------------------------
variables = ["United States", "Russia", "Saudi Arabia", "Canada", "Iran", "China", "Brazil", "Norway", "European Union (27)"]
secondary_y_variables = ["World"]

oil_production_df = oil_production_df[oil_production_df.Year >= oil_start_year]
start_year = oil_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year = oil_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()

title = f"Yearly oil production by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Oil production (in terawatt-hours)"
secondary_y_axis_title = "World oil production (in terawatt-hours)"

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
world_color = "red"  # Fixed color for the "World" category

# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["World"] = world_color  # Assign the fixed color for "World"

oil_production_df["Year"] = pd.to_datetime(oil_production_df["Year"], format="%Y")
oil_production_df = oil_production_df.pivot(index="Year", columns="Entity", values="oil_production_twh")
oil_production_df = oil_production_df[variables + secondary_y_variables]


fig_oil_production = data_graphing_instance.get_fig_consumption_production_oil_gas(
        data=oil_production_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_yearly_oil_production_by_country",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_oil_production.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil Production and Consumption
# ----------------------------------------------------------------------------------------
subplot_titles=(
    f"Yearly oil consumption by country (in terawatt-hours) over the time: {start_year} - {end_year}",
    f"Yearly oil production by country (in terawatt-hours) over the time: {start_year} - {end_year}")
title = f"Yearly oil consumption and production by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
secondary_y_variable = "World"

fig_oil_consumption_production_combine = data_graphing_instance.get_combined_production_consumption_graph(
        subplot_titles=list(subplot_titles),
        title="",
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        x_axis_title=x_axis_title,
        secondary_y_variable=secondary_y_variable,
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        fig_production=fig_oil_production,
        fig_consumption=fig_oil_consumption,
        save_fig=False,
        file_name="chap_01_yearly_oil_consumption_production_combined_graph",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_oil_consumption_production_combine.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Gas Consumption
# ----------------------------------------------------------------------------------------
variables = ["United States", "China", "Russia", "Iran", "Canada", "Australia", "Saudi Arabia", "European Union (27)"]
secondary_y_variables = ["World"]
# Figure out the common start date for a shared x-axis
start_year_consumption = gas_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year_consumption = gas_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()
start_year_production = gas_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year_production = gas_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()
gas_start_year = max(start_year_consumption, start_year_production)

start_year = gas_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.min()
end_year = gas_consumption_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()

title = f"Yearly gas consumption by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Gas consumption (in terawatt-hours)"
secondary_y_axis_title = "World gas consumption (in terawatt-hours)"

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
world_color = "red"  # Fixed color for the "World" category

# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["World"] = world_color  # Assign the fixed color for "World"

gas_consumption_df["Year"] = pd.to_datetime(gas_consumption_df["Year"], format="%Y")
gas_consumption_df = gas_consumption_df.pivot(index="Year", columns="Entity", values="gas_consumption_twh")
gas_consumption_df = gas_consumption_df[variables + secondary_y_variables]

fig_gas_consumption = data_graphing_instance.get_fig_consumption_production_oil_gas(
        data=gas_consumption_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_yearly_gas_consumption_by_country",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_consumption.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Gas Production
# ----------------------------------------------------------------------------------------
variables = ["United States", "China", "Iran", "Canada", "Saudi Arabia", "Mexico", "European Union (27)"]
secondary_y_variables = ["World"]

start_year = gas_start_year
end_year = gas_production_df.query("Entity.isin(@variables) or Entity.isin(@secondary_y_variables)").Year.max()

gas_production_df = gas_production_df[gas_production_df.Year >= gas_start_year]

title = f"Yearly gas production by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Production (in terawatt-hours)"
secondary_y_axis_title = "World gas production (in terawatt-hours)"


custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
world_color = "red"  # Fixed color for the "World" category

# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["World"] = world_color  # Assign the fixed color for "World"

gas_production_df["Year"] = pd.to_datetime(gas_production_df["Year"], format="%Y")
gas_production_df = gas_production_df.pivot(index="Year", columns="Entity", values="gas_production_twh")
gas_production_df = gas_production_df[variables + secondary_y_variables]

fig_gas_production = data_graphing_instance.get_fig_consumption_production_oil_gas(
        data=gas_production_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_yearly_gas_production_by_country",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_production.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Gas Consumption and Production
# ----------------------------------------------------------------------------------------
subplot_titles=(f"Yearly gas consumption by country (in terawatt-hours) over the time: {start_year} - {end_year}",
                f"Yearly gas production by country (in terawatt-hours) over the time: {start_year} - {end_year}")
title = f"Yearly gas consumption and production by country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Time"
secondary_y_variable = "World"

fig_gas_consumption_production_combine = data_graphing_instance.get_combined_production_consumption_graph(
        subplot_titles=subplot_titles,
        title="",
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        x_axis_title=x_axis_title,
        secondary_y_variable=secondary_y_variable,
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        fig_production=fig_gas_production,
        fig_consumption=fig_gas_consumption,
        save_fig=False,
        file_name="chap_01_yearly_gas_consumption_production_combined_graph",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_consumption_production_combine.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Financial Markets: Oil and Gas OI over time
# ----------------------------------------------------------------------------------------
start_date = "1995-01-01"
end_date = "2025-01-01"
report_type = "Futures-and-Options Combined Reports"
save_files_locally = False
save_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"

cftc_data = data_loading_instance.get_cftc_commitment_of_traders(
    start_date=start_date,
    end_date=end_date,
    report_type=report_type,
    save_files_locally=save_files_locally,
    save_path=save_path
)
cftc_data["As of Date in Form YYYY-MM-DD"] = pd.to_datetime(cftc_data["As of Date in Form YYYY-MM-DD"])

# Loading from Files
file_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"
file_name = r"cftc_Futures-and-Options Combined Reports_1995-01-01_to_2025-01-01.csv"

full_path = f"{file_path}/{file_name}"
cftc_data = pd.read_csv(full_path)
print(f"Dimension: {cftc_data.shape}")
cftc_data["Market and Exchange Names"] = cftc_data["Market and Exchange Names"].str.strip()

oil_gas_products_list = [
                         "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
                         "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
                         "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
                         "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"
                         ]

cftc_data_oil_gas = cftc_data[
    cftc_data["Market and Exchange Names"].isin(oil_gas_products_list)
].sort_values(by=["As of Date in Form YYYY-MM-DD"]).reset_index(drop=True)


cftc_data_oil_gas.columns = cftc_data_oil_gas.columns.str.strip()

columns_keep = [
    "As of Date in Form YYYY-MM-DD",
    "Market and Exchange Names",
    "Open Interest (All)",
    "Noncommercial Positions-Long (All)",
    "Noncommercial Positions-Short (All)",
    "Commercial Positions-Long (All)",
    "Commercial Positions-Short (All)",
    "Total Reportable Positions-Long (All)",
    "Total Reportable Positions-Short (All)",
    "Noncommercial Positions-Spreading (All)",
    ]
cftc_data_oil_gas = cftc_data_oil_gas[columns_keep]


cftc_data_oil_gas["Market and Exchange Names"] = cftc_data_oil_gas["Market and Exchange Names"].str.replace(" - NEW YORK MERCANTILE EXCHANGE", "")
# Gas
cftc_data_oil_gas["Market and Exchange Names"] = cftc_data_oil_gas["Market and Exchange Names"].str.replace("NATURAL GAS", "Gas")
cftc_data_oil_gas["Market and Exchange Names"] = cftc_data_oil_gas["Market and Exchange Names"].str.replace("NAT GAS NYME", "Gas")
# Oil
cftc_data_oil_gas["Market and Exchange Names"] = cftc_data_oil_gas["Market and Exchange Names"].str.replace("CRUDE OIL, LIGHT SWEET", "Oil")
cftc_data_oil_gas["Market and Exchange Names"] = cftc_data_oil_gas["Market and Exchange Names"].str.replace("WTI-PHYSICAL", "Oil")
# Date
cftc_data_oil_gas = cftc_data_oil_gas.rename(columns={"Market and Exchange Names": "Product"})
cftc_data_oil_gas["Date"] = pd.to_datetime(cftc_data_oil_gas["As of Date in Form YYYY-MM-DD"], format="%Y-%m-%d")
cftc_data_oil_gas = cftc_data_oil_gas.drop(columns=["As of Date in Form YYYY-MM-DD"])


# cftc_data_oil_gas
# Pivot the data to create separate columns for each unique Product
cftc_data_oil_gas_pivoted = cftc_data_oil_gas.pivot_table(
    index="Date",  # Use Date as the index
    columns="Product",  # Create columns for each unique Product
    values="Open Interest (All)",  # Populate with Open Interest (All) values
    aggfunc="sum"  # Use sum in case of duplicate entries
).reset_index()

# Rename the columns for clarity
cftc_data_oil_gas_pivoted.columns.name = None  # Remove the columns' name
cftc_data_oil_gas_pivoted = cftc_data_oil_gas_pivoted.rename_axis(None, axis=1)  # Remove the index name

# Display the resulting DataFrame
cftc_data_oil_gas_pivoted = cftc_data_oil_gas_pivoted.dropna().reset_index(drop=True).set_index("Date", drop=True)


# Load the USD Prices of the nearest Future and merge with the CFTC data for WTI Oil and Natural Gas
wti_contract_size = 1000  # barrels
ng_contract_size = 10000  # mmBtu
# Can be checked via:
# cftc_data_oil_gas["Contract Units"].unique()
wti_prices = yf.download(
    tickers="CL=F",
    start="1995-01-01",
    end="2025-01-01",
    interval="1d",
    progress=True,
    auto_adjust=False)
ng_prices = yf.download(
    tickers="NG=F",
    start="1995-01-01",
    end="2025-01-01",
    interval="1d",
    progress=True,
    auto_adjust=False)
# Merge the price data with the CFTC data
mask_wti = wti_prices.index.isin(cftc_data_oil_gas_pivoted.index)
mask_ng = ng_prices.index.isin(cftc_data_oil_gas_pivoted.index)

wti_prices_filtered = wti_prices[mask_wti]["Adj Close"]
ng_prices_filtered = ng_prices[mask_ng]["Adj Close"]

prices_df = wti_prices_filtered.merge(
   right=ng_prices_filtered,
    left_index=True,
    right_index=True,
    how="outer")


prices_oi_df = pd.merge(
    left=cftc_data_oil_gas_pivoted,
    right=prices_df,
    left_index=True,
    right_index=True
).dropna()
prices_oi_df["Gas Open Interest USD"] = prices_oi_df["Gas"] * prices_oi_df["NG=F"] * ng_contract_size
prices_oi_df["Oil Open Interest USD"] = prices_oi_df["Oil"] * prices_oi_df["CL=F"] * wti_contract_size
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil Open Interest
# ----------------------------------------------------------------------------------------
variables = ["Oil"]
secondary_y_variables = ["Oil Open Interest USD"]

start_year = prices_oi_df.index.min().year
end_year = prices_oi_df.index.max().year

title = f"Weekly oil open interest as total number of contracts and in USD over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Open interest (number of contracts)"
secondary_y_axis_title = "Open interest (in USD)"

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
oi_usd_color = "black"  # Fixed color for the "World" category

# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["Oil Open Interest USD"] = oi_usd_color

fig_oil_oi = data_graphing_instance.get_fig_open_interest(
        data=prices_oi_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_weekly_oil_open_interest",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_oil_oi.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Gas Open Interest
# ----------------------------------------------------------------------------------------
variables = ["Gas"]
secondary_y_variables = ["Gas Open Interest USD"]


title = f"Weekly gas open interest as total number of contracts and in USD over the time: {start_year} - {end_year}"
x_axis_title = "Time"
y_axis_title = "Open interest (number of contracts)"
secondary_y_axis_title = "Open interest (in USD)"

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(variables)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
oi_usd_color = "black"  # Fixed color for the "World" category

# Create a mapping for the colors
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(variables)}
color_mapping["Gas Open Interest USD"] = oi_usd_color

fig_gas_oi = data_graphing_instance.get_fig_open_interest(
        data=prices_oi_df,
        variables=variables,
        secondary_y_variables=secondary_y_variables,
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        secondary_y_axis_title=secondary_y_axis_title,
        color_mapping_dict=color_mapping,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        save_fig=False,
        file_name="chap_01_weekly_gas_open_interest",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_oi.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - Oil and Gas Open Interest
# ----------------------------------------------------------------------------------------
subplot_titles=(f"Weekly oil open interest as total number of contracts and in USD over the time: {prices_oi_df.index.min().strftime('%Y')} - {prices_oi_df.index.max().strftime('%Y')}",
                f"Weekly gas open interest as total number of contracts and in USD over the time: {prices_oi_df.index.min().strftime('%Y')} - {prices_oi_df.index.max().strftime('%Y')}")
start_year = fig_gas_oi.data[0].x[0]
end_year = fig_gas_oi.data[0].x[-1]
title = f"Weekly open interest of oil and gas products over the time: {pd.Timestamp(start_year).strftime('%Y')} - {pd.Timestamp(end_year).strftime('%Y')}"
x_axis_title = "Time"

fig_gas_oil_open_interest_combine = data_graphing_instance.get_combined_open_interest_graph(
        subplot_titles=subplot_titles,
        title="",
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        x_axis_title=x_axis_title,
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        fig_oil_oi=fig_oil_oi,
        fig_gas_oi=fig_gas_oi,
        save_fig=False,
        file_name="chap_01_weekly_open_interest_oil_gas_combined_graph.pdf",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_oil_open_interest_combine.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Inflation Dynamics and the Role of Energy Prices
#----------------------------------------------------------------------------------------
# Chapter 2)
series_dict_mapping = {
    'Headline': 'CPIAUCSL',
    'Core CPI': 'CPILFESL',
    'Energy': 'CPIENGSL',
    'Food': 'CPIUFDSL',
}
weights = {
    'Core CPI': 1 - 0.136 - 0.072,  # Core (everything except food & energy)
    'Energy': 0.072,
    'Food': 0.136,
}
start_date = '2000-01-01'
end_date = '2024-12-31'
headline_id = 'CPIAUCSL'  

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_us_df = pd.concat(list(data_dict.values()), axis=1).dropna()
data_us_full_info_table = pd.DataFrame(data_full_info_dict).T

def percent_change(series):
    return 100 * (series.iloc[-1] / series.iloc[-13] - 1)

# Compute 12-month percent change (annual inflation): (P_t / P_{t-12} - 1) * 100
data_pct = data_us_df.pct_change(periods=12) * 100

# Compute component contributions: weight × component inflation
contributions = pd.DataFrame(index=data_pct.index)
for col in weights:
    contributions[col] = data_pct[col] * weights[col]

# Only these bar components
bar_components = weights.keys()
contributions = contributions[bar_components].dropna()

# Sort contributors by value for each period (row), biggest on bottom
contributions_sorted = pd.DataFrame(
    np.sort(contributions[bar_components].values, axis=1)[:, ::-1],  # sort and reverse for descending
    index=contributions.index,
    columns=[f"{i+1}" for i in range(len(bar_components))]
)

# Next, get the names for the columns for legend and color mapping
def get_sorted_labels(row):
    # Sort values for the row, get corresponding labels
    vals = row.values
    labels = [x for _, x in sorted(zip(vals, bar_components), reverse=True)]
    return labels

labels_sorted = contributions[bar_components].apply(get_sorted_labels, axis=1)
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Energy Price Contribution to US Inflation
#----------------------------------------------------------------------------------------
variables = list(data_us_df.columns)
energy_color = "#9b0a7d"
cpi_color = "red"  # Fixed color for the "World" category

color_mapping = {
    'Core CPI': "grey",
    'Energy': energy_color,
    'Food': "lightgrey",
    'Headline': cpi_color,
}
title = f"US CPI: Headline and component contributions over the time: {contributions.index[0].year} - {contributions.index[-1].year}"
x_axis_title = "Time"
y_axis_title = 'Contribution to annual inflation (%)'

fig_inflation_decomp_usa = data_graphing_instance.get_fig_inflation_contribution_usa(
        data=contributions,
        data_pct=data_pct,
        cpi_color=cpi_color,
        variables=list(bar_components),
        title="",
        secondary_y_variables=[],
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="chap_02_us_cpi_inflation_decomposition",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
fig_inflation_decomp_usa.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Energy Price Contribution to Eu Area Inflation
#----------------------------------------------------------------------------------------
series_dict_mapping = {
    "Core CPI": "ICP.M.U2.N.XEF000.4.ANR",          # HICP excluding Energy and Food
    "Energy": "ICP.M.U2.N.NRGY00.4.ANR",         # HICP Energy euro area
    "Food": "ICP.M.U2.N.FOOD00.4.ANR",           # HICP Food (incl. alcohol & tobacco)
    "Headline": "ICP.M.U2.N.000000.4.ANR",      # HICP Overall Index euro area
}
start_date = "2000-01"
end_date = "2024-12"

data_dict = data_loading_instance.get_ecb_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )

data = pd.concat([df["OBS_VALUE"] for df in data_dict.values()], axis=1)
data.columns = data_dict.keys()
data.index = pd.to_datetime(data_dict[list(data_dict.keys())[0]]["TIME_PERIOD"])

# Use approximate euro area HICP weights (for 2022, update for accuracy):
# Euro area weights (approx. for 2022, update if needed)
weights = {
    "Food": 0.172,      # 17.2%
    "Energy": 0.098,    # 9.8%
    "Core CPI": 1 - 0.172 - 0.098,  # remainder
}
data = data.resample('M').last()  # If needed, ensure monthly end-of-period alignment

# Here, each value is ALREADY YoY % change. So simply multiply by weights for contributions:
contributions = pd.DataFrame(index=data.index)
for col in weights:
    contributions[col] = data[col] * weights[col]

bar_components = data.columns.difference(['Headline'])
contributions = contributions[bar_components].dropna()
# Sort contributors by value for each period (row), biggest on bottom
contributions_sorted = pd.DataFrame(
    np.sort(contributions[bar_components].values, axis=1)[:, ::-1],  # sort and reverse for descending
    index=contributions.index,
    columns=[f"{i+1}" for i in range(len(bar_components))]
)
labels_sorted = contributions[bar_components].apply(get_sorted_labels, axis=1)

variables = list(contributions.columns)
contributions["Headline"] = data["Headline"]

energy_color = "#9b0a7d"
cpi_color = "red"  # Fixed color for the "World" category

color_mapping = {
    'Core CPI': "grey",
    'Energy': energy_color,
    'Food': "lightgrey",
    'Headline': cpi_color,
}
title = f"EU Area CPI: Headline and component contributions over the time: {contributions.index[0].year} - {contributions.index[-1].year}"
x_axis_title = "Time"
y_axis_title = 'Contribution to annual inflation (%)'

fig_inflation_decomp_euro_area = data_graphing_instance.get_fig_inflation_contribution_euro_area(
        data=contributions,
        cpi_color=cpi_color,
        variables=variables,
        secondary_y_variables=[],
        title="",
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="chap_02_eu_area_cpi_inflation_decomposition",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_inflation_decomp_euro_area.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index
#----------------------------------------------------------------------------------------
usd_index_ticker = "DX-Y.NYB"
us_oil_ticker = "CL=F"
us_ng_ticker = "NG=F"

us_data = yf.download(
    tickers=[usd_index_ticker, us_oil_ticker, us_ng_ticker],
    start="2000-01-01",
    end="2024-12-31",
    interval="1d",
    progress=True,
    auto_adjust=False
)
us_data = us_data["Adj Close"].dropna()
us_data.columns = ["USD Index", "WTI Oil", "Natural Gas"]


data = us_data.copy()
variables = ["USD Index", "WTI Oil"]
secondary_yaxis_variables = ["Natural Gas"]
color_discrete_sequence = ["grey", "black", "#9b0a7d"]
title = f"WTI Oil, Natural Gas and USD Index over the time: {data.index[0].year} - {data.index[-1].year}"
x_axis_title = "Date"
y_axis_title = "WTI Oil & USD Index"
secondary_yaxis_title = "Natural Gas"
color_mapping = {
    'USD Index': "grey",
    'WTI Oil': "black",
    'Natural Gas': "#9b0a7d",
}
fig_main_relationships_commodities_fx = data_graphing_instance.get_fig_relationship_main_vars(
        data=data,
        variables=variables,
        secondary_y_variables=secondary_yaxis_variables,
        title=title,
        secondary_y_axis_title=secondary_yaxis_title,
        color_discrete_sequence=color_discrete_sequence,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="oil_gas_usd_index",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_main_relationships_commodities_fx.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index - Rolling Volatility and Crisis Periods
#----------------------------------------------------------------------------------------
with open("/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods_dict.json", "r") as f:
    crisis_periods_dict = json.load(f)

series_dict_mapping = {
    'EUR/USD': 'DEXUSEU',
    'WTI Oil': 'DCOILWTICO',
    "Natural Gas": "DHHNGSP",
}

start_date = "1960-01-01"
end_date = "2025-10-01"

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
spot_exchange_rate_data_df = pd.concat(list(data_dict.values()), axis=1).dropna()


country_keys_mapping = {
    "US": "United States",
}
exchange_rates_df = data_loading_instance.get_bis_exchange_rate_data(
    country_keys_mapping=country_keys_mapping,
    exchange_rate_type_list=[
        "Nominal effective exchange rate - daily - narrow basket",
        ])
exchange_rates_df = exchange_rates_df.dropna()
# Compute the rolling standard deviation as volatility proxy
window = 30
exchange_rate_vola_df = exchange_rates_df.rolling(window=window).std().dropna()
# Also compute the log differences
exchange_rate_log_diff_df = np.log(exchange_rates_df).diff().dropna()
# Load the daily Oil WTI data from FRED because the data history is longer than Yahoo Finance
data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping={"OIL WTI": "DCOILWTICO",
                         "Henry Hub Natural Gas Spot Price": "DHHNGSP"},
    start_date=start_date,
    end_date=end_date
)
energy_commodity_prices_df = pd.concat(list(data_dict.values()), axis=1).dropna()
energy_commodity_prices_vola_df = energy_commodity_prices_df.rolling(window=window).std().dropna()
# Also compute the log differences
energy_commodity_prices_log_diff_df = np.log(energy_commodity_prices_df).diff().dropna()

# Merge the exchange rate and US energy commodtiy vola data
crisis_volatility_data = exchange_rate_vola_df.merge(
    right=energy_commodity_prices_vola_df,
    left_index=True,
    right_index=True,
    how="inner"
).dropna()

crisis_log_first_diff_data = exchange_rate_log_diff_df.merge(
    right=energy_commodity_prices_log_diff_df,
    left_index=True,
    right_index=True,
    how="inner"
).dropna()
data = crisis_volatility_data.copy()
data_log_first_diff = crisis_log_first_diff_data.copy()
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Normalized Exchange rate volatilty, wti oil and natural gas with highlighted crisis periods
#----------------------------------------------------------------------------------------
data_log_diff_normalized = (data_log_first_diff - data_log_first_diff.min()) / (data_log_first_diff.max() - data_log_first_diff.min())
title=f"Daily normalized EUR/USD spot exchange rate, oil and gas log first differences volatility with highlighted crisis periods over the time: {data.index[0].year} - {data.index[-1].year}"
x_axis_title="Time"
y_axis_title="Volatility of log first differences (normalized)"


fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data_log_diff_normalized,
    crisis_periods_dict=crisis_periods_dict,
    variables=data_log_diff_normalized.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title="",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility (Normalized)",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    color_mapping_dict={
        'US_Nominal effective exchange rate - daily - narrow basket': "grey",
        'OIL WTI': "black",
        'Henry Hub Natural Gas Spot Price': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=False,
    save_fig=False,
    file_name="chap_02_daily_exchange_rate_oil_log_diff_vola_normalized_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Rolling correlation of exchange rates, wti oil and natural gas with highlighted crisis periods
#----------------------------------------------------------------------------------------
series_dict_mapping = {
    'EUR/USD': 'DEXUSEU',
    'WTI Oil': 'DCOILWTICO',
    "Natural Gas": "DHHNGSP",
}

start_date = "1960-01-01"
end_date = "2025-10-01"

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
spot_exchange_rate_data_df = pd.concat(list(data_dict.values()), axis=1).dropna()
window = 30
return_type = "log"  # "log" or "pct"

if return_type == "pct":
    log_return_data = np.log(spot_exchange_rate_data_df).diff().dropna()
    return_data = log_return_data.copy()
else:
    pct_return_data = spot_exchange_rate_data_df.pct_change().dropna()
    return_data = pct_return_data.copy()

rolling_corr_data_pct_returns = return_data.rolling(window=window).corr().dropna().unstack().loc[:, [("WTI Oil", "Natural Gas"), ("Natural Gas", "EUR/USD"), ("WTI Oil", "EUR/USD"), ]].copy()
rolling_corr_data_pct_returns.columns = [f"{col[0]} & {col[1]}" for col in rolling_corr_data_pct_returns.columns]

data = rolling_corr_data_pct_returns.copy()
with open("/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods_dict.json", "r") as f:
    crisis_periods_dict = json.load(f)

title=f"Daily rolling correlation between EUR/USD spot exchange rate, oil and gas log first differences with highlighted crisis periods over the time: {data.index[0].year} - {data.index[-1].year}"
x_axis_title="Time"
y_axis_title="Rolling correlation"


fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data,
    crisis_periods_dict=crisis_periods_dict,
    variables=data.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title="",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility (Normalized)",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    color_mapping_dict={
        'WTI Oil & Natural Gas': "grey",
        'Natural Gas & EUR/USD': "black",
        'WTI Oil & EUR/USD': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=True,
    save_fig=False,
    file_name="chap_02_rolling_correlation_exchange_rate_oil_log_diff_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Spot exchange rate distributions (raw data - normalized)
#----------------------------------------------------------------------------------------
series_dict_mapping = {
    'EUR/USD': 'DEXUSEU',
    'WTI Oil': 'DCOILWTICO',
    "Nat Gas": "DHHNGSP",
}

start_date = "1960-01-01"
end_date = "2025-10-01"

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
spot_exchange_rate_data_df = pd.concat(list(data_dict.values()), axis=1).dropna()
# Transform to log values and take first difference
log_spot_exchange_rate_data_df = np.log(spot_exchange_rate_data_df)
log_diff_spot_exchange_rate_data_df = log_spot_exchange_rate_data_df.diff().dropna()
window = 30
spot_exchange_rate_vola_df = log_diff_spot_exchange_rate_data_df.rolling(window=window).std().dropna()
# Normalize the data
spot_exchange_rate_data_df_normed = (spot_exchange_rate_data_df - spot_exchange_rate_data_df.mean()) / spot_exchange_rate_data_df.std()
start_year = spot_exchange_rate_data_df_normed.index.min().strftime('%Y')
end_year = spot_exchange_rate_data_df_normed.index.max().strftime('%Y')
color_mapping_dict = {
    'EUR/USD': 'grey',
    "WTI Oil": 'black',
    "Nat Gas": 'lightgrey'
}
title=f"Normalized daily EUR/USD spot exchange rate, oil and gas over the time range: {start_year} - {end_year}"
x_axis_title="Daily Observations (Normalized)"
y_axis_title="Probability density"

fig = data_graphing_instance.get_fig_histogram(
    data=spot_exchange_rate_data_df_normed,
    variables=spot_exchange_rate_data_df.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    color_discrete_sequence=CAU_COLOR_SCALE,
    color_mapping_dict=color_mapping_dict,
    histnorm="probability density",
    draw_vertical_line_at_0=True,
    showlegend=False,
    save_fig=False,
    file_path=FIGURES_PATH,
    file_name="chap_06_raw_data_normalized_histogram",
    margin_dict=dict(
        l=20,  # Left margin
        r=20,  # Right margin
        t=100,  # Top margin
        b=10   # Bottom margin
    )
)
fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Spot exchange rate distributions (log first differences)
#----------------------------------------------------------------------------------------
start_year = log_diff_spot_exchange_rate_data_df.index.min().strftime('%Y')
end_year = log_diff_spot_exchange_rate_data_df.index.max().strftime('%Y')
color_mapping_dict = {
    'EUR/USD': 'grey',
    "WTI Oil": 'black',
    "Nat Gas": 'lightgrey'
}
title=f"Daily log first differences of EUR/USD spot exchange rate, oil and gas over the time range: {start_year} - {end_year}",
x_axis_title="Daily observations (log first differences)"
y_axis_title="Probability density"

fig = data_graphing_instance.get_fig_histogram(
    data=log_diff_spot_exchange_rate_data_df,
    variables=log_diff_spot_exchange_rate_data_df.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    color_discrete_sequence=CAU_COLOR_SCALE,
    color_mapping_dict=color_mapping_dict,
    histnorm="probability density",
    draw_vertical_line_at_0=True,
    showlegend=False,
    save_fig=False,
    file_path=FIGURES_PATH,
    file_name="chap_06_log_first_diff_histogram",
    margin_dict=dict(
        l=20,  # Left margin
        r=20,  # Right margin
        t=100,  # Top margin
        b=10   # Bottom margin
    )
)
fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Normality (raw data)
#----------------------------------------------------------------------------------------
normality_test_results = test_data_for_normality(
    data=spot_exchange_rate_data_df,
    variables=spot_exchange_rate_data_df.columns.tolist(),
    significance_level=0.05,
    test_shapiro_wilks=True,
    test_anderson_darling=False,
    test_kolmogorov_smirnov=True,
    test_dagostino_k2=True
)
# Export results
data_loading_instance.export_dataframe(
        df=normality_test_results,
        file_name="norm_test_raw_series",
        excel_sheet_name="ADF Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Also export to Latex
data_loading_instance.export_dataframe(
        df=normality_test_results.round(3),
        file_name="norm_test_raw_series",
        latex_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables",
        save_latex=True,
        save_index=False,
        )
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Stationarity - ADF Tests (raw data)
#----------------------------------------------------------------------------------------
adf_test_df = pd.DataFrame()
regression_type_list = ["c", "ct", "ctt", "n"]
for variable in spot_exchange_rate_data_df.columns:
    for regression_type in regression_type_list:
        test_result = adf_test(
            data=spot_exchange_rate_data_df,
            variable=variable,
            title=f"ADF Test for {variable} with regression type {regression_type}",
            significance_level=0.05,
            return_regression_summary=False,
            regression_type=regression_type,
        )
        adf_test_df = pd.concat([adf_test_df, test_result], axis=0)

adf_test_df = adf_test_df.reset_index(drop=True)
adf_test_df["Significance-level"] = 0.05
adf_test_df["p-value < 0.05"] = adf_test_df["p-value"] < 0.05
adf_test_df["Result"] = np.where(adf_test_df["p-value"] < 0.05, "Stationary", "Non-Stationary")
# Export results
data_loading_instance.export_dataframe(
        df=adf_test_df,
        file_name="adf_test_raw_series",
        excel_sheet_name="ADF Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Also export to Latex
data_loading_instance.export_dataframe(
        df=adf_test_df[['ADF Statistic', 'p-value', 'Start Time:', 'End Time:', 'Regression Type', 'Observations:', "Variable", "Result"]].round(3),
        file_name="adf_test_raw_series",
        latex_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables",
        save_latex=True,
        save_index=False,
        )
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Stationarity - ADF Tests (log first differences)
#----------------------------------------------------------------------------------------
# Test again after taking first difference of the log values
spot_exchange_rate_data_df_log_diff = np.log(spot_exchange_rate_data_df).diff().dropna()
adf_test_df = pd.DataFrame()
for variable in spot_exchange_rate_data_df_log_diff.columns:
    for regression_type in regression_type_list:
        test_result = adf_test(
            data=spot_exchange_rate_data_df_log_diff,
            variable=variable,
            title=f"ADF Test for {variable} with regression type {regression_type}",
            significance_level=0.05,
            return_regression_summary=False,
            regression_type=regression_type,
        )
        adf_test_df = pd.concat([adf_test_df, test_result], axis=0)

adf_test_df = adf_test_df.reset_index(drop=True)
adf_test_df["Significance-level"] = 0.05
adf_test_df["p-value < 0.05"] = adf_test_df["p-value"] < 0.05
adf_test_df["Result"] = np.where(adf_test_df["p-value"] < 0.05, "Stationary", "Non-Stationary")
# Export results
data_loading_instance.export_dataframe(
        df=adf_test_df,
        file_name="adf_test_log_diff",
        # excel_sheet_name="ADF Test Results Log Diff",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Also export to Latex
data_loading_instance.export_dataframe(
        df=adf_test_df[['ADF Statistic', 'p-value', 'Start Time:', 'End Time:', 'Regression Type', 'Observations:', "Variable", "Result"]].round(3),
        file_name="adf_test_log_diff",
        latex_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables",
        save_latex=True,
        save_index=False,
        )
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Stationarity - Tests for Cointegration (raw data)
#----------------------------------------------------------------------------------------
trend_type_list = ["c", "ct", "ctt", "n"]
variable_y_list = ["WTI Oil", "Nat Gas"]
cointegration_test_df = pd.DataFrame()
for variable_y in variable_y_list:
    for trend in trend_type_list:
        cointegration_test_result_df = cointegration_test(
            data=spot_exchange_rate_data_df,
            variable_x='EUR/USD',
            variable_y=variable_y,
            significance_level=0.05,
            trend=trend,
        )
        cointegration_test_df = pd.concat([cointegration_test_df, cointegration_test_result_df], axis=0)
cointegration_test_df = cointegration_test_df.reset_index(drop=True)
cointegration_test_df["Significance-level"] = 0.05
cointegration_test_df["p-value < 0.05"] = cointegration_test_df["p-value"] < 0.05
cointegration_test_df["Result"] = np.where(cointegration_test_df["p-value"] < 0.05, "Cointegrated", "Not Cointegrated")
# Export results
data_loading_instance.export_dataframe(
        df=cointegration_test_df,
        file_name="cointegration_test_raw_series",
        excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Also export to Latex
data_loading_instance.export_dataframe(
        df=cointegration_test_df[['Cointegration Score', 'p-value', 'Start Time', 'End Time', 'Observations', 'Trend','Variable X', 'Variable Y', 'Result']].round(3),
        file_name="cointegration_test_raw_series",
        latex_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables",
        save_latex=True,
        save_index=False,
        )
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Stationarity - Tests for Cointegration (log differences)
#----------------------------------------------------------------------------------------
trend_type_list = ["c", "ct", "ctt", "n"]
variable_y_list = ["WTI Oil", "Nat Gas"]
cointegration_test_df = pd.DataFrame()
for variable_y in variable_y_list:
    for trend in trend_type_list:
        cointegration_test_result_df = cointegration_test(
            data=spot_exchange_rate_vola_df,
            variable_x='EUR/USD',
            variable_y=variable_y,
            significance_level=0.05,
            trend=trend,
        )
        cointegration_test_df = pd.concat([cointegration_test_df, cointegration_test_result_df], axis=0)
cointegration_test_df = cointegration_test_df.reset_index(drop=True)
cointegration_test_df["Significance-level"] = 0.05
cointegration_test_df["p-value < 0.05"] = cointegration_test_df["p-value"] < 0.05
cointegration_test_df["Result"] = np.where(cointegration_test_df["p-value"] < 0.05, "Cointegrated", "Not Cointegrated")
# Export results
data_loading_instance.export_dataframe(
        df=cointegration_test_df,
        file_name="coint_test_log_diff_rolvol",
        # excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Also export to Latex
data_loading_instance.export_dataframe(
        df=cointegration_test_df[['Cointegration Score', 'p-value', 'Start Time', 'End Time', 'Observations', 'Trend','Variable X', 'Variable Y', 'Result']].round(3),
        file_name="cointegration_test_log_diff",
        latex_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/tables",
        save_latex=True,
        save_index=False,
        )
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Autocorrelation (raw data)
#----------------------------------------------------------------------------------------
title = f"ACF values for daily observations of the EUR/USD spot exchange rate, oil and gas over the time: {spot_exchange_rate_data_df.index.min().strftime('%Y')} - {spot_exchange_rate_data_df.index.max().strftime('%Y')}"
x_axis_title = "Lags"
y_axis_title = "ACF value"

acf_fig = data_graphing_instance.get_fig_acf(
    data=spot_exchange_rate_data_df,
    variables=spot_exchange_rate_data_df.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    save_fig=False,
    file_name="chap_06_acf_plot_raw_series",
    file_path=FIGURES_PATH,
    nlags=30
    )
acf_fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Autocorrelation (log first differences)
#----------------------------------------------------------------------------------------
title = f"ACF values for daily observations (log first differences) of the spot exchange rate EUR/USD, oil and gas over the time: {spot_exchange_rate_data_df_log_diff.index.min().strftime('%Y')} - {spot_exchange_rate_data_df_log_diff.index.max().strftime('%Y')}"
x_axis_title = "Lags"
y_axis_title = "ACF value"

acf_fig = data_graphing_instance.get_fig_acf(
    data=spot_exchange_rate_data_df_log_diff,
    variables=spot_exchange_rate_data_df_log_diff.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    save_fig=False,
    file_name="chap_06_acf_plot_log_diff",
    file_path=FIGURES_PATH,
    nlags=30
    )
acf_fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Partial Autocorrelation (raw data)
#----------------------------------------------------------------------------------------
title = f"PACF values for daily observations of the spot exchange rate EUR/USD, oil and gas over the time: {spot_exchange_rate_data_df.index.min().strftime('%Y')} - {spot_exchange_rate_data_df.index.max().strftime('%Y')}"
x_axis_title = "Lags"
y_axis_title = "PACF value"

pacf_fig = data_graphing_instance.get_fig_pacf(
    data=spot_exchange_rate_data_df,
    variables=spot_exchange_rate_data_df.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    save_fig=False,
    file_name="chap_06_pacf_plot_raw_series",
    file_path=FIGURES_PATH,
    nlags=30
    )
pacf_fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Tests for Partial Autocorrelation (log first differences)
#----------------------------------------------------------------------------------------
title = f"PACF values for daily observations (log first differences) of the spot exchange rate EUR/USD, oil and gas over the time: {spot_exchange_rate_data_df_log_diff.index.min().strftime('%Y')} - {spot_exchange_rate_data_df_log_diff.index.max().strftime('%Y')}"
x_axis_title = "Lags"
y_axis_title = "PACF value"

pacf_fig = data_graphing_instance.get_fig_pacf(
    data=spot_exchange_rate_data_df_log_diff,
    variables=spot_exchange_rate_data_df_log_diff.columns.tolist(),
    title="",
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
    save_fig=False,
    file_name="chap_06_pacf_plot_log_diff",
    file_path=FIGURES_PATH,
    nlags=30
    )
pacf_fig.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Granger Causality Tests - EUR/USD and WTI Oil (raw data)
#----------------------------------------------------------------------------------------
exchange_rate_pairs = [col for col in spot_exchange_rate_data_df.columns if col.endswith("/USD")]
# for variable_y in exchange_rate_pairs:
granger_test_result, granger_test_result_df_oil = granger_causality_test(
    data=spot_exchange_rate_data_df,
    variable_x='WTI Oil',
    variable_y='EUR/USD',
    max_lag=10,
    significance_level=0.05
)
granger_test_result_df_oil["Significance-level"] = 0.05
granger_test_result_df_oil["p-value < 0.05"] = granger_test_result_df_oil["p-value"] < 0.05
granger_test_result_df_oil["Result"] = np.where(granger_test_result_df_oil["p-value"] < 0.05, "Granger Causality", "No Granger Causality")
# Export results
data_loading_instance.export_dataframe(
        df=granger_test_result_df_oil,
        file_name="granger_causality_test_oil_raw_series",
        excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Plotting the results
variables = granger_test_result_df_oil['Metric'].unique()
secondary_y_variables = ["p-value"]
color_mapping_dict = {
    'p-value': 'red',
    "ssr_ftest": 'darkgrey',
    "ssr_chi2test": 'grey',
    "lrtest": 'black',
    "params_ftest": 'lightgrey',
}
title=f"Granger causality test results testing granger causality of daily observations of oil for EUR/USD over the time: {spot_exchange_rate_data_df.index.min().strftime('%Y')} - {spot_exchange_rate_data_df.index.max().strftime('%Y')}"
secondary_y_axis_title="p-value"

granger_causality_test_plot = data_graphing_instance.plot_granger_test_results(
    data=granger_test_result_df_oil,
    variables=variables,
    secondary_y_variables=secondary_y_variables,
    color_discrete_sequence=["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"],
    title="",
    secondary_y_axis_title=secondary_y_axis_title,
    x_axis_title="Lag",
    y_axis_title="Test-Statistic",
    color_mapping_dict=color_mapping_dict,
    significance_level=0.05,
    margin_dict=dict(
            l=20,  # Left margin
            r=20,  # Right margin
            t=50,  # Top margin
            b=10   # Bottom margin
            ),
    showlegend=False,
    save_fig=False,
    file_name="chap_06_granger_causality_test_oil_raw_series",
    file_path=FIGURES_PATH
    )
granger_causality_test_plot.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Granger Causality Tests - EUR/USD and Nat Gas (raw data)
#----------------------------------------------------------------------------------------
granger_test_result, granger_test_result_df_gas = granger_causality_test(
    data=spot_exchange_rate_data_df,
    variable_x='Nat Gas',
    variable_y='EUR/USD',
    max_lag=10,
    significance_level=0.05
)
granger_test_result_df_gas["Significance-level"] = 0.05
granger_test_result_df_gas["p-value < 0.05"] = granger_test_result_df_gas["p-value"] < 0.05
granger_test_result_df_gas["Result"] = np.where(granger_test_result_df_gas["p-value"] < 0.05, "Granger Causality", "No Granger Causality")
# Export results
data_loading_instance.export_dataframe(
        df=granger_test_result_df_gas,
        file_name="granger_causality_test_gas_raw_series",
        excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Plotting the results
variables = granger_test_result_df_gas['Metric'].unique()
secondary_y_variables = ["p-value"]
title=f"Granger causality test results testing granger causality of daily observations of gas for EUR/USD over the time: {spot_exchange_rate_data_df.index.min().strftime('%Y')} - {spot_exchange_rate_data_df.index.max().strftime('%Y')}"

granger_causality_test_plot = data_graphing_instance.plot_granger_test_results(
    data=granger_test_result_df_gas,
    variables=variables,
    secondary_y_variables=secondary_y_variables,
    color_discrete_sequence=["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"],
    title="",
    secondary_y_axis_title="p-value",
    x_axis_title="Lag",
    y_axis_title="Test-Statistic",
    color_mapping_dict=color_mapping_dict,
    significance_level=0.05,
    margin_dict=dict(
            l=20,  # Left margin
            r=20,  # Right margin
            t=50,  # Top margin
            b=10   # Bottom margin
            ),
    showlegend=False,
    save_fig=False,
    file_name="chap_06_granger_causality_test_gas_raw_series",
    file_path=FIGURES_PATH
    )
granger_causality_test_plot.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Granger Causality Tests - EUR/USD and WTI Oil (log first differences)
#----------------------------------------------------------------------------------------
granger_test_result, granger_test_result_df_oil = granger_causality_test(
    data=spot_exchange_rate_data_df_log_diff,
    variable_x='WTI Oil',
    variable_y='EUR/USD',
    max_lag=10,
    significance_level=0.05
)
granger_test_result_df_oil["Significance-level"] = 0.05
granger_test_result_df_oil["p-value < 0.05"] = granger_test_result_df_oil["p-value"] < 0.05
granger_test_result_df_oil["Result"] = np.where(granger_test_result_df_oil["p-value"] < 0.05, "Granger Causality", "No Granger Causality")
# Export results
data_loading_instance.export_dataframe(
        df=granger_test_result_df_oil,
        file_name="granger_causality_test_oil_log_diff",
        excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Plotting the results
variables = granger_test_result_df_oil['Metric'].unique()
secondary_y_variables = ["p-value"]
color_mapping_dict = {
    'p-value': 'red',
    "ssr_ftest": 'darkgrey',
    "ssr_chi2test": 'grey',
    "lrtest": 'black',
    "params_ftest": 'lightgrey',
}
title=f"Granger causality test results testing granger causality of daily observations (log first differences) of oil for EUR/USD over the time: {spot_exchange_rate_data_df_log_diff.index.min().strftime('%Y')} - {spot_exchange_rate_data_df_log_diff.index.max().strftime('%Y')}"
secondary_y_axis_title="p-value"

granger_causality_test_plot = data_graphing_instance.plot_granger_test_results(
    data=granger_test_result_df_oil,
    variables=variables,
    secondary_y_variables=secondary_y_variables,
    color_discrete_sequence=["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"],
    title="",
    secondary_y_axis_title=secondary_y_axis_title,
    x_axis_title="Lag",
    y_axis_title="Test-Statistic",
    color_mapping_dict=color_mapping_dict,
    significance_level=0.05,
    margin_dict=dict(
            l=20,  # Left margin
            r=20,  # Right margin
            t=50,  # Top margin
            b=10   # Bottom margin
            ),
    showlegend=False,
    save_fig=False,
    file_name="chap_06_granger_causality_test_oil_log_diff",
    file_path=FIGURES_PATH
    )
granger_causality_test_plot.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 06 - Data Characteristics and Stylized Facts - Granger Causality Tests - EUR/USD and Nat Gas (log first differences)
#----------------------------------------------------------------------------------------
granger_test_result, granger_test_result_df_gas = granger_causality_test(
    data=spot_exchange_rate_data_df_log_diff,
    variable_x='Nat Gas',
    variable_y='EUR/USD',
    max_lag=10,
    significance_level=0.05
)
granger_test_result_df_gas["Significance-level"] = 0.05
granger_test_result_df_gas["p-value < 0.05"] = granger_test_result_df_gas["p-value"] < 0.05
granger_test_result_df_gas["Result"] = np.where(granger_test_result_df_gas["p-value"] < 0.05, "Granger Causality", "No Granger Causality")
# Export results
data_loading_instance.export_dataframe(
        df=granger_test_result_df_gas,
        file_name="granger_causality_test_gas_log_diff",
        excel_sheet_name="Granger Test Results",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=False,
        )
# Plotting the results
variables = granger_test_result_df_gas['Metric'].unique()
secondary_y_variables = ["p-value"]
title=f"Granger causality test results testing granger causality of daily observations (log first differences) of gas for EUR/USD over the time: {spot_exchange_rate_data_df_log_diff.index.min().strftime('%Y')} - {spot_exchange_rate_data_df_log_diff.index.max().strftime('%Y')}"

granger_causality_test_plot = data_graphing_instance.plot_granger_test_results(
    data=granger_test_result_df_gas,
    variables=variables,
    secondary_y_variables=secondary_y_variables,
    color_discrete_sequence=["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"],
    title="",
    secondary_y_axis_title="p-value",
    x_axis_title="Lag",
    y_axis_title="Test-Statistic",
    color_mapping_dict=color_mapping_dict,
    significance_level=0.05,
    margin_dict=dict(
            l=20,  # Left margin
            r=20,  # Right margin
            t=50,  # Top margin
            b=10   # Bottom margin
            ),
    showlegend=False,
    save_fig=False,
    file_name="chap_06_granger_causality_test_gas_log_diff",
    file_path=FIGURES_PATH
    )
granger_causality_test_plot.show(renderer="browser")
