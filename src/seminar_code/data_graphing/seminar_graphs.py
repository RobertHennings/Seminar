from typing import Dict, List, Tuple
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf


CAU_COLOR_SCALE = ["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"]
COLOR_DISCRETE_SEQUENCE_DEFAULT = CAU_COLOR_SCALE
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
#----------------------------------------------------------------------------------------
# 00 - Intro - Deviations of USD spot rates from PPP-values
#----------------------------------------------------------------------------------------
# Source: https://data.imf.org/en/datasets/IMF.STA:EER
# effective_exchange_rates_df = pd.read_csv(r'/Users/Robert_Hennings/Downloads/dataset_2025-09-16T13_25_58.318468730Z_DEFAULT_INTEGRATION_IMF.STA_EER_6.0.0.csv')
# countries_list = ["United States", "United Kingdom", "Japan", "Switzerland", "Euro Area (EA)"]
# indicator_list = ["Real effective exchange rate (REER), Index (2010=100) Adjusted by relative consumer prices"]
# frequency_list = ["Monthly"]

# effective_exchange_rates_df = effective_exchange_rates_df.query("INDICATOR.isin(@indicator_list) and COUNTRY.isin(@countries_list) and FREQUENCY.isin(@frequency_list)").reset_index(drop=True)

# effective_exchange_rates_df_pivoted = effective_exchange_rates_df.pivot(index="TIME_PERIOD", columns="COUNTRY", values="OBS_VALUE").reset_index(drop=False).set_index("TIME_PERIOD", drop=True).dropna()
# effective_exchange_rates_df_pivoted.index = effective_exchange_rates_df_pivoted.index.str.replace("M", "") + "-01"
# effective_exchange_rates_df_pivoted.index = pd.to_datetime(effective_exchange_rates_df_pivoted.index, format="mixed")
# effective_exchange_rates_df_pivoted = effective_exchange_rates_df_pivoted.sort_index()
# effective_exchange_rates_df_pivoted["GBPREAL"] = np.log(effective_exchange_rates_df_pivoted["United Kingdom"]) - np.log(effective_exchange_rates_df_pivoted["United States"])
# effective_exchange_rates_df_pivoted["JPYREAL"] = np.log(effective_exchange_rates_df_pivoted["Japan"]) - np.log(effective_exchange_rates_df_pivoted["United States"])
# effective_exchange_rates_df_pivoted["CHFREAL"] = np.log(effective_exchange_rates_df_pivoted["Switzerland"]) - np.log(effective_exchange_rates_df_pivoted["United States"])
# effective_exchange_rates_df_pivoted["EUROREAL"] = np.log(effective_exchange_rates_df_pivoted["Euro Area (EA)"]) - np.log(effective_exchange_rates_df_pivoted["United States"])

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
color_discrete_sequence = ["grey", "black", "#9b0a7d"]
title = f"Deviations of USD Spot Rate from PPP-values over the time: {data.index[0].year} - {data.index[-1].year}"
x_axis_title = "Date"
y_axis_title = "Deviation from PPP-values (log)"
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
        title=title,
        secondary_y_axis_title=secondary_yaxis_title,
        color_discrete_sequence=color_discrete_sequence,
        num_years_interval_x_axis=NUM_YEARS_INTERVAL_X_AXIS,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="deviations_of_usd_spotrates_from_ppp_values",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_deviations_from_ppp.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state
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
title = f"Oil Consumption by Country per year (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
y_axis_title = "Oil Consumption (in terawatt-hours)"
secondary_y_axis_title = "World Oil Consumption (in terawatt-hours)"


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
        file_name="oil_consumption_by_country",
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

title = f"Oil Production by Country per year (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
y_axis_title = "Oil Production (in terawatt-hours)"
secondary_y_axis_title = "World Oil Production (in terawatt-hours)"

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
        file_name="oil_production_by_country",
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
subplot_titles=("Oil Consumption by Country by Year", "Oil Production by Country by Year")
title = f"Yearly Oil Consumption and Production by Country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
secondary_y_variable = "World"

fig_oil_consumption_production_combine = data_graphing_instance.get_combined_production_consumption_graph(
        subplot_titles=list(subplot_titles),
        title=title,
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
        file_name="oil_consumption_production_combined_graph",
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

title = f"Gas Consumption by Country per year (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
y_axis_title = "Gas Consumption (in terawatt-hours)"
secondary_y_axis_title = "World Gas Consumption (in terawatt-hours)"

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
        file_name="gas_consumption_by_country",
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

title = f"Gas Production by Country per year (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
y_axis_title = "Gas Production (in terawatt-hours)"
secondary_y_axis_title = "World Gas Production (in terawatt-hours)"


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
        file_name="gas_production_by_country",
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
subplot_titles=("Gas Consumption by Country by Year", "Gas Production by Country by Year")
title = f"Yearly Gas Consumption and Production by Country (in terawatt-hours) over the time: {start_year} - {end_year}"
x_axis_title = "Year"
secondary_y_variable = "World"

fig_gas_consumption_production_combine = data_graphing_instance.get_combined_production_consumption_graph(
        subplot_titles=subplot_titles,
        title=title,
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
        file_name="gas_consumption_production_combined_graph",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_gas_consumption_production_combine.show(renderer="browser")
# ----------------------------------------------------------------------------------------
# 01 - Modern Commodities Markets - The current state - CFTC Commitment of Traders Data
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

title = f"Oil Open Interest as Total number of Contracts and in USD over the time: {start_year} - {end_year}"
x_axis_title = "Date"
y_axis_title = "Open Interest (number of contracts)"
secondary_y_axis_title = "Open Interest (in USD)"

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
        file_name="oil_open_interest",
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


title = f"Gas Open Interest as Total number of Contracts and in USD over the time: {start_year} - {end_year}"
x_axis_title = "Date"
y_axis_title = "Open Interest (number of contracts)"
secondary_y_axis_title = "Open Interest (in USD)"

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
        file_name="gas_open_interest",
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
subplot_titles=("Oil Open Interest as Total number of Contracts and in USD", "Gas Open Interest as Total number of Contracts and in USD")
start_year = fig_gas_oi.data[0].x[0]
end_year = fig_gas_oi.data[0].x[-1]
title = f"Open Interest of Oil and Gas Products over the time: {pd.Timestamp(start_year).strftime('%Y')} - {pd.Timestamp(end_year).strftime('%Y')}"
x_axis_title = "Date"

fig_gas_oil_open_interest_combine = data_graphing_instance.get_combined_open_interest_graph(
        subplot_titles=subplot_titles,
        title=title,
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
        file_name="open_interest_oil_gas_combined_graph.pdf",
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
title = f"US CPI: Headline and Component Contributions over the time: {contributions.index[0].year} - {contributions.index[-1].year}"
x_axis_title = "Date"
y_axis_title = 'Contribution to Annual Inflation (%)'

fig_inflation_decomp_usa = data_graphing_instance.get_fig_inflation_contribution_usa(
        data=contributions,
        data_pct=data_pct,
        cpi_color=cpi_color,
        variables=list(bar_components),
        title=title,
        secondary_y_variables=[],
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="us_cpi_inflation_decomposition",
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
title = f"EU Area CPI: Headline and Component Contributions over the time: {contributions.index[0].year} - {contributions.index[-1].year}"
x_axis_title = "Date"
y_axis_title = 'Contribution to Annual Inflation (%)'

fig_inflation_decomp_euro_area = data_graphing_instance.get_fig_inflation_contribution_euro_area(
        data=contributions,
        cpi_color=cpi_color,
        variables=variables,
        secondary_y_variables=[],
        title=title,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        color_mapping_dict=color_mapping,
        save_fig=False,
        file_name="eu_area_cpi_inflation_decomposition",
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
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index - Rolling Correlation
#----------------------------------------------------------------------------------------
window = 30
return_type = "log"  # "log" or "pct"

if return_type == "pct":
    log_return_data = np.log(us_data).diff().dropna()
    return_data = log_return_data.copy()
else:
    pct_return_data = us_data.pct_change().dropna()
    return_data = pct_return_data.copy()

rolling_corr_data_pct_returns = return_data.rolling(window=window).corr().dropna().unstack().loc[:, [("WTI Oil", "Natural Gas"), ("Natural Gas", "USD Index"), ("WTI Oil", "USD Index"), ]].copy()
rolling_corr_data_pct_returns.columns = [f"{col[0]} & {col[1]}" for col in rolling_corr_data_pct_returns.columns]

data = rolling_corr_data_pct_returns.copy()

variables = data.columns
secondary_yaxis_variables = []
title = f"WTI Oil, Natural Gas and USD Index percentage returns linear Correlation over the time: {data.index[0].year} - {data.index[-1].year} with rolling window of {window} days"
x_axis_title = "Date"
y_axis_title = "Linear Correlation"
secondary_yaxis_title = ""

color_mapping = {
    'WTI Oil & Natural Gas': "grey",
    'Natural Gas & USD Index': "black",
    'WTI Oil & USD Index': "#9b0a7d",
}

fig_rolling_correlation = data_graphing_instance.get_fig_rolling_correlation(
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
        file_name=f"oil_gas_usd_index_lin_correlation_{window}_pct_returns",
        file_path=FIGURES_PATH,
        width=1200,
        height=800,
        scale=3
        )
# Show the figure
fig_rolling_correlation.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index - Rolling Volatility and Crisis Periods
#----------------------------------------------------------------------------------------
crisis_periods_dict = {
    "Bretton Woods Breakdown": {
        "start": "1971-08-15",
        "end": "1973-03-19"
    },
    "Nixon Shock": {
        "start": "1971-08-15",
        "end": "1973-03-19"
    },
    "Oil Crisis I": {
        "start": "1973-10-17",
        "end": "1974-03-01"
    },
    "Oil Crisis II": {
        "start": "1979-01-01",
        "end": "1981-03-01"
    },
    "Black Monday Crash": {
        "start": "1987-10-19",
        "end": "1987-10-19"
    },
    "Asian Financial Crisis": {
        "start": "1997-07-02",
        "end": "1998-12-31"
    },
    "Russian Crisis": {
        "start": "1998-08-17",
        "end": "1998-09-01"
    },
    "Dot-com Bubble": {
        "start": "2000-03-01",
        "end": "2002-10-01"
    },
    "Global Financial Crisis": {
        "start": "2007-08-09",
        "end": "2009-03-09"
    },
    "US QE": {
        "start": "2008-11-25",
        "end": "2014-12-31"
    },
    "US Debt Ceiling Crisis": {
        "start": "2011-08-20",
        "end": "2011-08-05"
    },
    "US-China Trade War": {
        "start": "2018-03-22",
        "end": "2020-01-15"
    },
    "European Debt Crisis": {
        "start": "2009-10-01",
        "end": "2012-12-31"
    },
    "COVID-19 Pandemic": {
        "start": "2020-02-20",
        "end": "2021-11-16"
    },
    "Russia-Ukraine War": {
        "start": "2022-02-24",
        "end": "2022-06-01"
    },
}

#  Load additionally all US Recession Periods from FRED
start_date = '1964-01-01'
end_date = '2024-12-31'

series_dict_mapping = {
    "US Recessions": "USREC"
}

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
recession_periods = []
in_recession = False
for date, rec in data_dict["US Recessions"].items():
    if rec == 1 and not in_recession:
        start = date
        in_recession = True
    elif rec == 0 and in_recession:
        end = date
        recession_periods.append({"start": start, "end": end})
        in_recession = False
# If the last period is still in recession, close it at the last date
if in_recession:
    recession_periods.append({"start": start, "end": data_dict["US Recessions"].index[-1]})

# Append US recession periods to crisis_periods_dict
for period in recession_periods:
    name = f"US Recession {period['start'].year}-{period['end'].year}"
    crisis_periods_dict[name] = {
        "start": period["start"],
        "end": period["end"]
    }

crisis_periods_df = pd.DataFrame(crisis_periods_dict).T
crisis_periods_df["start"] = pd.to_datetime(crisis_periods_df["start"]).dt.strftime("%Y-%m-%d")
crisis_periods_df["end"] = pd.to_datetime(crisis_periods_df["end"]).dt.strftime("%Y-%m-%d")
crisis_periods_df.columns = ["Start-date", "End-date"]
crisis_periods_df["Event-Type"] = ["US-Recession" if "US Recession" in name else "Major Global Crisis" for name in crisis_periods_df.index]
crisis_periods_df["Source"] = ["FRED: USREC" if "US Recession" in name else "Various" for name in crisis_periods_df.index]
crisis_periods_df = crisis_periods_df.sort_values(by="Start-date")
# Save the crisis_periods_dict as json and the dataframe to an excel file
file_name = "crisis_periods"

data_loading_instance.export_dataframe(
    df=crisis_periods_df,
    file_name=file_name,
    excel_path=TABLES_PATH,
    txt_path=TABLES_PATH,
    pdf_path=FIGURES_PATH,
    json_path=f"{DATA_PATH}/raw/",
    figsize=(12, 6),
    font_size=8,
    col_widths=[0.2]*len(crisis_periods_df.columns),
    title="Crisis Periods",
    style={"edgecolor": "black"},
    font_family="Arial",
    font_weight="bold"
)

# Now load the exchange rate data from BIS - daily data
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
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index - Rolling Volatility and Crisis Periods - Raw Vola
#----------------------------------------------------------------------------------------
fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data,
    crisis_periods_dict=crisis_periods_dict,
    variables=data.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {data.index[0].year} - {data.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility",
    color_mapping_dict={
        'US_Nominal effective exchange rate - daily - narrow basket': "grey",
        'OIL WTI': "black",
        'Henry Hub Natural Gas Spot Price': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=False,
    save_fig=False,
    file_name="exchange_rate_oil_raw_vola_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
# Normalize the Volatility Data to compare the levels
data_normalized = (data - data.min()) / (data.max() - data.min())
fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data_normalized,
    crisis_periods_dict=crisis_periods_dict,
    variables=data_normalized.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Normalized Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {data.index[0].year} - {data.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility (Normalized)",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility (Normalized)",
    color_mapping_dict={
        'US_Nominal effective exchange rate - daily - narrow basket': "grey",
        'OIL WTI': "black",
        'Henry Hub Natural Gas Spot Price': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=False,
    save_fig=False,
    file_name="exchange_rate_oil_raw_vola_normalized_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
#----------------------------------------------------------------------------------------
# 02 - Research Hypothesis - Main Relationships between Oil, Gas and USD Index - Rolling Volatility and Crisis Periods - Log Diff Vola
#----------------------------------------------------------------------------------------
fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data_log_first_diff,
    crisis_periods_dict=crisis_periods_dict,
    variables=data_log_first_diff.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {data.index[0].year} - {data.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility",
    color_mapping_dict={
        'US_Nominal effective exchange rate - daily - narrow basket': "grey",
        'OIL WTI': "black",
        'Henry Hub Natural Gas Spot Price': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=False,
    save_fig=False,
    file_name="exchange_rate_oil_log_diff_vola_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
# Normalize the Volatility Data to compare the levels
data_log_diff_normalized = (data_log_first_diff - data_log_first_diff.min()) / (data_log_first_diff.max() - data_log_first_diff.min())
fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=data_log_diff_normalized,
    crisis_periods_dict=crisis_periods_dict,
    variables=data_log_diff_normalized.columns,
    secondary_y_variables=[],
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Normalized Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {data.index[0].year} - {data.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility (Normalized)",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility (Normalized)",
    color_mapping_dict={
        'US_Nominal effective exchange rate - daily - narrow basket': "grey",
        'OIL WTI': "black",
        'Henry Hub Natural Gas Spot Price': "#9b0a7d",
    },
    num_years_interval_x_axis=5,
    showlegend=False,
    save_fig=False,
    file_name="exchange_rate_oil_log_diff_vola_normalized_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")