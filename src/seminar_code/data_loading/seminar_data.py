from typing import Dict, List, Tuple
import logging
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_loader import DataLoading

data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/SettingsPackages",
    credential_file_name=r"credentials.json"
)
############################### Oil/Gas Production and Consumption ###############################
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
############################### CFTC Open Interest Data ###############################
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
############################### Yahoo Finance Data ###############################
prices_df = data_loading_instance.get_yahoo_data(
    ticker=["CL=F", "NG=F"],
    start_date="1990-01-01",
    end_date="2024-12-31",
    auto_adjust=False
    )
prices_df = data_loading_instance.filter_yahoo_data(
        yahoo_df=prices_df,
        columns=["Adj Close"]
        )

############################### FRED Data I) ###############################
series_dict_mapping = {
    'Headline': 'CPIAUCSL',
    'Core CPI': 'CPILFESL',
    'Energy': 'CPIENGSL',
    'Food': 'CPIUFDSL',
}
start_date = '2000-01-01'
end_date = '2024-12-31'

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_us_df = pd.concat(list(data_dict.values()), axis=1).dropna()
data_us_full_info_table = pd.DataFrame(data_full_info_dict).T


############################### ECB Data ###############################
series_dict_mapping = {
    "HICPX": "ICP.M.U2.N.XEF000.4.ANR",          # HICP excluding Energy and Food
    "Energy": "ICP.M.U2.N.NRGY00.4.ANR",         # HICP Energy euro area
    "Food": "ICP.M.U2.N.FOOD00.4.ANR",           # HICP Food (incl. alcohol & tobacco)
    "HICP": "ICP.M.U2.N.000000.4.ANR",      # HICP Overall Index euro area
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


############################### FRED Data II) ###############################
series_dict_mapping = {
    'US ToT Index': 'W370RG3Q020SBEA', # Terms of Trade: The ratio of export prices to import prices has a documented impact on real exchange rates, especially for commodity-exporting countries.
    # '': '', # Productivity Differentials: The Balassa-Samuelson effect links differences in productivity growth (particularly between traded and non-traded sectors) to real exchange rate movements.
    # '': '', # Net Capital Inflows and Capital Account Openness: Flows of foreign capital, including foreign direct investment and portfolio flows, affect currency valuation and regime changes.
    'US FX Reserves': 'TREAST', # Foreign Exchange Reserves: Changes in reserves can signal central bank intervention and influence regime shifts in exchange rates.
    'US Gov Expend': 'FGEXPND', # Government Consumption/Spending: Fiscal policy shocks may affect the equilibrium real exchange rate.
    'US GDP Growth': 'A191RL1Q225SBEA', # GDP Growth and Real Economic Activity: Fluctuations in output and business cycles are tightly connected to exchange rate regime changes, as identified empirically in many studies.
    # '': '', # Interest Rate Differentials: These affect investor sentiment and international capital flows, contributing to regime changes in currency markets.
    # '': '', # Inflation Differentials: Based on Purchasing Power Parity theory, inflation differences between countries drive changes in real exchange rates.
    # '': '', # Trade Openness: Increased openness can affect the volatility and the nature of regimes in exchange rate behavior.
    # '': '', # Other Relevant Commodity Prices: Besides energy, relevant commodities for the country’s export/import profile (such as metals or agricultural products) can be used as exogenous variables.
    # '': '', # Exchange Rate Regime (de jure and de facto): The country’s own exchange rate policy/arrangement (fixed, floating, managed float) can be considered.
    # '': '', # External shocks: Global financial crisis events or shocks such as sudden oil price falls.
    'VIX Equity': 'VIXCLS', # VIX Index: The VIX index is a widely used measure of market risk and investor sentiment. It reflects the market's expectations for volatility over the coming 30 days and can be a proxy for global financial market uncertainty.
    'VIX Crude Oil': 'OVXCLS', # VIX Crude Oil: Similar to the equity VIX, the oil VIX measures market expectations of near-term volatility in oil prices. It can be particularly relevant for countries heavily involved in oil production or consumption.
}
start_date = '1990-01-01'
end_date = '2024-12-31'

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_series_list = list(data_dict.values())
fred_full_info_table = pd.DataFrame(data_full_info_dict).T


############################### (Real) Exchange Rate Data - BIS ###############################
# Real effective exchange rate - monthly - broad basket
country_keys_mapping = {
    "US": "United States",
    # "XM": "Euro Area",
    # "GB": "United Kingdom",
    # "JP": "Japan",
    # "CH": "Switzerland",
}
exchange_rates_df = data_loading_instance.get_bis_exchange_rate_data(
    country_keys_mapping=country_keys_mapping,
    exchange_rate_type_list=[
        "Real effective exchange rate - monthly - broad basket",
        "Real effective exchange rate - monthly - narrow basket"
        ],
)
# Real effective exchange rate - monthly - broad basket
# Real effective exchange rate - monthly - narrow basket
# Nominal effective exchange rate - monthly - broad basket
# Nominal effective exchange rate - monthly - narrow basket
# Nominal effective exchange rate - daily - narrow basket
# Nominal effective exchange rate - daily - broad basket

series_dict_mapping = {
    'USD-EUR Spot Rate': 'DEXUSEU',
    'USD-JPY Spot Rate': 'DEXJPUS',
    'USD-GBP Spot Rate': 'DEXUSUK',
    'USD-CHF Spot Rate': 'DEXSZUS',
}
start_date = '2000-01-01'
end_date = '2024-12-31'

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_us_df = pd.concat(list(data_dict.values()), axis=1).dropna()
data_us_full_info_table = pd.DataFrame(data_full_info_dict).T



# For the regime switching analysis, first provide a benchmark model: Markov-Switching for the selected interest rate

# Import the necessary library
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching

# Select the interest rate series for the analysis
interest_rate_series = data_dict['Interest Rate']

# Fit a Markov-Switching model
markov_model = MarkovSwitching(interest_rate_series, k_regimes=2)
markov_fit = markov_model.fit()

# Print the summary of the model
print(markov_fit.summary())

# import matplotlib.pyplot as plt

# def export_dataframe(df: pd.DataFrame, txt_path: str, pdf_path: str,
#                      figsize=(8, 4), font_size=10, col_widths=None, title=None, style=None):
#     """
#     Export a DataFrame to a .txt file and a styled .pdf file.

#     Parameters:
#         df (pd.DataFrame): The DataFrame to export.
#         txt_path (str): Path to save the .txt file.
#         pdf_path (str): Path to save the .pdf file.
#         figsize (tuple): Aspect ratio for the PDF (width, height).
#         font_size (int): Font size for table text.
#         col_widths (list or None): List of column widths for the PDF.
#         title (str or None): Optional title for the PDF.
#         style (dict or None): Optional matplotlib table style dict.
#     """
#     # Export to .txt (tab-separated)
#     df.to_csv(txt_path, sep='\t', index=True)

#     # Export to .pdf with matplotlib
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.axis('off')
#     mpl_table = ax.table(cellText=df.values,
#                          colLabels=df.columns,
#                          rowLabels=df.index,
#                          loc='center',
#                          cellLoc='center')
#     mpl_table.auto_set_font_size(False)
#     mpl_table.set_fontsize(font_size)

#     # Set column widths if provided
#     if col_widths:
#         for i, width in enumerate(col_widths):
#             mpl_table.auto_set_column_width(i)
#             for key, cell in mpl_table.get_celld().items():
#                 if key[1] == i:
#                     cell.set_width(width)

#     # Apply custom style if provided
#     if style:
#         for key, cell in mpl_table.get_celld().items():
#             for prop, val in style.items():
#                 setattr(cell, prop, val)

#     # Add title if provided
#     if title:
#         plt.title(title, fontsize=font_size + 2)

#     plt.tight_layout()
#     plt.savefig(pdf_path, bbox_inches='tight')
#     plt.close(fig)


# export_dataframe(
#     df=fred_full_info_table[fred_full_info_table.columns[:-2]],
#     txt_path=r"/Users/Robert_Hennings/Downloads/fred_full_info_table.txt",
#     pdf_path=r"/Users/Robert_Hennings/Downloads/fred_full_info_table.pdf",
#     figsize=(12, 6),
#     font_size=8,
#     col_widths=[0.2]*len(fred_full_info_table.columns),
#     title="FRED Full Info Table",
#     style={"edgecolor": "black"}
# )



# lowest_freq = "QS-OCT"
# data_series_list[1] = data_series_list[1].resample(lowest_freq).ffill()
# data_series_list[4] = data_series_list[4].resample(lowest_freq).ffill()
# data_series_list[5] = data_series_list[5].resample(lowest_freq).ffill()

# data = pd.concat(data_series_list, axis=1).dropna()




# # Standardize data
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)

# # Clustering: try different cluster sizes and pick the best by silhouette score
# import sklearn.metrics

# # List all available metrics in sklearn.metrics
# metrics_list = [m for m in dir(sklearn.metrics) if not m.startswith("_")]
# print("Available sklearn.metrics:")
# for metric in metrics_list:
#     print(metric)

# best_n = 2
# best_score = -1
# n_clusters_list = []
# score_list = []
# for n_clusters in range(2, 10):  # Test a range of clusters
#     model = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = model.fit_predict(data_scaled)
#     score = silhouette_score(data_scaled, labels)
#     n_clusters_list.append(n_clusters)
#     score_list.append(score)
#     if score > best_score:
#         best_score = score
#         best_n = n_clusters
# pd.DataFrame({'n_clusters': n_clusters_list, 'silhouette_score': score_list})


# # Fit final model
# final_model = KMeans(n_clusters=best_n, random_state=42)
# data['regime'] = final_model.fit_predict(data_scaled)

# # View cluster assignments
# print(data['regime'].value_counts())
# print(data)


# import shap
# from sklearn.ensemble import RandomForestClassifier

# # Train classifier to predict regime
# clf = RandomForestClassifier()
# clf.fit(data.drop(columns=['regime']), data['regime'])
# data["regime clf"] = clf.predict(data.drop(columns=['regime']))
# data[["regime", "regime clf"]]

# # Use SHAP
# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(data.drop(columns=['regime']))

# # Plot summary
# shap.summary_plot(shap_values, data.drop(columns=['regime']))