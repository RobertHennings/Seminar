import os
import pandas as pd
import numpy as np

SEMINAR_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar"
SEMINAR_CODE_PATH = rf"{SEMINAR_PATH}/src/seminar_code"
MODELS_PATH = rf"{SEMINAR_CODE_PATH}/models"
FIGURES_PATH = rf"{SEMINAR_PATH}/reports/figures"
TABLES_PATH = rf"{SEMINAR_PATH}/reports/tables"
DATA_PATH = rf"{SEMINAR_PATH}/data"
PRESENTATION_DATA = rf"{SEMINAR_PATH}/reports/presentation_latex_version/data"

print(os.getcwd())
os.chdir(SEMINAR_CODE_PATH)
print(os.getcwd())

from data_loading.data_loader import DataLoading
from model.architecture import ModelObject
from data_graphing.data_grapher import DataGraphing

# Import util functions for model evaluation and comparisons
from utils.evaluation import get_model_metadata_df,\
    extract_predicted_labels_from_metadata_df,\
    get_recoded_predicted_labels_df, \
    get_regime_counts_df, \
    get_overlapping_regimes_df, \
    get_periods_overlaying_df, \
    parse_summary_file

# Import custom written model evaluation scores
from utils.evaluation_metrics import compute_rcm

data_graphing_instance = DataGraphing()
data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Basic Full time UIP Regression - BIS Central Bank Policy Rates
#----------------------------------------------------------------------------------------
# Load the BIS Central Bank Policy Rate data for the relevant countries as a proxy for the interest rates
country_keys_mapping = {
    "US": "United States",
    "XM": "Euro area",
}
central_bank_policy_rate_df = data_loading_instance.get_bis_central_bank_policy_rate_data(
        country_keys_mapping=country_keys_mapping,
        exchange_rate_type_list=["Central bank policy rate - daily"],
        )
# Rename the columns
central_bank_policy_rate_df.columns = [f"{country}_CBPR" for country in country_keys_mapping.keys()]
central_bank_policy_rate_df = central_bank_policy_rate_df.dropna()

# Now in every identified regime try to check the uncovered interest rate parity condition
# E_t[S_t+1] - S_t = (i_t - i*_t)
# where E_t[S_t+1] is the expected future spot exchange rate at time t
# S_t is the current spot exchange rate at time t
# i_t is the domestic interest rate at time t
# i*_t is the foreign interest rate at time t
# The left hand side is the expected appreciation/depreciation of the exchange rate
# The right hand side is the interest rate differential
# We can approximate the expected future spot exchange rate with the actual future spot exchange rate
# So we need to shift the exchange rate changes by -1 to get the future spot exchange rate
# We will do this for the US Dollar against the Euro, GBP, JPY and CHF
# Therefore we need the exchange rates for these currencies as well
# Load the Spot Exchange rates from FRED for the currency pairs
series_dict_mapping = {
    'USD/EUR': 'DEXUSEU',
}

start_date = central_bank_policy_rate_df.index.min().strftime('%Y-%m-%d')
end_date = central_bank_policy_rate_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_spot_rates_df = pd.concat(list(data_dict.values()), axis=1).dropna()
# Compute the EUR/USD rate
data_spot_rates_df["EUR/USD"] = 1 / data_spot_rates_df["USD/EUR"]
data_spot_rates_df = data_spot_rates_df.drop(columns=["USD/EUR"])

data_us_full_info_table = pd.DataFrame(data_full_info_dict).T

# Relevel both datasets to have the same length
data_spot_rates_df = data_spot_rates_df.reindex(central_bank_policy_rate_df.index).dropna()
# 1) Compute the log differentials of the spot exchange rates
log_spot_rate_diff_df = (np.log(data_spot_rates_df.shift(1)) - np.log(data_spot_rates_df)).dropna()
log_spot_rate_diff_df.columns = [col.split('/')[0] for col in log_spot_rate_diff_df.columns]
# 2)Compute the log differentials of the interest rate differentials
interest_rate_diff_df = pd.DataFrame()
for country in country_keys_mapping.keys():
    if country == "US":
        continue
    interest_rate_diff_df[f'i_diff_{country}'] = central_bank_policy_rate_df['US_CBPR'] - central_bank_policy_rate_df[f'{country}_CBPR']
interest_rate_diff_df = interest_rate_diff_df.rename(
    columns={
        "i_diff_XM": "i_diff_EUR",
    })
# Now combine both datasets
uip_data_df = pd.concat([log_spot_rate_diff_df, interest_rate_diff_df], axis=1).dropna()

def run_uip_regression(dep_var: str, indep_var: str, data: pd.DataFrame, cov_type: str="nonrobust", use_t: bool=True):
    import statsmodels.api as sm
    X = sm.add_constant(data[indep_var])
    y = data[dep_var]
    model = sm.OLS(y, X).fit(cov_type=cov_type, use_t=use_t)
    return model

# First run the UIP regression on the full sample
full_sample_uip_results = {}
currency_pairs = ['EUR']
for currency in currency_pairs:
    dep_var = f'{currency}'
    indep_var = f'i_diff_{currency}'
    model = run_uip_regression(dep_var, indep_var, uip_data_df, cov_type="HC1")
    full_sample_uip_results[currency] = model
# Print the summary of the full sample UIP regressions
for currency, model in full_sample_uip_results.items():
    print(f"Full Sample UIP Regression Results for {currency}/USD:")
    print(model.summary())
    print("\n")
# The correct Hypothesis for the tested parameters are:
# H0: β0 = 0 (no constant term), ß1 = 1 (interest rate differential fully explains exchange rate changes)
# H1: β0 ≠ 0, ß1 ≠ 1
estimated_params_df = pd.DataFrame(model.summary().tables[1].data)
estimated_params_df.columns = estimated_params_df.iloc[0]
estimated_params_df = estimated_params_df[1:]
# Transfer all columns to numeric where possible
estimated_params_df = estimated_params_df.apply(pd.to_numeric, errors='ignore')
# Therefore we have to adjust the t-test and p-values accordingly
from scipy import stats
corrected_t_i_diff = (estimated_params_df["coef"][2] - 1.0) / estimated_params_df["std err"][2]
corrected_p_i_diff = 2 * (1 - stats.t.cdf(np.abs(corrected_t_i_diff), df=model.df_resid))
# Save the model summary - save as .csv so later all summary files can be parsed for a better comparison
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=model.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_standard_full_time_uip_regression_central_bank_policy_rates",
)
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Basic Full time UIP Regression - 3M Interbank Lending Rates
#----------------------------------------------------------------------------------------
file_name = r"irates3m.csv"
full_file_path = rf"{DATA_PATH}/raw/{file_name}"
interbank_rates_df = pd.read_csv(full_file_path, sep=";")

interbank_rates_df["Date"] = interbank_rates_df["Year"].astype(str) + "-" + interbank_rates_df["Month"].astype(str).str.zfill(2) + "-" + interbank_rates_df["Day"].astype(str).str.zfill(2)
interbank_rates_df = interbank_rates_df.set_index(pd.to_datetime(interbank_rates_df["Date"]))
interbank_rates_df = interbank_rates_df[["USD", "EUR"]]
# Since these are the 3M rates but published in an annual term, we have to convert them by dividing by 4
interbank_rates_df = interbank_rates_df / 4
interbank_rates_df["i_diff_EUR"] = interbank_rates_df["USD"] - interbank_rates_df["EUR"]

uip_data_df = pd.concat([log_spot_rate_diff_df, interbank_rates_df["i_diff_EUR"]], axis=1).dropna()

# Save the data locally
data_loading_instance.export_dataframe(
    df=uip_data_df,
    file_name="chap_04_uip_data_df_3m_interbank_lending_rates",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=True,
)

full_sample_uip_results = {}
currency_pairs = ['EUR']
for currency in currency_pairs:
    dep_var = f'{currency}'
    indep_var = f'i_diff_{currency}'
    model = run_uip_regression(dep_var, indep_var, uip_data_df, cov_type="HC1")
    full_sample_uip_results[currency] = model
# Print the summary of the full sample UIP regressions
for currency, model in full_sample_uip_results.items():
    print(f"Full Sample UIP Regression Results for {currency}/USD:")
    print(model.summary())
    print("\n")
# The correct Hypothesis for the tested parameters are:
# H0: β0 = 0 (no constant term), ß1 = 1 (interest rate differential fully explains exchange rate changes)
# H1: β0 ≠ 0, ß1 ≠ 1
estimated_params_df = pd.DataFrame(model.summary().tables[1].data)
estimated_params_df.columns = estimated_params_df.iloc[0]
estimated_params_df = estimated_params_df[1:]
# Transfer all columns to numeric where possible
estimated_params_df = estimated_params_df.apply(pd.to_numeric, errors='ignore')
# Therefore we have to adjust the t-test and p-values accordingly
from scipy import stats
corrected_t_i_diff = (estimated_params_df["coef"][2] - 1.0) / estimated_params_df["std err"][2]
corrected_p_i_diff = 2 * (1 - stats.t.cdf(np.abs(corrected_t_i_diff), df=model.df_resid))
# Save the model summary
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=model.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_standard_full_time_uip_regression_3m_interbank_lending_rates",
)
# Observe the error terms with Durbin-Watson and Ljung-Box tests for autocorrelated error terms
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Standard - Central Bank Policy Rates - Model B1
#----------------------------------------------------------------------------------------
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
country_keys_mapping = {
    "US": "United States",
    "XM": "Euro area",
}
central_bank_policy_rate_df = data_loading_instance.get_bis_central_bank_policy_rate_data(
        country_keys_mapping=country_keys_mapping,
        exchange_rate_type_list=["Central bank policy rate - daily"],
        )
# Rename the columns
central_bank_policy_rate_df.columns = [f"{country}_CBPR" for country in country_keys_mapping.keys()]
central_bank_policy_rate_df = central_bank_policy_rate_df.dropna()

series_dict_mapping = {
    'USD/EUR': 'DEXUSEU',
}

start_date = central_bank_policy_rate_df.index.min().strftime('%Y-%m-%d')
end_date = central_bank_policy_rate_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_spot_rates_df = pd.concat(list(data_dict.values()), axis=1).dropna()
# Compute the EUR/USD rate
data_spot_rates_df["EUR/USD"] = 1 / data_spot_rates_df["USD/EUR"]
data_spot_rates_df = data_spot_rates_df.drop(columns=["USD/EUR"])

data_us_full_info_table = pd.DataFrame(data_full_info_dict).T

# Relevel both datasets to have the same length
data_spot_rates_df = data_spot_rates_df.reindex(central_bank_policy_rate_df.index).dropna()
# 1) Compute the log differentials of the spot exchange rates
log_spot_rate_diff_df = (np.log(data_spot_rates_df.shift(1)) - np.log(data_spot_rates_df)).dropna()
log_spot_rate_diff_df.columns = [col.split('/')[0] for col in log_spot_rate_diff_df.columns]
# 2)Compute the log differentials of the interest rate differentials
interest_rate_diff_df = pd.DataFrame()
for country in country_keys_mapping.keys():
    if country == "US":
        continue
    interest_rate_diff_df[f'i_diff_{country}'] = central_bank_policy_rate_df['US_CBPR'] - central_bank_policy_rate_df[f'{country}_CBPR']
interest_rate_diff_df = interest_rate_diff_df.rename(
    columns={
        "i_diff_XM": "i_diff_EUR",
    })
# Now combine both datasets
uip_data_df = pd.concat([log_spot_rate_diff_df, interest_rate_diff_df], axis=1).dropna()

msm = MarkovRegression(
    endog=uip_data_df['EUR'],
    exog=uip_data_df[['i_diff_EUR']],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
)
msm_fit = msm.fit(em_iter=10, search_reps=20)
print("Markov-Switching UIP Regression Results for EUR/USD:")
print(msm_fit.summary())
# Save the model summary
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=msm_fit.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_msm_central_bank_policy_rates_model_b1",
)
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Oil, Gas added - Central Bank Policy Rates - Model B1
#----------------------------------------------------------------------------------------
# Additionally to the EUR/USD spot exchange rate and the interest rate differential load the oil and gas data 
series_dict_mapping = {
    'USD/EUR': 'DEXUSEU',
    'WTI Oil': 'DCOILWTICO',
    "Nat Gas": "DHHNGSP",
}
start_date = central_bank_policy_rate_df.index.min().strftime('%Y-%m-%d')
end_date = central_bank_policy_rate_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
spot_exchange_rate_data_df = pd.concat(list(data_dict.values()), axis=1).dropna()
# Compute the EUR/USD rate
spot_exchange_rate_data_df["EUR/USD"] = 1 / spot_exchange_rate_data_df["USD/EUR"]
spot_exchange_rate_data_df = spot_exchange_rate_data_df.drop(columns=["USD/EUR"])

spot_exchange_rate_data_df_log_diff = np.log(spot_exchange_rate_data_df).diff().dropna()
# Next we need the rolling volatility of the energy commodities
window = 30
spot_exchange_rate_data_df_log_diff_rolling = spot_exchange_rate_data_df_log_diff.rolling(window=window).std().dropna()
# Compute the interest rate differentials
interest_rate_diff_df = pd.DataFrame()
for country in country_keys_mapping.keys():
    if country == "US":
        continue
    interest_rate_diff_df[f'i_diff_{country}'] = central_bank_policy_rate_df['US_CBPR'] - central_bank_policy_rate_df[f'{country}_CBPR']
interest_rate_diff_df = interest_rate_diff_df.rename(
    columns={
        "i_diff_XM": "i_diff_EUR",
    })
# Also add the Trading Volume - Reuters Data
oi_tv_oil_gas_reuters_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/open_interest_trading_volume_oil_gas_reuters.xlsx")
oi_tv_oil_gas_reuters_df["Date"] = pd.to_datetime(oi_tv_oil_gas_reuters_df["Date"], format="%Y-%m-%d")
oi_tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df.set_index("Date", drop=True)
# Separate out the Open-Interest
oi_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "open interest" in col.lower()]
tv_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "volume" in col.lower()]
tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df[tv_columns].dropna().copy()
tv_oil_gas_reuters_df.columns = ["WTI Oil TV", "Nat Gas TV"]
# Now combine both datasets
uip_data_df = pd.concat([
    spot_exchange_rate_data_df_log_diff["EUR/USD"],
    interest_rate_diff_df,
    spot_exchange_rate_data_df_log_diff_rolling[["WTI Oil", "Nat Gas"]],
    tv_oil_gas_reuters_df
    ], axis=1).dropna()

# Save the data locally
data_loading_instance.export_dataframe(
    df=uip_data_df,
    file_name="chap_04_uip_data_df_central_bank_policy_rates_oil_gas_rol_vol_b1",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=True,
)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
msm = MarkovRegression(
    endog=uip_data_df['EUR/USD'],
    exog=uip_data_df[['i_diff_EUR', 'WTI Oil', 'Nat Gas', "WTI Oil TV", "Nat Gas TV"]],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
)
msm_fit = msm.fit(em_iter=100, search_reps=200)
print("Markov-Switching UIP Regression Results for EUR/USD:")
print(msm_fit.summary())
# Save the model summary
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=msm_fit.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_msm_central_bank_policy_rates_oil_gas_rol_vol_model_b1",
)
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Standard - 3M Interbank rates - Model B2
#----------------------------------------------------------------------------------------
series_dict_mapping = {
    'USD/EUR': 'DEXUSEU',
}

start_date = central_bank_policy_rate_df.index.min().strftime('%Y-%m-%d')
end_date = central_bank_policy_rate_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_spot_rates_df = pd.concat(list(data_dict.values()), axis=1).dropna()
# Compute the EUR/USD rate
data_spot_rates_df["EUR/USD"] = 1 / data_spot_rates_df["USD/EUR"]
data_spot_rates_df = data_spot_rates_df.drop(columns=["USD/EUR"])

data_us_full_info_table = pd.DataFrame(data_full_info_dict).T

# 1) Compute the log differentials of the spot exchange rates
log_spot_rate_diff_df = (np.log(data_spot_rates_df.shift(1)) - np.log(data_spot_rates_df)).dropna()
# log_spot_rate_diff_df.columns = [col.split('/')[0] for col in log_spot_rate_diff_df.columns]


interbank_rates_df = pd.read_csv(r"/Users/Robert_Hennings/Uni/Master/Applied_Econometrics_of_Foreign_Exchange_Markets/Data/irates3m.csv", sep=";")
interbank_rates_df["Date"] = interbank_rates_df["Year"].astype(str) + "-" + interbank_rates_df["Month"].astype(str).str.zfill(2) + "-" + interbank_rates_df["Day"].astype(str).str.zfill(2)
interbank_rates_df = interbank_rates_df.set_index(pd.to_datetime(interbank_rates_df["Date"]))
interbank_rates_df = interbank_rates_df[["USD", "EUR"]]
# Since these are the 3M rates but published in an annual term, we have to convert them by dividing by 4
interbank_rates_df = interbank_rates_df / 4
interbank_rates_df["i_diff_EUR"] = interbank_rates_df["USD"] - interbank_rates_df["EUR"]

uip_data_df = pd.concat([log_spot_rate_diff_df, interbank_rates_df["i_diff_EUR"]], axis=1).dropna()


# Save the data locally
data_loading_instance.export_dataframe(
    df=uip_data_df,
    file_name="chap_04_uip_data_df_3m_interbank_lending_rates_b2",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=True,
)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
msm = MarkovRegression(
    endog=uip_data_df['EUR/USD'],
    exog=uip_data_df[['i_diff_EUR']],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
)
msm_fit = msm.fit(em_iter=10, search_reps=20)
print("Markov-Switching UIP Regression Results for EUR/USD:")
print(msm_fit.summary())
# Save the model summary
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=msm_fit.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_msm_3m_interbank_lending_rates_model_b2",
)
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Oil, Gas added - 3M Interbank rates - Model B2
#----------------------------------------------------------------------------------------
# Also add the Trading Volume - Reuters Data
oi_tv_oil_gas_reuters_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/open_interest_trading_volume_oil_gas_reuters.xlsx")
oi_tv_oil_gas_reuters_df["Date"] = pd.to_datetime(oi_tv_oil_gas_reuters_df["Date"], format="%Y-%m-%d")
oi_tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df.set_index("Date", drop=True)
# Separate out the Open-Interest
oi_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "open interest" in col.lower()]
tv_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "volume" in col.lower()]
tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df[tv_columns].dropna().copy()
tv_oil_gas_reuters_df.columns = ["WTI Oil TV", "Nat Gas TV"]

uip_data_df = pd.concat([
    log_spot_rate_diff_df,
    interbank_rates_df["i_diff_EUR"],
    tv_oil_gas_reuters_df,
    spot_exchange_rate_data_df_log_diff_rolling[["WTI Oil", "Nat Gas"]]
    ],
    axis=1).dropna()

# Save the data locally
data_loading_instance.export_dataframe(
    df=uip_data_df,
    file_name="chap_04_uip_data_df_3m_interbank_lending_rates_oil_gas_rol_vol_model_b2",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=True,
)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
msm = MarkovRegression(
    endog=uip_data_df['EUR/USD'],
    exog=uip_data_df[['i_diff_EUR', "WTI Oil", "Nat Gas", 'WTI Oil TV', 'Nat Gas TV']],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
)
msm_fit = msm.fit(em_iter=10, search_reps=20)
print("Markov-Switching UIP Regression Results for EUR/USD:")
print(msm_fit.summary())
# Save the model summary
model_object_instance = ModelObject()
model_object_instance.save_model_summary(
    model_summary=msm_fit.summary(),
    save_txt=True,
    file_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
    file_name="chap_07_msm_3m_interbank_lending_rates_oil_gas_rol_vol_model_b2",
)
#----------------------------------------------------------------------------------------
# 07 - Model Results - Saving all model coefficients for comparison
#----------------------------------------------------------------------------------------
file_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results"
file_name_list = [
    "chap_07_standard_full_time_uip_regression_central_bank_policy_rates.txt",
    "chap_07_standard_full_time_uip_regression_3m_interbank_lending_rates.txt"
    ]
full_model_coefs_df = pd.DataFrame()
for file_name in file_name_list:
    # Create the full file path from file path and file name
    full_path_file = fr"{file_path}/{file_name}"
    df_coef = parse_summary_file(path=full_path_file, summary_file_type="statsmodels_coef_table")
    # Add the model file name as unique identifier
    df_coef['model_file_name'] = file_name
    full_model_coefs_df = pd.concat([full_model_coefs_df, df_coef], axis=0)

# Then read in the msm results for both specifications
file_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results"
file_name_list = [
    "chap_07_msm_central_bank_policy_rates_model_b1.txt",
    "chap_07_msm_3m_interbank_lending_rates_oil_gas_rol_vol_model_b2.txt",
    "chap_07_msm_central_bank_policy_rates_oil_gas_rol_vol_model_b1.txt",
    "chap_07_msm_3m_interbank_lending_rates_model_b2.txt"
    ]
full_model_coefs_markov_df = pd.DataFrame()
for file_name in file_name_list:
    # Create the full file path from file path and file name
    full_path_file = fr"{file_path}/{file_name}"
    df_coef = parse_summary_file(path=full_path_file, summary_file_type="markov_switching_coef_table")
    # Add the model file name as unique identifier
    df_coef['model_file_name'] = file_name
    full_model_coefs_markov_df = pd.concat([full_model_coefs_markov_df, df_coef], axis=0)
# Combine both tables for a final comparison
combined_model_coefs_df = pd.concat([full_model_coefs_df, full_model_coefs_markov_df], axis=0)
# Save the data locally
data_loading_instance.export_dataframe(
        df=combined_model_coefs_df,
        file_name="chap_04_combined_model_coefs_df",
        excel_sheet_name="Granger Test Results",
        excel_path=PRESENTATION_DATA,
        save_excel=True,
        save_index=True,
        )