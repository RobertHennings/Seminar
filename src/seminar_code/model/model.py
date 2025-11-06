import os
import json
import datetime as dt
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
os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code")
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
# 07 - Econometric Modelling - Own Models for Regime Identification
#----------------------------------------------------------------------------------------
# Recreate the Markov-Switching model from the paper: The impact of oil shocks on exchange rates: A Markov-switching approach
# Two regimes assumed: High volatility and low volatility regime, page 14
# Dependent variable: first difference of the log real exchange rate for country i, page 14
# The Markov-switching models for exchange rates were estimated using the fMarkovSwitching package in R (Perlin, 2008).
# The models were estimated with two states, state dependent regression coefficients and state dependent volatility for
# the error process. Exchange rates are known to exhibit volatility clustering which is why we allow volatility to vary across regimes. Models were estimated using two different as- sumptions about the error term (normal, Student-t).
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import MiniBatchKMeans
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import silhouette_score

# Read in the optimised hyperparameters from the grid search
# optimised_fit_kwargs_dict = {}
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
spot_exchange_rate_data_df_log_diff = np.log(spot_exchange_rate_data_df).diff().dropna()
# Iterate through different data that shall be used
# Next after the benchmark has been trained on just the Spot rate itself, we try the rolling vola of the log first differences
window = 30
exchange_rates_vola_df = spot_exchange_rate_data_df_log_diff.rolling(window=window).std().dropna()

N_REGIMES = 2
spot_rate = ["EUR/USD"]

# Rename the data columns accordingly
spot_exchange_rate_data_df_log_diff.columns = [f"{col}_log_diff" for col in spot_exchange_rate_data_df_log_diff.columns]
exchange_rates_vola_df.columns = [f"{col}_log_diff_rol_vol" for col in exchange_rates_vola_df.columns]
log_diff_rolling_vola_df = pd.concat([spot_exchange_rate_data_df_log_diff, exchange_rates_vola_df], axis=1).dropna().drop(columns=["EUR/USD_log_diff_rol_vol", "WTI Oil_log_diff", "Nat Gas_log_diff"])

# Add the trading volume as proxy for degree of financial market integration
oi_tv_oil_gas_reuters_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/open_interest_trading_volume_oil_gas_reuters.xlsx")
oi_tv_oil_gas_reuters_df["Date"] = pd.to_datetime(oi_tv_oil_gas_reuters_df["Date"], format="%Y-%m-%d")
oi_tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df.set_index("Date", drop=True)
# Separate out the Open-Interest
oi_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "open interest" in col.lower()]
tv_columns = [col for col in oi_tv_oil_gas_reuters_df.columns if "volume" in col.lower()]
oi_tv_oil_gas_reuters_df[tv_columns].dropna()

tv_oil_gas_reuters_df = oi_tv_oil_gas_reuters_df[tv_columns].dropna()

log_diff_rolling_vola_df_tv_added = pd.concat([log_diff_rolling_vola_df, tv_oil_gas_reuters_df], axis=1).dropna()
##### Apply outlier detection

data_list = [
    spot_exchange_rate_data_df[spot_rate], # 1) Only the raw spot exchange rate itself
    spot_exchange_rate_data_df_log_diff[f"{spot_rate[0]}_log_diff"], # 2) The log first difference of the spot exchange rate
    exchange_rates_vola_df[f"{spot_rate[0]}_log_diff_rol_vol"], # 3) The rolling volatility of the log first difference of the spot exchange rate
    spot_exchange_rate_data_df, # 4) The raw oil and gas prices as external variables
    exchange_rates_vola_df, # 5) The rolling volatility of WTI Oil and Nat Gas as external variables
    log_diff_rolling_vola_df, # 6) The log first difference of the spot exchange rate EUR/US and rolling volatility of WTI Oil and Nat Gas as external variables
    log_diff_rolling_vola_df_tv_added, # 7) The log first difference of the spot exchange rate EUR/US, rolling volatility of WTI Oil and Nat Gas plus trading volume of oil and gas as external variables
]
# Ensure that every data item in the data_list os of type pd.DataFrame
for i, data in enumerate(data_list):
    if type(data) == pd.Series:
        data_list[i] = data.to_frame()

for data in data_list:
    if type(data) == pd.Series:
        data = data.to_frame()
    endog_col_name = [col for col in data.columns if spot_rate[0] in col][0]
    endog = data[endog_col_name]
    # First determine if the input data has multiple features or is univariate
    if data.shape[1] > 1: # we have external variables
        exog_col_names = [col for col in data.columns if spot_rate[0] not in col]
        exog = data[exog_col_names]
    else:
        exog = None
    print(f"endog:\n{endog}\nexog: {exog}")
    # Set up the models
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42)
    agg = AgglomerativeClustering(n_clusters=N_REGIMES)
    dbscan = DBSCAN(eps=1.5, min_samples=8)
    # spectral = SpectralClustering(n_clusters=n_regimes, affinity='nearest_neighbors', random_state=42)
    ms = MeanShift()
    msm = MarkovRegression(
        endog=endog,
        exog=exog,
        k_regimes=N_REGIMES,
        trend='c',  # or 'nc' for no constant
        switching_trend=True,
        switching_exog=True,
        switching_variance=True,
    )
    gmm = GaussianMixture(n_components=N_REGIMES, random_state=42)
    birch = Birch(n_clusters=N_REGIMES)
    affinity = AffinityPropagation()
    optics = OPTICS()
    minibatch_kmeans = MiniBatchKMeans(n_clusters=N_REGIMES, random_state=42)
    # In the below dict, additional fit kwargs for each model class can be specified
    # that will be passed to the fit() method of the respective model class
    fit_kwargs_dict = {
        "KMeans": {"n_init": 10},
        "AgglomerativeClustering": {},
        "DBSCAN": {},
        "MeanShift": {},
        "MarkovRegression": {"em_iter": 10, "search_reps": 20},
        "MarkovAutoregression": {"em_iter": 10, "search_reps": 20},
    }
    # fit_kwargs_dict = optimised_fit_kwargs_dict
    # We store them in a list that we will loop through
    models_list = [kmeans, agg, dbscan, ms, msm, gmm, birch, affinity, optics, minibatch_kmeans]
    for model in models_list:
        # 1) Initialize a new instance for each model class
        model_object_instance = ModelObject() # Initialize a new instance for each model
        # 2) Set the model
        model_object_instance.set_model_object(model_object=model)
        # 3) Set the data - Here we only want an In-sample comparison
        # therefore train and test data are the same
        model_object_instance.set_data(
            training_data=data,
            testing_data=data
        )
        # 4) Fit the model
        # based on the model class name extract additional parameters for the fit
        model_name = model.__class__.__name__
        fit_kwargs = fit_kwargs_dict.get(model_name, {})
        model_object_instance.fit(**fit_kwargs)
        # 5) Predict the labels - In sample forecast
        predicted_labels = model_object_instance.predict()
        # 6) Evaluate the model - pick the desired score to evaluate
        # Here we could also think of providing a list with multiple functions at once
        evaluation_score = model_object_instance.evaluate(metric_function_list=[silhouette_score])
        # Save the model and the model info with dynamic names based on the model class name
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{model.__class__.__name__}_{timestamp}"
        # First save the fitted model object itself to a .pkl file - we can also read it in later
        model_object_instance.save_model(
            file_format=r".pkl",
            file_path=MODELS_PATH,
            model_file_name=file_name
            )
        # Second save all the related metadata/information about the model - from here the relevant metadata will be pulled
        model_object_instance.get_full_model_info(
            save=True,
            return_info_dict=False,
            file_format=r".json",
            file_path=MODELS_PATH,
            file_name=file_name
            )
# For a proper evaluation now, loading all full_info.json files and compare the evaluation scores
all_models_comp_df = get_model_metadata_df(
    full_model_info_path=MODELS_PATH,
    )
all_models_comp_df[["model_type", "silhouette_score", "feature_names_in"]].sort_values(by="silhouette_score", ascending=False)
# Save the data locally
data_loading_instance.export_dataframe(
    df=all_models_comp_df,
    file_name="chap_04_all_models_comp_df",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=False,
)
# Plot the model training results as a bar plot with the used features as index
unique_df = all_models_comp_df.drop_duplicates(subset=["model_type", "silhouette_score", "feature_names_in"]).dropna(subset=["silhouette_score"])

predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
data_loading_instance.export_dataframe(
    df=predicted_labels_df,
    file_name="chap_04_predicted_labels_df",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=True,
)
#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Exemplary loading of a fitted and saved model
#----------------------------------------------------------------------------------------
new_model_object_instance = ModelObject()
fitted_model, model_info_dict, model_info_df = new_model_object_instance.load_model(
    file_format_model=".pkl",
    file_path_model=MODELS_PATH,
    model_file_name="MarkovRegression_2025-10-08_22-26-05",
    return_info_dict=True,
    return_info_df=True,
    file_format_model_info=".json",
    file_path_model_info=MODELS_PATH,
    model_file_name_info="MarkovRegression_2025-10-08_22-26-05"
    )
fitted_model.predict(start=0, end=len(model_info_dict.get("testing_data_dates"))-1)
#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Models with exogenous Variables for Regime Identification
#----------------------------------------------------------------------------------------
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
    'EUR/USD': 'DEXUSEU',
}

start_date = central_bank_policy_rate_df.index.min().strftime('%Y-%m-%d')
end_date = central_bank_policy_rate_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_spot_rates_df = pd.concat(list(data_dict.values()), axis=1).dropna()
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
    interest_rate_diff_df[f'i_diff_{country}'] = central_bank_policy_rate_df[f'US_CBPR'] - central_bank_policy_rate_df[f'{country}_CBPR']
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

# Now pull in the identified regimes and among them test again for the UIP condition
predicted_labels_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results/predicted_labels_df.xlsx", index_col=0)

# Relevel the predicted labels df to have the same index as the UIP data df
predicted_labels_df = predicted_labels_df.reindex(uip_data_df.index).dropna()
uip_data_df = uip_data_df.reindex(predicted_labels_df.index).dropna()
# Now run the UIP regression for each identified regime
currency_pairs = ["EUR"]
regime_uip_results = {}
for model_name in predicted_labels_df.columns:
    regime_uip_results[model_name] = {}
    for regime in predicted_labels_df[model_name].unique():
        regime_data = uip_data_df[predicted_labels_df[model_name] == regime]
        regime_uip_results[model_name][regime] = {}
        for currency in currency_pairs:
            dep_var = f'{currency}'
            indep_var = f'i_diff_{currency}'
            if len(regime_data) < 10:  # Skip regimes with too few data points
                continue
            model = run_uip_regression(dep_var, indep_var, regime_data)
            regime_uip_results[model_name][regime][currency] = model
# Print the summary of the UIP regressions for each regime
# Extract the estimated coefficients and save them in a master table along the model name and regime
uip_identified_regimes_results_list = []
for model_name, regimes in regime_uip_results.items():
    for regime, currencies in regimes.items():
        for currency, model in currencies.items():
            estimated_params_df = pd.DataFrame(model.summary().tables[1].data)
            estimated_params_df.columns = estimated_params_df.iloc[0]
            estimated_params_df = estimated_params_df[1:]
            estimated_params_df.columns = ["param"] + estimated_params_df.columns[1:].tolist()
            # Transfer all columns to numeric where possible
            estimated_params_df = estimated_params_df.apply(pd.to_numeric, errors='ignore')
            estimated_params_df["model_name"] = model_name
            estimated_params_df["regime"] = regime
            uip_identified_regimes_results_list.append(estimated_params_df)
# Save the results
uip_identified_regimes_results_df = pd.concat(uip_identified_regimes_results_list, axis=0).reset_index(drop=True)
# Disentangle the confidence upper and lower columns
uip_identified_regimes_results_df = uip_identified_regimes_results_df.rename(
    columns={
        "[0.025": "ci_lower",
        "0.975]": "ci_upper",
    })
# Save the data locally
data_loading_instance.export_dataframe(
    df=uip_identified_regimes_results_df,
    file_name="chap_04_uip_identified_regimes_results_df",
    excel_sheet_name="04",
    excel_path=PRESENTATION_DATA,
    save_excel=True,
    save_index=False,
)
