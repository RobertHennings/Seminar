import pandas as pd
import numpy as np
import json
import datetime as dt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models"
FIGURES_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"

#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Basic Full time UIP Regression
#----------------------------------------------------------------------------------------
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
    get_periods_overlaying_df
# Import custom written model evaluation scores
from utils.evaluation_metrics import compute_rcm

data_graphing_instance = DataGraphing()
data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)
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
    interest_rate_diff_df[f'i_diff_{country}'] = central_bank_policy_rate_df['US_CBPR'] - central_bank_policy_rate_df[f'{country}_CBPR']
interest_rate_diff_df = interest_rate_diff_df.rename(
    columns={
        "i_diff_XM": "i_diff_EUR",
    })
# Now combine both datasets
uip_data_df = pd.concat([log_spot_rate_diff_df, interest_rate_diff_df], axis=1).dropna()

def run_uip_regression(dep_var: str, indep_var: str, data: pd.DataFrame):
    import statsmodels.api as sm
    X = sm.add_constant(data[indep_var])
    y = data[dep_var]
    model = sm.OLS(y, X).fit()
    return model

# First run the UIP regression on the full sample
full_sample_uip_results = {}
currency_pairs = ['EUR']
for currency in currency_pairs:
    dep_var = f'{currency}'
    indep_var = f'i_diff_{currency}'
    model = run_uip_regression(dep_var, indep_var, uip_data_df)
    full_sample_uip_results[currency] = model
# Print the summary of the full sample UIP regressions
for currency, model in full_sample_uip_results.items():
    print(f"Full Sample UIP Regression Results for {currency}/USD:")
    print(model.summary())
    print("\n")

# Observe the error terms with Durbin-Watson and Ljung-Box tests for autocorrelated error terms
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Standard
#----------------------------------------------------------------------------------------
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
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
#----------------------------------------------------------------------------------------
# 07 - Econometric Modelling - Benchmark Models for Regime Identification - Markov Switching Model - Oil, Gas added
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
spot_exchange_rate_data_df_log_diff = np.log(spot_exchange_rate_data_df).diff().dropna()




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

data_list = [
    spot_exchange_rate_data_df[spot_rate], # 1) Only the raw spot exchange rate itself
    spot_exchange_rate_data_df_log_diff[f"{spot_rate[0]}_log_diff"], # 2) The log first difference of the spot exchange rate
    exchange_rates_vola_df[f"{spot_rate[0]}_log_diff_rol_vol"], # 3) The rolling volatility of the log first difference of the spot exchange rate
    spot_exchange_rate_data_df, # 4) The raw oil and gas prices as external variables
    exchange_rates_vola_df, # 5) The rolling volatility of WTI Oil and Nat Gas as external variables
    log_diff_rolling_vola_df, # 6) The log first difference of the spot exchange rate EUR/US and rolling volatility of WTI Oil and Nat Gas as external variables
]
# Perform GridSearch like approach to find the best model and data combination
from sklearn.model_selection import ParameterGrid, GridSearchCV, TimeSeriesSplit
param_grids = {
    "KMeans": {
        "n_clusters": [2, 3, 4],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
        "max_iter": [300, 500],
        "tol": [1e-4, 1e-3],
        "random_state": [42],
        "algorithm": ["lloyd", "elkan"]
    },
    "AgglomerativeClustering": {
        "n_clusters": [2, 3, 4],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "l1", "l2", "manhattan", "cosine"],
        "compute_full_tree": ["auto", True, False],
        "distance_threshold": [None]
    },
    "DBSCAN": {
        "eps": [0.5, 1.0, 1.5, 2.0],
        "min_samples": [3, 5, 8, 10],
        "metric": ["euclidean", "manhattan", "cosine"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2]
    },
    "MeanShift": {
        "bandwidth": [None, 0.5, 1.0, 2.0],
        "seeds": [None],
        "bin_seeding": [True, False],
        "cluster_all": [True, False],
        "max_iter": [300, 500]
    },
    "MarkovRegression": {
        "k_regimes": [2, 3],
        "trend": ["n", "c", "ct", "t"],
        "switching_trend": [True, False],
        "switching_exog": [True, False],
        "switching_variance": [True, False],
        "em_iter": [10, 20, 30],
        "cov_type": ["diag", "unstructured"]
    },
    "GaussianMixture": {
        "n_components": [2, 3, 4],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "tol": [1e-3, 1e-4],
        "max_iter": [100, 200],
        "n_init": [1, 5, 10],
        "init_params": ["kmeans", "random"],
        "random_state": [42],
        "reg_covar": [1e-6, 1e-5]
    },
    "Birch": {
        "n_clusters": [None, 2, 3, 4],
        "threshold": [0.3, 0.5, 1.0, 1.5],
        "branching_factor": [25, 50, 100],
        "copy": [True, False],
        "compute_labels": [True, False]
    },
    "AffinityPropagation": {
        "damping": [0.5, 0.7, 0.9],
        "max_iter": [200, 500],
        "convergence_iter": [15, 30, 50],
        "preference": [-50, -10, None],
        "affinity": ["euclidean", "precomputed"],
        "verbose": [False]
    },
    "OPTICS": {
        "min_samples": [5, 10],
        "max_eps": [np.inf, 1.5, 2.0],
        "metric": ["euclidean", "manhattan", "cosine"],
        "p": [2],
        "xi": [0.05, 0.1, 0.2],
        "min_cluster_size": [0.05, 0.1],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40]
    },
    "MiniBatchKMeans": {
        "n_clusters": [2, 3, 4],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
        "batch_size": [50, 100, 200],
        "max_iter": [100, 300],
        "tol": [1e-4, 1e-3],
        "max_no_improvement": [5, 10],
        "init_size": [None, 100, 300],
        "random_state": [42],
        "reassignment_ratio": [0.01, 0.1]
    }
}
model_class = {
        "MarkovRegression": MarkovRegression,
        "KMeans": KMeans,
        "AgglomerativeClustering": AgglomerativeClustering,
        "DBSCAN": DBSCAN,
        "MeanShift": MeanShift,
        "GaussianMixture": GaussianMixture,
        "Birch": Birch,
        "AffinityPropagation": AffinityPropagation,
        "OPTICS": OPTICS,
        "MiniBatchKMeans": MiniBatchKMeans,
    }
results = []
for model_name, cls in model_class.items():
    print(f"Model: {model_name}, Class: {cls}")
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
        if model_name == "MarkovRegression":
            cls_instance = cls(
                endog=endog,
                exog=exog,
                k_regimes=N_REGIMES,
                trend='c',  # or 'nc' for no constant
                switching_trend=True,
                switching_exog=True,
                switching_variance=True,
            )
        else:
            cls_instance = cls()
        try:
            clf = GridSearchCV(estimator=cls_instance,
                            param_grid=param_grids[model_name]
                            )
            clf.fit(data)
            clf_results = clf.cv_results_
            results.append(clf_results)
        except Exception as e:
            print(f"Model {model_name} failed to fit: {e}")
            continue

len(results)
len(results[0])
# purged k-fold cross validation with embargoing

results = []
for model_name, param_grid in param_grids.items():
    grid = ParameterGrid(param_grid)
    model_class = {
        "KMeans": KMeans,
        "AgglomerativeClustering": AgglomerativeClustering,
        "DBSCAN": DBSCAN,
        "MeanShift": MeanShift,
        "MarkovRegression": MarkovRegression,
        "GaussianMixture": GaussianMixture,
        "Birch": Birch,
        "AffinityPropagation": AffinityPropagation,
        "OPTICS": OPTICS,
        "MiniBatchKMeans": MiniBatchKMeans,
    }[model_name]
    for params in grid:
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
            model = model_class(**params)
            model_object_instance = ModelObject()
            model_object_instance.set_model_object(model_object=model)
            model_object_instance.set_data(training_data=data, testing_data=data)
            try:
                model_object_instance.fit()
                predicted_labels = model_object_instance.predict()
                evaluation_score = model_object_instance.evaluate(metric_function_list=[silhouette_score])
                results.append({
                    "model_type": model_name,
                    "params": params,
                    "score": evaluation_score,
                    "labels": predicted_labels,
                    "endog": endog,
                    "exog": exog
                })
            except Exception as e:
                print(f"Model {model_name} with params {params} failed to fit: {e}")
                continue
scores_list = [item["score"] for item in results]
extracted_scores_list = []
for item in scores_list:
    try:
        score = item["silhouette_score"]
        extracted_scores_list.append(score)
    except Exception as e:
        extracted_scores_list.append(0)
        print(f"Error extracting score: {e}")

best_score = np.nanmax(extracted_scores_list)
best_index = np.argmax(extracted_scores_list)
# Pick the model with the index
best_result = results[best_index]
print("Best model:", best_result["model_type"], "Params:", best_result["params"], "Score:", best_result["score"])
best_result["endog"]
best_result["exog"]

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
        # Add more models as needed
    }
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
# all_models_comp_df.to_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/models/all_models_comparison_1.xlsx", index=False)

# Plot the model training results as a bar plot with the used features as index
unique_df = all_models_comp_df.drop_duplicates(subset=["model_type", "silhouette_score", "feature_names_in"]).dropna(subset=["silhouette_score"])

predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
for column in predicted_labels_df.columns:
    if predicted_labels_df[column].max() > 1 or predicted_labels_df[column].min() <0 or (predicted_labels_df[column].dropna() == 0.0).all():
        print(f"Removing column: {column}")
        predicted_labels_df = predicted_labels_df.drop(columns=[column])

keep_models = [model.replace(":", "-") for model in predicted_labels_df.columns]
keep_models = [model.replace(" ", "_") for model in keep_models]
exact_model = unique_df["model_type"] + "_" + unique_df["time_last_fitted"].str.replace(":", "-").str.replace(" ", "_")
unique_df["model_file_name"] = exact_model
unique_df = unique_df[unique_df["model_file_name"].isin(keep_models)]
title="Model comparison using the silhouette score (1 being best, 0 indicating overlapping clusters, -1 being worst) for various regime identification model configurations"

fig_model_comp_bar_plot = data_graphing_instance.get_model_comparison_bar_plot(
    data=unique_df,
    evaluation_score_col_name="silhouette_score",
    title="",
    x_axis_title="Feature names",
    y_axis_title="Silhouette score (1 best - 0 overlapping clusters - -1 worst)",
    color_mapping_dict={
        "KMeans": "black",
        "AgglomerativeClustering": "darkgrey",
        "DBSCAN": "blue",
        "MeanShift": "red",
        "MarkovRegression": "#9b0a7d",
        "GaussianMixture": "lightgrey",
        "Birch": "pink",
        "AffinityPropagation": "gray",
        "OPTICS": "olive",
        "MiniBatchKMeans": "#00677c"
    },
    save_fig=True,
    file_name="model_comparison_bar_plot",
    file_path=FIGURES_PATH,
    showlegend=True,
    textfont_size=8.5,
    width=1400,
    height=800,
    scale=3
    )
fig_model_comp_bar_plot.show(renderer="browser")
# Now also map the regimes across the models correctly, in that we assume the high vola regime is encoded as 1 and the low vola regime is a 0
# KMeans is already correctly encoded
# AgglomerativeClustering needs to be encoded
# MeanShift is also correctly encoded
# MarkovRegression needs to be encoded
predicted_labels_df = get_recoded_predicted_labels_df(
    predicted_labels_df=predicted_labels_df,
    label_mapping_dict={0: 1, 1: 0},
    column_names_list=["AgglomerativeClustering", "MarkovRegression"]
    )
# Compare the fitted in sample regimes
regime_counts_df = get_regime_counts_df(
    predicted_labels_df=predicted_labels_df
    )
# Also identify the time periods that are overlapping in the classification of the algorithms and count their share
predicted_labels_overlap_df = get_overlapping_regimes_df(
    predicted_labels_df=predicted_labels_df
    )
# Notice: Overlap is defined as two models having the exact same regime encoding so 1 == 1
# It might be that the regimes are just encoded differently, e.g. 0 == 1 and 1 == 0 for two models
# This is not captured in the above overlap calculation

# Plot them all together in a plotly graph, together with the target data
# Plot the regimes on a secondary y-axis
predicted_labels_df.index = spot_exchange_rate_data_df.index

# Now also load in the table of crisis periods and overlay these time periods as well
# to see if regime changes have been picked up
graphing_df = pd.concat(
    [spot_exchange_rate_data_df, predicted_labels_df],
    axis=1
)

with open("/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods_dict.json", "r") as f:
    crisis_periods_dict = json.load(f)

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(graphing_df.columns)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(graphing_df.columns.tolist())}

for column in graphing_df.columns:
    if not column == "EUR/USD":
        if graphing_df[column].max() > 1 or graphing_df[column].min() <0 or (graphing_df[column].dropna() == 0.0).all():
            print(f"Removing column: {column}")
            graphing_df = graphing_df.drop(columns=[column])

title=f"Resulting predicted model regimes for various variants with highlighted crisis periods over the time: {graphing_df.index[0].year} - {graphing_df.index[-1].year}",
fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=graphing_df,
    crisis_periods_dict=crisis_periods_dict,
    variables=["EUR/USD"],
    secondary_y_variables=graphing_df.columns[1:].tolist(),
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title="",
    secondary_y_axis_title="Predicted model regimes (N=2)",
    x_axis_title="Date",
    y_axis_title="Spot exchange rate EUR/USD",
    color_mapping_dict=color_mapping,
    num_years_interval_x_axis=5,
    showlegend=True,
    save_fig=False,
    file_name="predicted_model_regimes_with_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")
# evaluate numerically if the models were able to identify the crisis periods
# by checking how many of the crisis period data points were classified as high volatility regime
# by each model
crisis_periods_df = pd.DataFrame(crisis_periods_dict).T.reset_index().rename(columns={"index": "Crisis", "start": "Start-date", "end": "End-date"})
overlay_df = get_periods_overlaying_df(
    crisis_periods_df=crisis_periods_df,
    predicted_labels_df=predicted_labels_df,
    predicted_labels_df_column_names_list=predicted_labels_df.columns[1:].tolist(),
)
# Load a specific model/model object back - Here a MarkovRegression Model
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
# Extract the regime periods
#----------------------------------------------------------------------------------------
# 0X - Econometric Modelling - Models with exogenous Variables for Regime Identification
#----------------------------------------------------------------------------------------
# Now try to also include exogeneous variables like WTI Oil Price vola and Henry Hub gas Vola and see if something changes
series_dict_mapping = {
    'WTI Oil': 'DCOILWTICO',
    "Nat Gas": "DHHNGSP",
}

start_date = exchange_rates_vola_df.index.min().strftime('%Y-%m-%d')
end_date = exchange_rates_vola_df.index.max().strftime('%Y-%m-%d')

data_dict, data_full_info_dict, lowest_freq = data_loading_instance.get_fred_data(
    series_dict_mapping=series_dict_mapping,
    start_date=start_date,
    end_date=end_date
    )
data_us_df = pd.concat(list(data_dict.values()), axis=1).dropna()
data_us_full_info_table = pd.DataFrame(data_full_info_dict).T



# Relevel the data to have the same dimensions
data_us_df = data_us_df.reindex(exchange_rates_vola_df.index).dropna()
data_us_df["log_WTI_Oil"] = np.log(data_us_df["WTI Oil"])
data_us_df["log_Nat_Gas"] = np.log(data_us_df["Nat Gas"])
data_us_df["log_WTI_Oil_diff"] = data_us_df["log_WTI_Oil"].diff()
data_us_df["log_Nat_Gas_diff"] = data_us_df["log_Nat_Gas"].diff()
data_us_df = data_us_df.dropna()
# Calculate rolling volatilities
window = 30
data_us_df["rolling_vola_oil"] = data_us_df["log_WTI_Oil_diff"].rolling(window=window).std().dropna()
data_us_df["rolling_vola_gas"] = data_us_df["log_Nat_Gas_diff"].rolling(window=window).std().dropna()
data = pd.concat([exchange_rates_vola_df, data_us_df], axis=1).dropna()

# Now apply the very same loop from earlier, but now with the exogeneous variable included
# We initialize a new instance of the ModelObject class for each model with desired settings
# We store them in a list that we will loop through
msm = MarkovRegression(
    endog=data[["rolling_vola"]],
    exog=data[['rolling_vola_oil', 'rolling_vola_gas']],
    k_regimes=N_REGIMES,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
) # <- remember to always create a new msm object if new data is used because else the dimensions will not align due to different data
models_list = [kmeans, agg, dbscan, ms, msm]
for model in models_list:
    # 1) Initialize a new instance for each model class
    model_object_instance = ModelObject() # Initialize a new instance for each model
    # 2) Set the model
    model_object_instance.set_model_object(model_object=model)
    # 3) Set the data - Here we only want an In-sample comparison
    # therefore train and test data are the same
    model_object_instance.set_data(
        training_data=data[["rolling_vola", "rolling_vola_oil", "rolling_vola_gas"]],
        testing_data=data[["rolling_vola", "rolling_vola_oil", "rolling_vola_gas"]]
    )
    # 4) Fit the model
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
    # First save the fitted model object itself
    model_object_instance.save_model(
        file_format=r".pkl",
        file_path=MODELS_PATH,
        model_file_name=file_name
        )
    # Second save all the related metadata/information about the model
    model_object_instance.get_full_model_info(
        save=True,
        return_info_dict=False,
        file_format=r".json",
        file_path=MODELS_PATH,
        file_name=file_name
        )
#################################################################################
# Again reevaluate
# For a proper evaluation now, loading all full_info files and compare the evaluation scores
all_models_comp_df = get_model_metadata_df(
    full_model_info_path=MODELS_PATH,
    )
all_models_comp_df[["model_type", "silhouette_score", "feature_names_in"]].sort_values(by="silhouette_score", ascending=False)
predicted_labels_df = extract_predicted_labels_from_metadata_df(
    metadata_df=all_models_comp_df,
)
# Before we can export the predicted labels for each model, we need to make sure that
# the regimes are encoded the same way across the models
# For that first plot all the regimes over time
exchange_rates_vola_df = exchange_rates_vola_df.reindex(predicted_labels_df.index)
# Now also load in the table of crisis periods and overlay these time periods as well
# to see if regime changes have been picked up
graphing_df = pd.concat(
    [exchange_rates_vola_df, predicted_labels_df],
    axis=1
)


with open("/Users/Robert_Hennings/Uni/Master/Seminar/data/raw/crisis_periods_dict.json", "r") as f:
    crisis_periods_dict = json.load(f)

custom_color_scale = data_graphing_instance.create_custom_diverging_colorscale(
    start_hex="#9b0a7d",
    end_hex="black",
    center_color="grey",
    steps=round((len(graphing_df.columns)+1)/2),
    lightening_factor=0.8,
)
# Extract only the hex color codes from the created list
custom_color_scale_codes = [color[1] for color in custom_color_scale]
color_mapping = {var: custom_color_scale_codes[i % len(custom_color_scale_codes)] for i, var in enumerate(graphing_df.columns.tolist())}

# Ignore AffinityPropagation as it only identified one regime
graphing_df = graphing_df.drop(columns=[
    "AffinityPropagation_2025-10-09 16:24:55",
    "OPTICS_2025-10-09 16:25:21",
    "MeanShift_2025-10-09 16:01:39",
    "MeanShift_2025-10-09 15:22:35"
    ])

fig_crisis_periods_highlighted = data_graphing_instance.get_fig_crisis_periods_highlighted(
    data=graphing_df,
    crisis_periods_dict=crisis_periods_dict,
    variables=["rolling_vola"],
    secondary_y_variables=graphing_df.columns[1:].tolist(),
    recession_shading_color="rgba(155, 10, 125, 0.3)",
    title=f"Exchange Rate and Oil Volatility with Crisis Periods highlighted over the time: {graphing_df.index[0].year} - {graphing_df.index[-1].year}",
    secondary_y_axis_title="WTI Oil & Natural Gas Volatility",
    x_axis_title="Date",
    y_axis_title="Exchange Rate Volatility",
    color_mapping_dict=color_mapping,
    num_years_interval_x_axis=5,
    showlegend=True,
    save_fig=False,
    file_name="exchange_rate_oil_raw_vola_crisis_periods_highlighted",
    file_path=FIGURES_PATH,
    width=1200,
    height=800,
    scale=3
    )
# Show the figure
fig_crisis_periods_highlighted.show(renderer="browser")

column_names_list = [
    "AgglomerativeClustering_2025-10-09 15:19:59",
    "MarkovRegression_2025-10-09 12:58:53",
    "MarkovRegression_2025-10-09 15:22:41",
    "MarkovRegression_2025-10-09 16:17:48",
    "AgglomerativeClustering_2025-10-09 12:56:28",
    "AgglomerativeClustering_2025-10-09 16:15:43",
    "KMeans_2025-10-09 15:19:53"
                     ]
graphing_df = get_recoded_predicted_labels_df(
    predicted_labels_df=graphing_df,
    label_mapping_dict={0: 1, 1: 0},
    column_names_list=column_names_list
    )

# Exporting the predicted labels dataframe to an excel file
data_loading_instance.export_dataframe(
        df=graphing_df[graphing_df.columns[1:]],
        file_name="predicted_labels_df",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results",
        save_excel=True,
        save_index=True,
        )
# Plotting the results as a plotly bar plot


#################################################################################
model_object_instance = ModelObject()
msm = MarkovRegression(
    endog=data["rolling_vola"],
    exog=data[['rolling_vola_oil']],
    k_regimes=N_REGIMES,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=False,
)
model_object_instance.set_model_object(model_object=msm)
model_object_instance.set_data(
    training_data=data[['rolling_vola']], # see comment above
    testing_data=data[['rolling_vola']] # see comment above
    )
model_object_instance.fit(em_iter=10, search_reps=20)
predicted_labels = model_object_instance.predict()
model_object_instance.evaluate(metric_function_list=[silhouette_score])
model_object_instance.fitted_model.summary()
# Get the smoothed probabilities for each regime
smoothed_probabilities = model_object_instance.fitted_model.smoothed_marginal_probabilities
data["msm_regime"] = smoothed_probabilities[1]
data["regime"] = (data["msm_regime"] >= 0.5).astype(int)
# Data Points per regime
data["regime"].value_counts()
rcm_value = compute_rcm(S=2, smoothed_probs=data["msm_regime"])
print(f"Regime Classification Measure (RCM): {rcm_value}")

regime_0_periods = data[data["regime"] == 0].index
regime_1_periods = data[data["regime"] == 1].index
print(f"Regime 0 periods: {regime_0_periods}")
print(f"Regime 1 periods: {regime_1_periods}")

# Plot the regimes together with the exchange rate changes as plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["rolling_vola"],
                         mode='lines', name='Rolling Volatility'))
fig.add_trace(go.Scatter(x=data.index, y=data["msm_regime"],
                         mode='lines', name='MSM Regime Probability', yaxis='y2'))
fig.update_layout(
    title='Markov Switching Model Regimes with Rolling Volatility',
    xaxis_title='Date',
    yaxis_title='Log US NEER Diff',
    yaxis2=dict(
        title='MSM Regime Probability',
        overlaying='y',
        side='right'    
    ),
    legend=dict(x=0, y=1)
)
fig.show(renderer="browser")
#################################################################################
model_object_instance = ModelObject()
model_object_instance.set_model_object(model_object=kmeans)
model_object_instance.set_data(
    training_data=data[['rolling_vola']], # see comment above
    testing_data=data[['rolling_vola']] # see comment above
    )
model_object_instance.fit()
# Try to replicate the example from the Lecture Notes
# Page 3/16 Modeling Nonlinearities I: Markov-Switching
# A model incorporating GARCH effects in the return equation has been introduced by Engle, Lilien and Robins (1987)
# The so-called GARCH-M model:
# s_t - s_t-1 = beta_0 + beta_1 * (i_t-1 - i*_t-1) + beta_2 * h_t + u_t
# u_t = epsilon_t * sqrt(h_t)
# h_t = c + alpha * epsilon_t-1^2 + beta * h_t-1
# Of course, we expect beta_0 = 0, beta_1 = 1 and beta_2 > 0!
# The parameter estimates are calculated by numerical maximization of
# the log likelihood
# Estimate the model
import statsmodels.api as sm

msm_garch = sm.tsa.MarkovAutoregression(
    endog=exchange_rates_df["log_US_NEER_diff"],
    k_regimes=2,
    trend='c',  # or 'nc' for no constant
    switching_trend=True,
    switching_exog=True,
    switching_variance=True,
    order=1,
    switching_ar=True,
    garch_order=(1, 1)
)
msm_garch_fitted = msm_garch.fit(em_iter=10, search_reps=20)
print(msm_garch_fitted.summary())
# Get the smoothed probabilities for each regime
exchange_rates_df["msm_garch_regime"] = msm_garch_fitted.smoothed_marginal_probabilities[1]
# Plot the smoothed probabilities
exchange_rates_df["msm_garch_regime"]
# Evaluate the RCM for the Markov Switching GARCH model
rcm_garch_value = compute_rcm(S=2, smoothed_probs=exchange_rates_df["msm_garch_regime"])
print(f"Regime Classification Measure (RCM) for GARCH model: {rcm_garch_value}")
# Plot the regimes together with the exchange rate changes as plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=exchange_rates_df.index, y=exchange_rates_df["log_US_NEER_diff"],
                         mode='lines', name='Log US NEER Diff'))
fig.add_trace(go.Scatter(x=exchange_rates_df.index, y=exchange_rates_df["msm_garch_regime"],
                         mode='lines', name='MSM GARCH Regime Probability', yaxis='y2'))
fig.update_layout(
    title='Markov Switching GARCH Model Regimes',
    xaxis_title='Date',
    yaxis_title='Log US NEER Diff',
    yaxis2=dict(
        title='MSM GARCH Regime Probability',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1)
)
fig.show(renderer="browser")

########################################################################               
# Alternatively we could use the Central Bank Policy Rates from the BIS as well on a daily level
# Source: https://data.bis.org/search?q=central+bank+policy+rates&page_size=100&filter=FREQ%3DD
country_keys_mapping = {
    "US": "United States",
    "XM": "Euro area",
    "GB": "United Kingdom",
    "JP": "Japan",
    "CH": "Switzerland",
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
    'GBP/USD': 'DEXUSUK',
    'JPY/USD': 'DEXJPUS',
    'CHF/USD': 'DEXSZUS',
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
        "i_diff_GB": "i_diff_GBP",
        "i_diff_JP": "i_diff_JPY",
        "i_diff_CH": "i_diff_CHF"
    })
# Now combine both datasets
uip_data_df = pd.concat([log_spot_rate_diff_df, interest_rate_diff_df], axis=1).dropna()

def run_uip_regression(dep_var: str, indep_var: str, data: pd.DataFrame):
    import statsmodels.api as sm
    X = sm.add_constant(data[indep_var])
    y = data[dep_var]
    model = sm.OLS(y, X).fit()
    return model

# First run the UIP regression on the full sample
full_sample_uip_results = {}
currency_pairs = ['EUR', 'GBP', 'JPY', 'CHF']
for currency in currency_pairs:
    dep_var = f'{currency}'
    indep_var = f'i_diff_{currency}'
    model = run_uip_regression(dep_var, indep_var, uip_data_df)
    full_sample_uip_results[currency] = model
# Print the summary of the full sample UIP regressions
for currency, model in full_sample_uip_results.items():
    print(f"Full Sample UIP Regression Results for {currency}/USD:")
    print(model.summary())
    print("\n")

# Now pull in the identified regimes and among them test again for the UIP condition
predicted_labels_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/data/results/predicted_labels_df.xlsx", index_col=0)

# Relevel the predicted labels df to have the same index as the UIP data df
predicted_labels_df = predicted_labels_df.reindex(uip_data_df.index).dropna()
uip_data_df = uip_data_df.reindex(predicted_labels_df.index).dropna()
# Now run the UIP regression for each identified regime
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
for model_name, regimes in regime_uip_results.items():
    for regime, currencies in regimes.items():
        for currency, model in currencies.items():
            print(f"UIP Regression Results for {currency}/USD in {model_name} Regime {regime}:")
            print(model.summary())
            print("\n")

