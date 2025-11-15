import datetime as dt
import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans

# from sklearn.cluster import DBSCAN
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import OPTICS
# from sklearn.cluster import Birch

SEMINAR_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar"
SEMINAR_CODE_PATH = rf"{SEMINAR_PATH}/src/seminar_code"
MODELS_PATH = rf"{SEMINAR_CODE_PATH}/models"
FIGURES_PATH = rf"{SEMINAR_PATH}/reports/figures"

print(os.getcwd())
os.chdir(SEMINAR_CODE_PATH)
print(os.getcwd())

from data_loading.data_loader import DataLoading
from model_optimisation import config as cfg
from model_optimisation.model_optimiser import ModelOptimiser
from grid_search.grid_searching_strategies import PurgedTimeSeriesSplit, SkfolioWrapper

# Import custom written model evaluation scores
from utils.evaluation_metrics import silhouette_scorer
from utils.evaluation import extract_best_params, results_list_to_df

param_grids_dict = cfg.PARAM_GRIDS

data_loading_instance = DataLoading(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
)


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
# Perform GridSearch like approach to find the best model and data combination
model_class_dict = {
        "KMeans": KMeans,
        # "AgglomerativeClustering": AgglomerativeClustering,
        # # "DBSCAN": DBSCAN,
        # "MeanShift": MeanShift,
        # "GaussianMixture": GaussianMixture,
        # # "Birch": Birch,
        # "AffinityPropagation": AffinityPropagation,
        # # "OPTICS": OPTICS,
        # "MiniBatchKMeans": MiniBatchKMeans,
    }
scoring_dict = {
    "silhouette": silhouette_scorer
}
# purged_cv = PurgedTimeSeriesSplit(n_splits=5, gap=5)  # 5-period gap
purged_cv = SkfolioWrapper(
    n_folds=5,
    n_test_folds=2,
    embargo_size=30, # Match your rolling window
    purged_size=0.05
)
# Only use standard K-Fold if variable is almost iid or errors are iid - then the idea is that the data is randomly distributed
# Else use Time order respecting CV like PurgedTimeSeriesSplit or Skfolio's CombinatorialPurgedCV
model_optimiser_instance = ModelOptimiser(
    data_list=data_list,
    param_grids_dict=param_grids_dict,
    model_class_dict=model_class_dict,
    scoring_dict=scoring_dict,
)
results_list = model_optimiser_instance.run_grid_search_cv(
        grid_search_refit_scorer="silhouette",
        grid_search_return_train_score=True,
        grid_search_cv=purged_cv,
        # grid_search_n_jobs_outer=10,
        )

# Extract best parameters for all results
best_params_list = extract_best_params(
    results_list=results_list,
    score_name="silhouette"
)
timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
best_params_df = pd.DataFrame(best_params_list)
data_loading_instance.export_dataframe(
        df=best_params_df,
        file_name="best_params_df",
        excel_sheet_name=rf"best_params_{timestamp_str}",
        excel_path=r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/optimisation_results",
        save_excel=True,
        save_index=False,
        )
best_params_df[["feature_names_in", "model_name", "best_score"]].sort_values(by="best_score", ascending=False)
best_params_df[["model_name", "best_score"]].sort_values(by="best_score", ascending=False)
# Save the results of the grid search CV as a json file for later parsing
model_optimiser_instance.save_grid_search_results(
    results=best_params_list,
    file_path="/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/optimisation_results",
    file_name=f"grid_search_results_{timestamp_str}"
    )
# Read in the optimised hyperparameters from the grid search
model_optimiser_instance = ModelOptimiser(
    data_list=[],
    param_grids_dict={},
    model_class_dict={},
    scoring_dict={},
)
results_list_read_in = model_optimiser_instance.load_grid_search_results(
    file_path="/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/optimisation_results",
    file_name=f"grid_search_results_20251109_121829"
    )

# usage
df = results_list_to_df(results_list_read_in)
df[["model_name", "best_score"]].sort_values(by="best_score", ascending=False)
df[["feature_names_in", "model_name", "best_score"]].sort_values(by="best_score", ascending=False)

best_params_df = pd.read_excel(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/optimisation_results/best_params_df.xlsx")