# Seminar
# Identifying (volatility-)regimes in the the EUR/USD spot exchange rate using clustering algorithms: An Oil and Gas Perspective on Parity Conditions.
Seminar in Applied Financial Economics: Applied Econometrics of FX Markets - Professor Reitz

[PDF Presentation](https://github.com/RobertHennings/Seminar/blob/main/reports/presentation_latex_version/main.pdf)

**NOTE: The in the presentation included graphs can be viewed as interactive, online html version by just clicking on them, or by just completing the url in a new browser tab: https://roberthennings.github.io/Seminar/graph_name.html**

All graphs can be accessed in the respective subfolder in the reports section.

## Installation
1. Either click on download in the GitHub UI
2. Open a terminal -> Navigate to the desired local folder path where to save the files (cd some/path/to/store) -> enter: git clone https://github.com/RobertHennings/Seminar 
<br>
The full terminal code:

```zsh
cd some/path/to/store
git clone https://github.com/RobertHennings/Seminar 
```

## The seminar project
### Abstract
The seminar paper deals with the question, if alternative regime identification methods, including additional explanatory variables in the framework of clustering algorithms, can help to better identify (volatility driven) market regimes in the EUR/USD spot exchange rate. Especially including different variations of energy commodity prices (and their rolling volatility) is tested for an improved regime identification performance, with variations of the Markov-Switching Model
as benchmark. Finally, after having separated the Time-series of the EUR/USD spot exchange rate into different (volatility-) regimes, the standard UIP relationship is tested and compared among the regimes.

Main considered energy commodity prices are (USA):

1. [WTI Crue Oil - Cushing, Oklahoma - Spot Prices - Daily](https://fred.stlouisfed.org/series/DCOILWTICO)
2. [Natural Gas - Henry Hub - Spot Prices - Daily](https://fred.stlouisfed.org/series/DHHNGSP)

Main considered interest rate data (USA, EU Area):

1. [Central Bank Policy Rate (CBPR) - Daily - USA - Bank for International Settlements](https://data.bis.org/topics/CBPOL/BIS,WS_CBPOL,1.0/D.US)
2. [Central Bank Policy Rate (CBPR) - Daily - EU Area - Bank for International Settlements](https://data.bis.org/topics/CBPOL/BIS,WS_CBPOL,1.0/D.XM)

Data sources are primarily:

1. [FRED](https://fred.stlouisfed.org)
2. [ECB](https://data.ecb.europa.eu)
3. [BIS](https://data.bis.org/topics/CBPOL)
4. [Our World in Data](https://ourworldindata.org/energy#explore-data-on-energy)

<br>
<u>Main Research Hypothesis:</u>
<br>
1. “The standard UIP-equilibrium condition is time-dependent and primarily controlled by two main regimes, characterised by either high or low (market-) volatility.”
<br>
<u>Aditional Research Hypothesis I:</u>
2. “Monetary policy, i.e. interest rates are, partly driven by energy commodity prices that induce volatility through the inﬂationary pass-through channel, especially during phases of market distress in economies heavily relying on import/export of energy commodities.”
<br>
<u>Aditional Research Hypothesis II:</u>
<br>
3. “Factoring in variables related to energy commodity prices in combination with using alternative clustering techniques improves the identiﬁcation of the regimes to better pinpoint the time-dependent testing of the standard UIP relation, compared to Markov-Switching benchmark models.”
<br>
<br>
In order to test the research hypotheses a two stage procedure is folowed:
<br>

1. First, for various in Sci-Kit Learn implemented clustering algorithms, the optimal hyperparameter configuration is optimised using {}, and as evaluation metric the soilhouette score is considered, for every dataset variant.

2. Then, with these optimised hyperparameters, the regimes are identified (0: low volatility, 1: high volatility), with the respective clustering algorithms and compared against Markov-Switching benchmark models. Within each identified regime, for each algorithm, the standard UIP relationship is tested for.

## Organizational Information
Tips for academic work at the QBER-Kiel: [Link](https://alexanderklos.github.io/py_dos_and_donts/index_py_dos_and_donts.html)

### Fromalities
Formal guidelines for the $\LaTeX$ format offered by the CAU can be found [here](https://www.wiso.uni-kiel.de/de/studium/sonstiges/info-inst-vwl/abschlussarbeiten).


## Repository Structure
```zsh
Seminar
├── CITATION.cff <- Citation file allowing for quick citation of the repo
├── LICENSE
├── README.md
├── data <- folder containing all used data
│   ├── raw
│   └── results
│       ├── chap_00_exchange_rates_ppp_deviations.xlsx
│       ├── chap_01_daily_exchange_rate_oil_log_diff_vola_normalized_crisis_periods_highlighted.xlsx
│       ├── chap_01_eu_inflation_contribution_data.xlsx
│       ├── chap_01_us_inflation_contribution_data.xlsx
│       ├── chap_04_all_models_comp_df.xlsx
│       ├── chap_04_combined_model_coefs_df.xlsx
│       ├── chap_04_predicted_labels_df.xlsx
│       ├── chap_04_uip_identified_regimes_results_df.xlsx
│       ├── chap_04_unique_df_full.xlsx
│       ├── chap_06_adf_test_log_diff.xlsx
│       ├── chap_06_adf_test_raw_series.xlsx
│       ├── chap_06_cointegration_test_log_diff.xlsx
│       ├── chap_06_cointegration_test_raw_series.xlsx
│       ├── chap_06_gas_consumption_data.xlsx
│       ├── chap_06_gas_production_data.xlsx
│       ├── chap_06_granger_causality_test_gas_log_diff.xlsx
│       ├── chap_06_granger_causality_test_gas_raw_series.xlsx
│       ├── chap_06_granger_causality_test_oil_log_diff.xlsx
│       ├── chap_06_granger_causality_test_oil_raw_series.xlsx
│       ├── chap_06_interest_rate_comparison_df.xlsx
│       ├── chap_06_norm_test_log_diff.xlsx
│       ├── chap_06_norm_test_raw_series.xlsx
│       ├── chap_06_normed_histogram_data_log_first_differences.xlsx
│       ├── chap_06_normed_histogram_data_log_first_differences_rolling_volatility.xlsx
│       ├── chap_06_oil_consumption_data.xlsx
│       ├── chap_06_oil_production_data.xlsx
│       ├── chap_06_prices_oi_df.xlsx
│       ├── chap_06_spot_exchange_rate_data_df.xlsx
│       └── crisis_periods_dict.json
├── docs
├── literature <- folder containing all used literature
├── notebooks <- folder containing the final submission Notebooks
│   ├── Seminar_Fella_Hennings_Graphs_Presentation.ipynb
│   ├── Seminar_Fella_Hennings_Modelling_Benchmark.ipynb
│   └── Seminar_Fella_Hennings_Modelling_Clustering.ipynb
├── reports <- folder contining all report types accompanying the project
│   ├── presentation_latex_version <- The LaTex  presentation
│   │   ├── chapters <- single chapter files pulled together in main.tex
│   │   │   ├── abbreviations.tex
│   │   │   ├── acknowledgements.tex
│   │   │   ├── appendix.tex
│   │   │   ├── chapter-00.tex
│   │   │   ├── chapter-01.tex
│   │   │   ├── chapter-02.tex
│   │   │   ├── chapter-03.tex
│   │   │   ├── chapter-04.tex
│   │   │   ├── chapter-05.tex
│   │   │   ├── chapter-06.tex
│   │   │   ├── chapter-07.tex
│   │   │   ├── chapter-08.tex
│   │   │   ├── closing-discussion.tex
│   │   │   ├── closing.tex
│   │   │   ├── further-material-questions.tex
│   │   │   ├── list-of-figures.tex
│   │   │   ├── list-of-tables.tex
│   │   │   └── references.tex
│   │   ├── data <- all data used to produce the graphs in the presentation
│   │   │   ├── chap_00_exchange_rates_ppp_deviations.xlsx
│   │   │   ├── chap_01_daily_exchange_rate_oil_log_diff_vola_normalized_crisis_periods_highlighted.xlsx
│   │   │   ├── chap_01_eu_inflation_contribution_data.xlsx
│   │   │   ├── chap_01_us_inflation_contribution_data.xlsx
│   │   │   ├── chap_04_all_models_comp_df.xlsx
│   │   │   ├── chap_04_combined_model_coefs_df.xlsx
│   │   │   ├── chap_04_model_input_data_list.xlsx
│   │   │   ├── chap_04_open_interest_trading_volume_oil_gas_reuters.xlsx
│   │   │   ├── chap_04_predicted_labels_df.xlsx
│   │   │   ├── chap_04_uip_data_df.xlsx
│   │   │   ├── chap_04_uip_data_df_3m_interbank_lending_rates.xlsx
│   │   │   ├── chap_04_uip_data_df_3m_interbank_lending_rates_b2.xlsx
│   │   │   ├── chap_04_uip_data_df_3m_interbank_lending_rates_oil_gas_rol_vol_model_b2.xlsx
│   │   │   ├── chap_04_uip_data_df_central_bank_policy_rates_oil_gas_rol_vol_b1.xlsx
│   │   │   ├── chap_04_uip_identified_regimes_results_df.xlsx
│   │   │   ├── chap_04_unique_df_full.xlsx
│   │   │   ├── chap_06_adf_test_log_diff.xlsx
│   │   │   ├── chap_06_adf_test_raw_series.xlsx
│   │   │   ├── chap_06_cointegration_test_log_diff.xlsx
│   │   │   ├── chap_06_cointegration_test_raw_series.xlsx
│   │   │   ├── chap_06_gas_consumption_data.xlsx
│   │   │   ├── chap_06_gas_production_data.xlsx
│   │   │   ├── chap_06_granger_causality_test_gas_log_diff.xlsx
│   │   │   ├── chap_06_granger_causality_test_gas_raw_series.xlsx
│   │   │   ├── chap_06_granger_causality_test_oil_log_diff.xlsx
│   │   │   ├── chap_06_granger_causality_test_oil_raw_series.xlsx
│   │   │   ├── chap_06_interest_rate_comparison_df.xlsx
│   │   │   ├── chap_06_norm_test_log_diff.xlsx
│   │   │   ├── chap_06_norm_test_raw_series.xlsx
│   │   │   ├── chap_06_normed_histogram_data_log_first_differences.xlsx
│   │   │   ├── chap_06_normed_histogram_data_log_first_differences_rolling_volatility.xlsx
│   │   │   ├── chap_06_oil_consumption_data.xlsx
│   │   │   ├── chap_06_oil_production_data.xlsx
│   │   │   ├── chap_06_prices_oi_df.xlsx
│   │   │   ├── chap_06_spot_exchange_rate_data_df.xlsx
│   │   │   └── crisis_periods_dict.json
│   │   ├── figures <- all included graphs as .pdf version
│   │   │   ├── chap_00_deviations_of_usd_spotrates_from_ppp_values.pdf
│   │   │   ├── chap_02_eu_area_cpi_inflation_decomposition.pdf
│   │   │   ├── chap_02_exchange_rate_oil_raw_vola_normalized_crisis_periods_highlighted.pdf
│   │   │   ├── chap_02_us_cpi_inflation_decomposition.pdf
│   │   │   ├── chap_03_literature_systematic_overview.pdf
│   │   │   ├── chap_04_theo_framework_prices_measures.pdf
│   │   │   ├── chap_04_theo_framework_simple_model.pdf
│   │   │   ├── chap_04_theo_framework_theoretical_framework_I.pdf
│   │   │   ├── chap_04_theo_framework_theoretical_framework_II.pdf
│   │   │   ├── chap_04_theo_framework_theoretical_framework_III.pdf
│   │   │   ├── chap_04_theo_framework_theoretical_framework_IV.pdf
│   │   │   ├── chap_04_theo_framework_theoretical_framework_full.pdf
│   │   │   ├── chap_06_acf_plot_log_diff.pdf
│   │   │   ├── chap_06_acf_plot_raw_series.pdf
│   │   │   ├── chap_06_granger_causality_test_gas_log_diff.pdf
│   │   │   ├── chap_06_granger_causality_test_gas_raw_series.pdf
│   │   │   ├── chap_06_granger_causality_test_oil_log_diff.pdf
│   │   │   ├── chap_06_granger_causality_test_oil_raw_series.pdf
│   │   │   ├── chap_06_interest_rate_comparison_bis_cbpr_vs_3m_interbank.pdf
│   │   │   ├── chap_06_interest_rate_comparison_bis_cbpr_vs_3m_interbank_diffs.pdf
│   │   │   ├── chap_06_log_first_diff_histogram.pdf
│   │   │   ├── chap_06_pacf_plot_log_diff.pdf
│   │   │   ├── chap_06_pacf_plot_raw_series.pdf
│   │   │   ├── chap_06_raw_data_normalized_histogram.pdf
│   │   │   ├── chap_06_weekly_open_interest_oil_gas_combined_graph.pdf
│   │   │   ├── chap_06_yearly_gas_consumption_production_combined_graph.pdf
│   │   │   ├── chap_06_yearly_oil_consumption_production_combined_graph.pdf
│   │   │   ├── chap_07_model_comparison_bar_plot.pdf
│   │   │   ├── chap_07_predicted_model_regimes_rel_share_overlap.pdf
│   │   │   ├── chap_07_predicted_model_regimes_rel_share_overlap_theo_crisis_regimes.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted_I.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted_II.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted_III.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted_IV.pdf
│   │   │   ├── chap_07_predicted_model_regimes_with_crisis_periods_highlighted_V.pdf
│   │   │   ├── chap_07_uip_estimation_benchmark_models.pdf
│   │   │   ├── chap_07_uip_estimation_identified_regimes.pdf
│   │   │   ├── siegel.pdf
│   │   │   └── wiso_logo.png
│   │   ├── main.pdf <- the presentation document itself
│   │   ├── main.sty <- the presentation style file
│   │   ├── main.tex <- the presentation .tex main file
│   │   ├── references.bib <- the presentation references.bib file holding all references
│   │   ├── table_of_contents.tex
│   │   └── titlepage.tex
│   ├── presentation_pptx_version
│   │   └── PPTX_WiSo_Template.pptx <- the .pptx presentation with some needed schematic graphs
│   └── tables
│       ├── chap_06_adf_test_log_diff.tex
│       ├── chap_06_adf_test_raw_series.tex
│       ├── chap_06_cointegration_test_log_diff.tex
│       ├── chap_06_cointegration_test_raw_series.tex
│       ├── chap_06_norm_test_log_diff.tex
│       ├── chap_06_norm_test_raw_series.tex
│       ├── crisis_periods.tex
│       ├── crisis_periods.txt
│       └── crisis_periods.xlsx
├── requirements
│   └── requirements.txt <- the needed (python) libraries to run all the code
└── src <-all the seminar code
    └── seminar_code
        ├── data_graphing
        │   ├── config.py
        │   ├── data_grapher.py
        │   └── seminar_graphs.py
        ├── data_loading
        │   ├── config.py
        │   ├── data_loader.py
        │   └── seminar_data.py
        ├── model
        │   ├── architecture.py
        │   ├── model.py
        │   └── model_benchmark.py
        ├── model_optimisation
        │   ├── config.py
        │   ├── model_optimisation.py
        │   └── model_optimiser.py
        ├── models
        │   ├── AffinityPropagation_2025-10-15_17-06-20.json
        │   ├── AffinityPropagation_2025-10-15_17-06-20.pkl
        │   ├── AffinityPropagation_2025-10-15_17-10-02.json
        │   ├── AffinityPropagation_2025-10-15_17-10-02.pkl
        │   ├── AffinityPropagation_2025-10-15_17-14-15.json
        │   ├── AffinityPropagation_2025-10-15_17-14-15.pkl
        │   ├── AffinityPropagation_2025-10-15_17-19-17.json
        │   ├── AffinityPropagation_2025-10-15_17-19-17.pkl
        │   ├── AffinityPropagation_2025-10-15_17-22-32.json
        │   ├── AffinityPropagation_2025-10-15_17-22-32.pkl
        │   ├── AffinityPropagation_2025-10-15_20-49-20.json
        │   ├── AffinityPropagation_2025-10-15_20-49-20.pkl
        │   ├── AgglomerativeClustering_2025-10-08_19-37-45.json
        │   ├── AgglomerativeClustering_2025-10-08_19-37-45.pkl
        │   ├── AgglomerativeClustering_2025-10-15_17-03-27.json
        │   ├── AgglomerativeClustering_2025-10-15_17-03-27.pkl
        │   ├── AgglomerativeClustering_2025-10-15_17-06-41.json
        │   ├── AgglomerativeClustering_2025-10-15_17-06-41.pkl
        │   ├── AgglomerativeClustering_2025-10-15_17-10-32.json
        │   ├── AgglomerativeClustering_2025-10-15_17-10-32.pkl
        │   ├── AgglomerativeClustering_2025-10-15_17-14-46.json
        │   ├── AgglomerativeClustering_2025-10-15_17-14-46.pkl
        │   ├── AgglomerativeClustering_2025-10-15_17-19-41.json
        │   ├── AgglomerativeClustering_2025-10-15_17-19-41.pkl
        │   ├── AgglomerativeClustering_2025-10-15_20-46-37.json
        │   ├── AgglomerativeClustering_2025-10-15_20-46-37.pkl
        │   ├── Birch_2025-10-15_17-04-14.json
        │   ├── Birch_2025-10-15_17-04-14.pkl
        │   ├── Birch_2025-10-15_17-07-37.json
        │   ├── Birch_2025-10-15_17-07-37.pkl
        │   ├── Birch_2025-10-15_17-12-07.json
        │   ├── Birch_2025-10-15_17-12-07.pkl
        │   ├── Birch_2025-10-15_17-17-05.json
        │   ├── Birch_2025-10-15_17-17-05.pkl
        │   ├── Birch_2025-10-15_17-20-24.json
        │   ├── Birch_2025-10-15_17-20-24.pkl
        │   ├── Birch_2025-10-15_20-47-18.json
        │   ├── Birch_2025-10-15_20-47-18.pkl
        │   ├── DBSCAN_2025-10-08_19-37-49.json
        │   ├── DBSCAN_2025-10-08_19-37-49.pkl
        │   ├── DBSCAN_2025-10-15_17-03-29.json
        │   ├── DBSCAN_2025-10-15_17-03-29.pkl
        │   ├── DBSCAN_2025-10-15_17-06-42.json
        │   ├── DBSCAN_2025-10-15_17-06-42.pkl
        │   ├── DBSCAN_2025-10-15_17-10-34.json
        │   ├── DBSCAN_2025-10-15_17-10-34.pkl
        │   ├── DBSCAN_2025-10-15_17-14-47.json
        │   ├── DBSCAN_2025-10-15_17-14-47.pkl
        │   ├── DBSCAN_2025-10-15_17-19-43.json
        │   ├── DBSCAN_2025-10-15_17-19-43.pkl
        │   ├── DBSCAN_2025-10-15_20-46-38.json
        │   ├── DBSCAN_2025-10-15_20-46-38.pkl
        │   ├── GaussianMixture_2025-10-15_17-04-13.json
        │   ├── GaussianMixture_2025-10-15_17-04-13.pkl
        │   ├── GaussianMixture_2025-10-15_17-07-37.json
        │   ├── GaussianMixture_2025-10-15_17-07-37.pkl
        │   ├── GaussianMixture_2025-10-15_17-12-07.json
        │   ├── GaussianMixture_2025-10-15_17-12-07.pkl
        │   ├── GaussianMixture_2025-10-15_17-17-04.json
        │   ├── GaussianMixture_2025-10-15_17-17-04.pkl
        │   ├── GaussianMixture_2025-10-15_17-20-24.json
        │   ├── GaussianMixture_2025-10-15_17-20-24.pkl
        │   ├── GaussianMixture_2025-10-15_20-47-18.json
        │   ├── GaussianMixture_2025-10-15_20-47-18.pkl
        │   ├── KMeans_2025-10-08_19-37-34.json
        │   ├── KMeans_2025-10-08_19-37-34.pkl
        │   ├── KMeans_2025-10-15_17-03-22.json
        │   ├── KMeans_2025-10-15_17-03-22.pkl
        │   ├── KMeans_2025-10-15_17-06-38.json
        │   ├── KMeans_2025-10-15_17-06-38.pkl
        │   ├── KMeans_2025-10-15_17-10-29.json
        │   ├── KMeans_2025-10-15_17-10-29.pkl
        │   ├── KMeans_2025-10-15_17-14-43.json
        │   ├── KMeans_2025-10-15_17-14-43.pkl
        │   ├── KMeans_2025-10-15_17-19-38.json
        │   ├── KMeans_2025-10-15_17-19-38.pkl
        │   ├── KMeans_2025-10-15_20-46-33.json
        │   ├── KMeans_2025-10-15_20-46-33.pkl
        │   ├── MarkovRegression_2025-10-08_19-39-55.json
        │   ├── MarkovRegression_2025-10-08_19-39-55.pkl
        │   ├── MarkovRegression_2025-10-15_17-04-12.json
        │   ├── MarkovRegression_2025-10-15_17-04-12.pkl
        │   ├── MarkovRegression_2025-10-15_17-07-35.json
        │   ├── MarkovRegression_2025-10-15_17-07-35.pkl
        │   ├── MarkovRegression_2025-10-15_17-12-06.json
        │   ├── MarkovRegression_2025-10-15_17-12-06.pkl
        │   ├── MarkovRegression_2025-10-15_17-17-03.json
        │   ├── MarkovRegression_2025-10-15_17-17-03.pkl
        │   ├── MarkovRegression_2025-10-15_17-20-22.json
        │   ├── MarkovRegression_2025-10-15_17-20-22.pkl
        │   ├── MarkovRegression_2025-10-15_20-47-17.json
        │   ├── MarkovRegression_2025-10-15_20-47-17.pkl
        │   ├── MeanShift_2025-10-08_19-39-48.json
        │   ├── MeanShift_2025-10-08_19-39-48.pkl
        │   ├── MeanShift_2025-10-15_17-04-10.json
        │   ├── MeanShift_2025-10-15_17-04-10.pkl
        │   ├── MeanShift_2025-10-15_17-07-32.json
        │   ├── MeanShift_2025-10-15_17-07-32.pkl
        │   ├── MeanShift_2025-10-15_17-12-03.json
        │   ├── MeanShift_2025-10-15_17-12-03.pkl
        │   ├── MeanShift_2025-10-15_17-16-58.json
        │   ├── MeanShift_2025-10-15_17-16-58.pkl
        │   ├── MeanShift_2025-10-15_17-20-18.json
        │   ├── MeanShift_2025-10-15_17-20-18.pkl
        │   ├── MeanShift_2025-10-15_20-47-12.json
        │   ├── MeanShift_2025-10-15_20-47-12.pkl
        │   ├── MiniBatchKMeans_2025-10-15_17-06-37.json
        │   ├── MiniBatchKMeans_2025-10-15_17-06-37.pkl
        │   ├── MiniBatchKMeans_2025-10-15_17-10-28.json
        │   ├── MiniBatchKMeans_2025-10-15_17-10-28.pkl
        │   ├── MiniBatchKMeans_2025-10-15_17-14-42.json
        │   ├── MiniBatchKMeans_2025-10-15_17-14-42.pkl
        │   ├── MiniBatchKMeans_2025-10-15_17-19-37.json
        │   ├── MiniBatchKMeans_2025-10-15_17-19-37.pkl
        │   ├── MiniBatchKMeans_2025-10-15_17-22-49.json
        │   ├── MiniBatchKMeans_2025-10-15_17-22-49.pkl
        │   ├── MiniBatchKMeans_2025-10-15_20-49-39.json
        │   ├── MiniBatchKMeans_2025-10-15_20-49-39.pkl
        │   ├── OPTICS_2025-10-15_17-06-36.json
        │   ├── OPTICS_2025-10-15_17-06-36.pkl
        │   ├── OPTICS_2025-10-15_17-10-28.json
        │   ├── OPTICS_2025-10-15_17-10-28.pkl
        │   ├── OPTICS_2025-10-15_17-14-42.json
        │   ├── OPTICS_2025-10-15_17-14-42.pkl
        │   ├── OPTICS_2025-10-15_17-19-36.json
        │   ├── OPTICS_2025-10-15_17-19-36.pkl
        │   ├── OPTICS_2025-10-15_17-22-48.json
        │   ├── OPTICS_2025-10-15_17-22-48.pkl
        │   ├── OPTICS_2025-10-15_20-49-38.json
        │   └── OPTICS_2025-10-15_20-49-38.pkl
        ├── optimisation_results
        └── utils
            ├── config.py
            ├── email_notification.py
            ├── evaluation.py
            └── evaluation_metrics.py
```

<u>**seminar** subfolder:</u>
<br>
The ($ \LaTeX $) Thesis project can be found in the subfolder called "seminar".
</br>
<u>**reports** subfolder:</u>
<br>
The "reports" subfolder contains everything related to the final seminar presentation ($ \LaTeX $). Itself has the subfolders "presentation_latex_version", that holds the final $ \LaTeX $ presentation with its subfolders "chapters", "figures" and "code". In the folder "presentation_pptx_version", an accompanying PPTX presentation is available that includes flow and schema figures. There are also the folders "figures", "tables" and "logs" that house all material, whereas in the "presentation_" folders there are only the figures and tables included that are actually used in the documents.
</br>
<u>**src** subfolder:</u>
<br>
The "src" subfolder contains everything related to the code of the seminar. Itself has two subfolders: "seminar_code" and "tests". The first one is the most important one, whereas itself holds all the main components for the seminar content: "data_processing", "model", "utils", "data_loading", "models", "data_graphing".
</br>

## Running the seminar code locally in a scripting file (.py)
To run the seminar code locally (i.e. after cloning the repository), one can just simply create a virtual environment. See the dedetailed documentation [here](https://docs.python.org/3/library/venv.html)

Depending on your python version, open a terminal window, move to the desired loaction via `cd` and create a new virtual environment.

If the interested user wants to reproduce the results of the seminar project, there are two main steps that need to be taken care of before trying to execute code:
1. Installing the correct Python Version
2. Setting up a virtual environment and loading all necessary libraries/packages in the correct version

A quick guide on how to achieve these pre-requirements is provided below:
### Creating a virtual environment
<br><strong>ON MAC</strong></br>
Python < 3:
```zsh
python -m venv name_of_your_virtual_environment
```
Or provide the full path directly:
```zsh
python -m venv /path/to/new/virtual/name_of_your_virtual_environment
```
Python >3:
```zsh
python3 -m venv name_of_your_virtual_environment
```
Or provide the full path directly:
```zsh
python3 -m venv /path/to/new/virtual/name_of_your_virtual_environment
```
### Activating a virtual environment
Activate the virtual environment by:
```zsh
source /path/to/new/virtual/name_of_your_virtual_environment/bin/activate
```
or move into the virtual environment directly and execute:
```zsh
source /bin/activate
```
### Deactivating a virtual environment
Deactivate the virtual environment from anywhere via:
```zsh
deactivate
```
### Downloading dependencies inside the virtual environment
Move to the virtual environment or create a new one, activate it and install the dependencies from the requirements.txt file via:
```zsh
pip install -r ./requirements/requirements.txt
```
or:
```zsh
pip3 install -r ./requirements/requirements.txt
```
Alternatively by providing the full path to the requirements.txt file:
```zsh
pip3 install -r /Users/path/to/project/requirements.txt
```
Make sure the dependencies were correctly loaded:
```zsh
pip list
```
or:
```zsh
pip3 list
```
## The Package 

## Citation
### BibTex
```bibtex
@misc{FellaHenningsSeminar2025,
  author        = {Fella, Josef; Hennings, Robert},
  title         = {Seminar},
  year          = {2025},
  version       = {0.0.1},
  author        = {Fella, Josef; Hennings, Robert},
  license       = {MIT},
  title         = {Identifying (volatility-)regimes in the the EUR/USD spot exchange rate using clustering algorithms: An Oil and Gas Perspective on Parity Conditions.},
  url           = {https://github.com/RobertHennings/Seminar},
  note          = {Submitted: 10.11.2025, Presentation held: 14.11.2025, Seminar project at the chair of economics, Prof. Dr. Stefan Reitz; QBER - Kiel}
}
```
### APA
```apa
Fella, J.; Hennings, R. Seminar (Version 1.0.0) [Computer software]. https://github.com/RobertHennings/Seminar
```
## Contributing
### Commit Style
Please also consider writting meaningful messages in your commits.
```zsh
API: an (incompatible) API change
BENCH: changes to the benchmark suite
BLD: change related to building numpy
BUG: bug fix
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
REL: related to releasing numpy
```

## Authors
Josef Fella, 2025
<br>
Robert Hennings, 2025