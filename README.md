# Seminar
Seminar in Applied Financial Economics: Applied Econometrics of FX Markets - Professor Reitz

## Installation
1. Either click on download in the GitHub UI
2. Open a terminal -> Navigate to the desired local folder path where to save the files (cd some/path/to/store) -> enter: git clone https://github.com/RobertHennings/Seminar 
<br>
The full terminal code:

```zsh
cd some/path/to/store
git clone https://github.com/RobertHennings/Seminar 
```

## The seminar paper
### Abstract
The seminar paper deals with the question if and if so how heavily energy commodity price shocks affect exchange rates. The main research hypothesis are the following ones:
<br>
<u>Main Research Hypothesis:</u>
1. Exchange Rates and energy commodity prices are interconnected over several time frequencies and horizons, predominantly during times of (financial market) distress. Energy commodity price shocks primarily enter through the inflation dynamics channel, influencing both short-term price levels and long-term inflation expectations, thereby also affecting monetary policy decisions.
<br>
<u>Aditional Research Hypothesis I:</u>
2. “The pass-through effect of energy commodity price shocks to overall inflation is asymmetric, non-linear and time-varying, with price increases having a more pronounced effect than price decreases.”
<br>
<u>Aditional Research Hypothesis II:</u>
3. “The pass-through effect intensified with growing financialization of energy commodity markets, leading to stronger correlations between

In order to test the research hypotheses a two stage procedure is folowed:
1. First, since the relationship is highly time-frequency dependent and time-varying, several techniques are compared to identify major regimes in the data. For this purpose the following benchmark models have been set up: Markov-Switching, STR, STAR. Additionally these approaches are compared with alternative techniques like clustering algorithms that use external variables to identify regime switches.
2. After the main regimes in the data generating process have been identified, inference is carried out using ...., finally comparing the in-regime metrics.


## Organizational Information
Notion link: [Notion](https://www.notion.so/Seminar-25e94b1118e9809ebb1bf9c73503b7c8?source=copy_link)

Tips for academic work at the QBER-Kiel: [Link](https://alexanderklos.github.io/py_dos_and_donts/index_py_dos_and_donts.html)

### Fromalities
Formal guidelines for the $\LaTeX$ format offered by the CAU can be found [here](https://www.wiso.uni-kiel.de/de/studium/sonstiges/info-inst-vwl/abschlussarbeiten).



### To Do
- [ ] Read through papers and identify all major global crisis periods as these will be the theoretical regimes that modles should be able to identify

#### Chapter 00
Intro: The PPP puzzle and Commodity Currencies
- [ ] Find source for figure
#### Chapter 01
Oil: Global Production and Consumption over time
- [ ] Find source for figure, add data reference appropriately
Gas: Global Production and Consumption over time
- [ ] Find source for figure, add data reference appropriately
Financial Markets: Oil and Gas OI over time
- [ ] Find source for figure, add data reference appropriately
#### Chapter 02
Energy Price Contributions to Inflation - USA
- [ ] Find source for figure, add data reference appropriately
Energy Price Contributions to Inflation - EU area
- [ ] Find source for figure, add data reference appropriately
Formulated Research Hypothesis
- [ ] Sharpen and reduce the research hypotheses
#### Chapter 03
Systematic Literature Overview: Main Approaches
- [ ] Finalize the schematic figure in PPTX and include, find source
#### Chapter 04
Definitions - prices and measurements
- [ ] Find source and include the single formulae in Appendix/Definitions
A simple model of exchange rates and commodity
prices
- [ ] Finalize the schematic figure in PPTX and include, find source, include the derived proxy variables into modelling
Theoretical Framework (I)
- [ ] Finalize the schematic figure in PPTX and include, find source, include the derived proxy variables into modelling
Theoretical Framework (II)
- [ ] Finalize the schematic figure in PPTX and include, find source, include the derived proxy variables into modelling
#### Chapter 05
Systematic Methodology Overview: Linear vs.
Non-linear approaches
- [ ] Finalize the schematic figure in PPTX and include, find source
(Financial) Market Distress: Important periods and
their characteristics
- [ ] Finalize the schematic figure in PPTX and include, find source
#### Chapter 06
Spot exchange rate distributions (raw data -
normalized)
- [ ] Find source for figure, add data reference appropriately
Spot exchange rate distributions (log first differences)
- [ ] Find source for figure, add data reference appropriately, edit the x-axis title (shorten) for better fit
Tests for Normality (raw data)
- [ ] Find source, add test reference, add Jaque-Berra Test
Tests for Stationarity - ADF Tests (raw data)
- [ ] Find source, add test reference
Tests for Stationarity - ADF Tests (log first differences)
- [ ] Find source, add test reference
Tests for Cointegration (raw data)
- [ ] Find source, add test reference
Tests for Cointegration (log differences)
- [ ] Find source, add test reference
Tests for Autocorrelation (raw data)
- [ ] Find source, add test reference
Tests for Autocorrelation (log first differences)
- [ ] Find source, add test reference
Tests for Partial Autocorrelation (raw data)
- [ ] Find source, add test reference
Tests for Partial Autocorrelation (log first differences)
- [ ] Find source, add test reference
Granger Causality Tests - EUR/USD and oil (raw data)
- [ ] Find source, add test reference
Granger Causality Tests - EUR/USD and gas (raw
data)
- [ ] Find source, add test reference
Granger Causality Tests - EUR/USD and oil (log first
differences)
- [ ] Find source, add test reference
Granger Causality Tests - EUR/USD and gas (log first
differences)
- [ ] Find source, add test reference
#### Chapter 07
- [ ] Set up benchmark models to identify the regimes like the commonly used Markov-Switching and/or STR/STAR models
- [ ] Data: Try to get the Interbank Lending rates 3M EUR/USD for the interest rate differentials, also include a proxy for financialisation -> Pull the VET Open Interest and Trading Volume and factor this into the Model Optimisation for finding clusters
- [ ] Train these benchmark models and compare against our versions
- [ ] After regimes have been identified properly, compare all the in-regime characteristics using the classical econometric approaches
#### Chapter 08

#### Appendix - Figures and Tables
- [ ] move figures, tables here if necessary
#### Appendix - Data
- [ ] move figures, tables here if necessary
#### Appendix - Definitions
- [ ] define all used metrics, scores, tests, algorithms, exchange rate types here for clarifications, find sources
#### References - Literature
- [ ] add all used literature here
#### References - Data
- [ ] add all used data sources here
#### List of Figures
- [ ] add all used figures here, use the automated label logic for self referencing with page number
#### List of Tables
- [ ] add all used tables here, use the automated label logic for self referencing with page number
#### Further Material for Illustrations - Questions
- [ ] prepare material for questions/further illustrations
#### Else
- [X] Clearly identify what data/variables have been used so far (what exact exchange rate and why?, what driving variables?)

## Repository Structure
```zsh
Seminar
├── CITATION.cff <- Citation file allowing for quick citation of the repo
├── LICENSE
├── MANIFEST.in
├── README.md
├── data <- folder containing all used data
│   ├── features
│   ├── processed
│   ├── raw
│   └── results
├── docs
│   └── core
├── literature <- folder containing all used literature
│   ├── A
│   │   ├── CAN EXCHANGE RATES FORECAST COMMODITY PRICES.pdf
│   │   ├── COMMODITY PRICE VOLATILITY ACROSS EXCHANGE RATE REGIMES.pdf
│   │   ├── Commodity currencies revisited The role of global commodity price uncertainty.pdf
│   │   ├── Commodity prices, interest rates and the dollar.pdf
│   │   ├── Commodity-currencies or currency-commodities Evidence from causality tests.pdf
│   │   ├── Common factors of commodity prices.pdf
│   │   ├── Crude oil price and exchange rate Evidence from the period before and after the launch of China’s crude oil futures.pdf
│   │   ├── Econometric Analysis of the effect of energy prices on xchange rates during war periods.pdf
│   │   ├── Empirical Analysis of Crude Oil Price Effects on Exchange Rate Volatility.pdf
│   │   ├── Energy price shocks, exchange rates and inflation nexus.pdf
│   │   ├── Energy prices and the real exchange rate of commodity-exporting countries.pdf
│   │   ├── Exchange rate pass-through and inflation targeting regime under energy price shocks.pdf
│   │   ├── Exploring exchange rate sensitivity to crude oil futures A study of selected global economies.pdf
│   │   ├── Foreign exchange market and crude oil market- volatility spillover analysis.pdf
│   │   ├── Have oil and gas prices got separated.pdf
│   │   ├── Identifying structural changes and associations in exchange rates with Markov switching models The evidence from Central European currency markets.pdf
│   │   ├── Oil price shocks and inflation rate persistence A Fractional Cointegration VAR approach.pdf
│   │   ├── Spillover effect of US dollar exchange rate on oil prices.pdf
│   │   ├── The effect of energy price shocks on commodity currencies during the war in Ukraine.pdf
│   │   ├── The empirical relationship between energy futures prices and exchange rates.pdf
│   │   ├── The oil price-inflation nexus The exchange rate pass- through effect.pdf
│   │   ├── Trends and contagion in WTI and Brent crude oil spot and futures markets - The role of OPEC in the last decade.pdf
│   │   └── Unveiling the dynamic linkages between energy, forex and financial markets amidst natural and man-made outbreaks.pdf
│   ├── A+
│   │   ├── Asymmetric and nonlinear pass-through of crude oil prices to gasoline and natural gas prices.pdf
│   │   ├── Connectedness Between Natural Gas Price and BRICS Exchange Rates Evidence from Time and Frequency Domains.pdf
│   │   ├── Dependence and extreme dependence of crude oil and natural gas prices with applications to risk management.pdf
│   │   ├── Does exchange rate management affect the causality between exchange rates and oil prices- Evidence from oil-exporting countries.pdf
│   │   ├── Dynamic connectedness of oil price shocks and exchange rates.pdf
│   │   ├── Dynamic frequency relationships and volatility spillovers in natural gas, crude oil, gas oil, gasoline, and heating oil markets Implications for portfolio management.pdf
│   │   ├── Inflation, oil prices and exchange rates. The Euro’s dampening effect.pdf
│   │   ├── Long-run equilibrium real exchange rates and oil prices.pdf
│   │   ├── Oil price fluctuations and U.S. dollar exchange rates.pdf
│   │   ├── Oil price shocks and exchange rate movements.pdf
│   │   ├── Oil prices and real exchange rates.pdf
│   │   ├── Oil prices and the rise and fall of the US real exchange rate.pdf
│   │   ├── The Relationship between Crude Oil and Natural Gas Prices The Role of the Exchange Rate.pdf
│   │   ├── The power play of natural gas and crude oil in the move towards the financialization of the energy market.pdf
│   │   └── Towards a solution to the puzzles in exchange rate economics where do we stand .pdf
│   ├── B
│   │   ├── Are electricity prices affected by the US dollar to Euro exchange rate The Spanish case.pdf
│   │   ├── Commodity currencies and currency commodities.pdf
│   │   ├── Crude oil, forex, and stock markets unveiling the higher-order moment and cross-moment risk spillovers in times of turmoil.pdf
│   │   ├── Energy markets crucial relationship between prices.pdf
│   │   ├── Exchange Rates, Energy Prices, Unemployment, Money, and Inflation A Further Test.pdf
│   │   ├── Oil Prices and Inflation Expectations.pdf
│   │   ├── Oil price pass-through into inflation.pdf
│   │   └── Uncertainty about interest rates and crude oil prices.pdf
│   ├── C
│   │   ├── Asymmetric effect of oil prices on stock market prices New evidence from oil-exporting and oil-importing countries.pdf
│   │   └── Detecting regime switches in the dependence structure of high dimensional financial data.pdf
│   ├── Reitz Paper
│   │   ├── Are oil price forecasters finally right Regressive expectations toward more fundamental values of the oil price.pdf
│   │   ├── Commodity price cycles and heterogeneous speculators A STAR-GARCH model.pdf
│   │   ├── Non Linear Oil Price Dynamics - A Tale of Heterogeneous Speculators.pdf
│   │   └── Non-Linear Oil Price Dynamics A Tale of Heterogeneous Speculators.pdf
│   ├── basis
│   │   ├── Commodity Futures Prices Some Evidence on Forecast Power, Premiums, and the Theory of Storage.pdf
│   │   ├── Commodity Price Effects on Currencies.pdf
│   │   ├── DEVIATIONS FROM COVERED INTEREST RATE PARITY.pdf
│   │   ├── Do all oil price shocks have the same impact- Evidence from the euro area.pdf
│   │   ├── Exchange-rate return predictability and the adaptive markets hypothesis Evidence from major foreign exchange rates.pdf
│   │   ├── Impact of US Tight Oil on NYMEX WTI Futures.pdf
│   │   ├── THE UNCOVERED RETURN PARITY CONDITION.pdf
│   │   ├── Testing persistence of WTI and Brent long-run relationship after the shale oil supply shock.pdf
│   │   ├── The Handbook of Energy Trading.pdf
│   │   ├── Time varying market efficiency in the Brent and WTI crude market.pdf
│   │   ├── Uncovered Interest-Rate Parity over the Past Two Centuries.pdf
│   │   ├── Uncovered Interst parity at distant horizons evidence on emerging  ECONOMIES & NONLINEARITIES.pdf
│   │   ├── Virtual Barrels Quantitative Trading in the Oil Market.pdf
│   │   └── What drives commodity price variation.pdf
│   ├── good_overviews
│   │   └── The validity of interest parity in times of crisis.pdf
│   ├── main_sources
│   │   ├── Commodity currencies and the real exchange rate.pdf
│   │   ├── Commodity currencies.pdf
│   │   ├── Energy prices and exchange rates of the U.S. dollar Further evidence from linear andnonlinear causality analysis.pdf
│   │   ├── Financial stability risks from energy derivatives markets.pdf
│   │   ├── How Exchange Rate Volatility Shapes Commodity Derivatives Market Lessons from Five Global Shocks (2007–2023).pdf
│   │   ├── International energy trade and inflation dynamics The role of invoicing currency use during the low carbon transition.pdf
│   │   ├── Non-linear relationship between real commodity price volatility and real effective exchange rate The case of commodity-exporting.pdf
│   │   ├── Oil prices, exchange rates and asset prices.pdf
│   │   ├── Regime switches in exchange rate volatility and uncovered interest parity.pdf
│   │   ├── Regime switching in foreign exchange rates Evidence from currency option prices.pdf
│   │   ├── The impact of oil shocks on exchange rates A Markov-switching approach.pdf
│   │   ├── The relationship between oil prices and exchange rates Revisiting theory and evidence.pdf
│   │   └── Volatility spillover between oil prices and main exchange rates Evidence from a DCC-GARCH-connectedness approach.pdf
│   ├── not_categorised
│   │   ├── CAN EXCHANGE RATES FORECAST COMMODITY PRICES.pdf
│   │   ├── Exchange Rates, Energy Prices, Unemployment, Money, and Inflation A Further Test.pdf
│   │   ├── Regime Shifts in the Behaviour of International Currency and Equity Markets A Markov‐Switching Analysis.pdf
│   │   └── Regime-specific exchange rate predictability.pdf
│   ├── not_relevant
│   │   ├── Pricing energy futures options The role of seasonality and liquidity.pdf
│   │   └── The impact of commodity benchmarks on derivatives markets The case of the dated Brent assessment and Brent futures.pdf
│   └── other
│       ├──  Contango with me Covid-19 and the U.S. Crude Oil Market.pdf
│       ├── Battle of the benchmarks Brent Crude Oil and Western Texas Intermediate.pdf
│       ├── Commodity derivatives in 2023.pdf
│       ├── Impact_of_FX_fluctuations_on_Stock_market_indices_1752850073 2.pdf
│       ├── InterimStaffReportNYMEX_WTICrudeOil.pdf
│       ├── Oil Prices and Inflation Expectations - Is there a Link.pdf
│       ├── Real Effective Exchange Rates - The BIS methodology.pdf
│       ├── Rise in energy prices, the exchange rate of the euro and Germany’s price competitiveness.pdf
│       ├── The Nature and Drivers of Commodity Price Cycles.pdf
│       └── The relationship between crude oil and natural gas prices the role of the exchange rate.pdf
├── notebooks <- folder containing quick explorations and tests using Jupyter Notebooks
│   ├── 01_eda.ipynb
│   └── experiments.ipynb
├── pyproject.toml
├── reports <- folder contining all report types accompanying the project
│   ├── PPTX_WiSo_Template.pptx
│   ├── figures
│   │   ├── PET_PRI_FUT_S1_D.xls
│   │   ├── cftc_Futures-and-Options Combined Reports_1995-01-01_to_2025-01-01.csv
│   │   ├── deviations_of_usd_spotrates_from_ppp_values.html
│   │   ├── deviations_of_usd_spotrates_from_ppp_values.pdf
│   │   ├── eu_area_cpi_components_contribution.html
│   │   ├── eu_area_cpi_components_contribution.pdf
│   │   ├── figures.ipynb
│   │   ├── gas_consumption_production_combined_graph.html
│   │   ├── gas_consumption_production_combined_graph.pdf
│   │   ├── oil_consumption_production_combined_graph.html
│   │   ├── oil_consumption_production_combined_graph.pdf
│   │   ├── oil_gas_usd_index.html
│   │   ├── oil_gas_usd_index.pdf
│   │   ├── oil_gas_usd_index_lin_correlation_30_pct_returns.html
│   │   ├── oil_gas_usd_index_lin_correlation_30_pct_returns.pdf
│   │   ├── open_interest_oil_gas_combined_graph.html
│   │   ├── open_interest_oil_gas_combined_graph.pdf
│   │   ├── us_cpi_components_contribution.html
│   │   └── us_cpi_components_contribution.pdf
│   ├── logs
│   ├── poster <- raw unused poster templates
│   │   ├── poster_CAU.ppt
│   │   ├── poster_Math.ppt
│   │   └── poster_WISO.ppt
│   ├── presentation <- raw unused presentation templates
│   │   ├── PPTX_CAU_Template.pptx
│   │   ├── PPTX_Math_Template.pptx
│   │   └── PPTX_WiSo_Template.pptx
│   ├── presentation_latex_version <- The LaTex  presentation
│   │   ├── chapters <- single chapter files pulled together in main.tex
│   │   │   ├── appendix.tex
│   │   │   ├── chapter-00.tex
│   │   │   ├── chapter-01.tex
│   │   │   ├── chapter-02.tex
│   │   │   ├── chapter-03.tex
│   │   │   ├── chapter-04.tex
│   │   │   ├── chapter-05.tex
│   │   │   ├── chapter-06.tex
│   │   │   ├── chapter-07.tex
│   │   │   └── references.tex
│   │   ├── code
│   │   ├── figures <- all figures used in the presentation
│   │   │   ├── deviations_of_usd_spotrates_from_ppp_values.pdf
│   │   │   ├── eu_area_cpi_components_contribution.pdf
│   │   │   ├── gas_consumption_production_combined_graph.pdf
│   │   │   ├── literature_systematic_overview.pdf
│   │   │   ├── oil_consumption_production_combined_graph.pdf
│   │   │   ├── oil_gas_usd_index.pdf
│   │   │   ├── open_interest_oil_gas_combined_graph.pdf
│   │   │   ├── siegel.pdf
│   │   │   ├── us_cpi_components_contribution.pdf
│   │   │   └── wiso_logo.png
│   │   ├── main.pdf <- the presentation document itself
│   │   └── main.tex
│   └── tables
├── requirements <- requirements that need to be fullfilled running the project (code)
│   ├── environment.yml
│   └── requirements.txt <- main packages used
├── scripts
│   ├── evaluate_results.py
│   ├── run_backtest.py
│   └── run_training.py
├── seminar <- the seminar paper itself as LaTex project
│   ├── bibliography.bib
│   ├── chapters
│   │   ├── 00-abstract.tex
│   │   ├── 01-introduction.tex
│   │   ├── 02-literature-review.tex
│   │   ├── 03-methodology.tex
│   │   ├── 04-data.tex
│   │   ├── 05-experiments.tex
│   │   ├── 06-results.tex
│   │   ├── 07-discussion.tex
│   │   ├── 08-conclusion.tex
│   │   ├── 09-references.aux
│   │   ├── 09-references.tex
│   │   ├── 10-appendix.tex
│   │   └── 11-affirmation.tex
│   ├── committee_members.aux
│   ├── figures
│   ├── logos
│   │   ├── Logo_Kiel_University.png
│   │   ├── cau.png
│   │   ├── cau_kiel_maths.png
│   │   ├── cau_kiel_wiso.pdf
│   │   ├── cau_kiel_wiso.png
│   │   ├── cau_logo.pdf
│   │   ├── cau_logo.png
│   │   ├── vattenfall.pdf
│   │   └── vattenfall.png
│   ├── main.pdf
│   ├── main.tex
│   ├── preamble.tex
│   ├── seminar.sty
│   ├── tables
│   └── titlepage.tex
├── setup.cfg
├── setup.py
└── src <- the seminar code
    ├── seminar_code
    │   ├── backtesting
    │   │   ├── metrics.py
    │   │   └── portfolio.py
    │   ├── data_graphing
    │   │   ├── __pycache__
    │   │   │   ├── config.cpython-313.pyc
    │   │   │   └── data_grapher.cpython-313.pyc
    │   │   ├── config.py
    │   │   ├── data_grapher.py
    │   │   └── seminar_graphs.py
    │   ├── data_loading
    │   │   ├── __pycache__
    │   │   │   ├── ORDERBOOK_GENERATION.cpython-313.pyc
    │   │   │   ├── config.cpython-313.pyc
    │   │   │   └── data_loader.cpython-313.pyc
    │   │   ├── config.py
    │   │   ├── data_loader.py
    │   │   └── seminar_data.py
    │   ├── data_processing
    │   │   └── feature_engineering.py
    │   ├── model
    │   │   ├── architecture.py
    │   │   ├── model.py
    │   │   └── training.py
    │   ├── models
    │   │   ├── AgglomerativeClustering_model.pkl
    │   │   ├── AgglomerativeClustering_model.txt
    │   │   ├── DBSCAN_model.pkl
    │   │   ├── DBSCAN_model.txt
    │   │   ├── KMeans_model.pkl
    │   │   ├── KMeans_model.txt
    │   │   ├── MeanShift_model.pkl
    │   │   ├── MeanShift_model.txt
    │   │   ├── SpectralClustering_model.pkl
    │   │   ├── SpectralClustering_model.txt
    │   │   └── all_models_comparison.xlsx
    │   └── utils
    │       ├── config.py
    │       └── helpers.py
    └── tests
        ├── __init__.py
        ├── test_backtest.py
        ├── test_data.py
        ├── test_features.py
        ├── test_models.py
        └── test_strategy.py
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
@software{Fella_Hennings_Seminar,
  author = {Fella, Josef; Hennings, Robert},
  title = {Seminar},
  year = {2025},
  version = {1.0.0},
  url = {
author = {Fella, Josef; Hennings, Robert},
doi = {10.1234/example-doi},
license = {MIT},
title = {{Seminar}},
url = {https://github.com/RobertHennings/Seminar},
version = {1.0.0}
}
```
### APA
```apa
Fella, J.; Hennings, R. Seminar (Version 1.0.0) [Computer software]. https://doi.org/10.1234/example-doi
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