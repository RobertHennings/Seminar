# Seminar
# Identifying (volatility-)regimes in the the EUR/USD spot exchange rate using clustering algorithms: An Oil and Gas Perspective on Parity Conditions.
Seminar in Applied Financial Economics: Applied Econometrics of FX Markets - Professor Reitz

[PDF Presentation]()

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
The seminar paper deals with the question, if alternative regime identification methods, including additional explanatory variables in the framework of clustering algorithms, can help to better identify (volatility driven) market regimes in the EUR/USD spot exchange rate. Especially including different variations of energy commodity prices (and their rolling volatility) is tested for an improved regime identification performance, with variations of the Markov-Switching Model
as benchmark. Finally, after having separated the Time-series of the EUR/USD spot exchange rate into different (volatility-) regimes, the standard UIP relationship is tested and compared among the regimes.

Main considered energy commodity prices are (USA):

1. [WTI Crue Oil - Cushing, Oklahoma - Spot Prices - Daily](https://fred.stlouisfed.org/series/DCOILWTICO)
2. [Natural Gas - Henry Hub - Spot Prices - Daily](https://fred.stlouisfed.org/series/DHHNGSP)

Data sources are primarily:

1. [FRED](https://fred.stlouisfed.org)

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
├── notebooks <- folder containing quick explorations and tests using Jupyter Notebooks
│   ├── 01_eda.ipynb
│   └── experiments.ipynb
├── pyproject.toml
├── reports <- folder contining all report types accompanying the project
│   ├── PPTX_WiSo_Template.pptx
│   ├── figures
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