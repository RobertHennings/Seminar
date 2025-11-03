import logging
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import sklearn

"""
This file contains the main evaluation metrics used for optimising and fitting the main model classes,
like the clustering algorithms that use the silhouette score as evaluation metric and the Markov-Switching
models, that use the RCM metric.
"""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_rcm(
    S: int,
    smoothed_probs: pd.Series,
    threshold: float=0.5
    ) -> float:
    """Compute the Regime Classification Measure (RCM) for a series of smoothed probabilities.
       The regime classification measure (RCM) of Ang and Bekaert (2002) is used
       to determine the accuracy of the Markov-switching models. This statistic is computed using the following formula:
       The RCM is computed as the average of the product of smoothed probabilities p~; where S is the number of
       regimes (states, S). The switching variable follows a Bernoulli distribution and as a result, the RCM provides an
       estimate of the variance. The RCM statistic ranges between 0 (perfect regime classification) and 100
       (failure to detect any regime classification) with lower values of the RCM preferable to higher values of the RCM.
       Thus to ensure significantly different regimes, it is important that a model's RCM is close to zero and its
       smoothed probability indicator be close to 1.

       Source: Ang, A., & Bekaert, G. (2002). Regime switches in interest rates
       Link: https://www.tandfonline.com/doi/abs/10.1198/073500102317351930
       See: Page 15, equations 40-41
    Args:
        S (int): The number of regimes.
        smoothed_probs (pd.Series): The smoothed probabilities for the regimes.
        threshold (float, optional): The threshold to classify regimes. Defaults to 0.5.

    Examples:
        from evaluation_metrics import compute_rcm
        rcm = compute_rcm(S=2, smoothed_probs=exchange_rates_df["msm_regime"])

    Returns:
        float: The RCM value.
    """
    regime_classification = (smoothed_probs >= threshold).astype(int)
    rcm = 100 * S**2 * (1 - (1 / len(smoothed_probs)) * np.sum(regime_classification * smoothed_probs + (1 - regime_classification) * (1 - smoothed_probs)))
    return rcm


def silhouette_scorer(
    estimator: sklearn.base.BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series = None
    ) -> float:
    """Compute the silhouette score for a clustering estimator.

    Args:
        estimator (sklearn.base.BaseEstimator): The clustering estimator.
        X (pd.DataFrame): The input features.
        y (pd.Series, optional): The true labels. Defaults to None.

    Raises:
        AttributeError: If the estimator has no fit_predict/predict method.

    Returns:
        float: The silhouette score.
    """
    # get labels (prefer fit_predict for clustering)
    if hasattr(estimator, "fit_predict"):
        labels = estimator.fit_predict(X)

    # get labels (prefer fit_predict for clustering)
    if hasattr(estimator, "fit_predict"):
        labels = estimator.fit_predict(X)
    elif hasattr(estimator, "predict"):
        # some estimators require fit first
        estimator.fit(X)
        labels = estimator.predict(X)
    else:
        raise AttributeError("Estimator has no fit_predict/predict")
    # require at least 2 clusters
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)
