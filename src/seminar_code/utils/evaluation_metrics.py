import pandas as pd
import numpy as np
# The regime classification measure (RCM) of Ang and Bekaert (2002) is used
# to determine the accuracy of the Markov-switching models. This statistic is computed using the following formula:
# The RCM is computed as the average of the product of smoothed probabilities p~; where S is the number of
# regimes (states, S). The switching variable follows a Bernoulli distribution and as a result, the RCM provides an
# estimate of the variance. The RCM statistic ranges be- tween 0 (perfect regime classification) and 100
# (failure to detect any re- gime classification) with lower values of the RCM preferable to higher values of the RCM.
# Thus to ensure significantly different regimes, it is important that a model's RCM is close to zero and its
# smoothed proba- bility indicator be close to 1.
def compute_rcm(S: int, smoothed_probs: pd.Series, threshold: float=0.5) -> float:
    """Compute the Regime Classification Measure (RCM) for a series of smoothed probabilities.

    Args:
        S (int): The number of regimes.
        smoothed_probs (pd.Series): The smoothed probabilities for the regimes.
        threshold (float, optional): The threshold to classify regimes. Defaults to 0.5.

    Examples:
        from model import compute_rcm
        rcm = compute_rcm(S=2, smoothed_probs=exchange_rates_df["msm_regime"])

    Returns:
        float: The RCM value.
    """
    regime_classification = (smoothed_probs >= threshold).astype(int)
    rcm = 100 * S**2 * (1 - (1 / len(smoothed_probs)) * np.sum(regime_classification * smoothed_probs + (1 - regime_classification) * (1 - smoothed_probs)))
    return rcm