import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


def regression_evaluation(pred_test, y_test):

    # The mean_squared_error function computes mean square error, a risk metric corresponding
    # to the expected value of the squared (quadratic) error or loss.
    mse = mean_squared_error(y_test, pred_test)

    # The mean_absolute_error function computes mean absolute error, a risk metric corresponding
    # to the expected value of the absolute error loss or l1-norm loss.
    mae = mean_absolute_error(y_test, pred_test)

    # coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variance in
    # the dependent variable that is predictable from the independent variable(s).
    r2 = r2_score(y_test, pred_test)

    # In statistics, explained variation measures the proportion to which a mathematical model accounts for the
    # variation (dispersion) of a given data set. Often, variation is quantified as variance; then, the more specific
    # term explained variance can be used.
    # The complementary part of the total variation is called unexplained or residual variation.
    evs = explained_variance_score(y_test, pred_test)

    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'explained_variance_score': evs,
        'r2_score': r2
    }