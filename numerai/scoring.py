from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from scipy.stats import skew, kurtosis


def spearman(y_true, y_pred):
    """ Calculate Spearman correlation """
    corr = spearmanr(y_true, y_pred, axis=0)[0]
    return 0 if np.isnan(corr) else corr


def era_correlations(y_true, y_pred):
    y_preds = pd.Series(np.ravel(y_pred), index=y_true.index, name='y_pred')
    return (pd.concat([y_true, y_preds], axis=1)
            .reset_index()
            .groupby('era')
            .apply(lambda x: spearman(x['target'], x['y_pred'])))


def sharpe(y_true, y_pred):
    """ Calculates sharpe ratio (mean/std of spearman correlation) on each era,
    assuming that `era` is part of index"""
    corr = era_correlations(y_true, y_pred)
    return corr.mean() / corr.std()


def max_drawdown(y_true, y_pred):
    corr = era_correlations(y_true, y_pred)
    rolling_max = (corr + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (corr + 1).cumprod()
    return (rolling_max - daily_value).max()


def feature_exposures(X, y_pred):
    exposures = [spearman(X[col], np.ravel(y_pred))
                 for col in X.columns]
    max_exposure = np.max(np.abs(exposures))
    return np.std(exposures), max_exposure


def sortino_ratio(y_true, y_pred):
    corr = era_correlations(y_true, y_pred)
    return np.mean(corr) / np.sqrt((np.sum(np.minimum(0, corr) ** 2) / (len(corr) - 1)))


def autocorr_penalty(x):
    n = len(x)
    ar1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    p = np.abs(ar1)
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


def smart_sharpe(x):
    return np.mean(x) / (np.std(x, ddof=1) * autocorr_penalty(x))


def adj_sharpe(y_true, y_pred):
    corr = era_correlations(y_true, y_pred)
    sharpe = smart_sharpe(corr)
    return sharpe * (1 + ((skew(corr) / 6) * sharpe) - ((kurtosis(corr) - 3) / 24) * (sharpe ** 2))


scoring = dict(spearman=make_scorer(spearman),
               sharpe=make_scorer(sharpe),
               max_drawdown=make_scorer(max_drawdown),
               adj_sharpe=make_scorer(adj_sharpe))


def create_fold_groups(df, n_splits):
    zeroes = np.zeros(df.shape[0])
    zeroes_split = np.array_split(zeroes, n_splits)
    groups_list = [np.full(len(arr, ), val)
                   for val, arr in enumerate(zeroes_split)]
    return np.hstack(groups_list)
