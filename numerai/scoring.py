from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def spearman(y_true, y_pred):
    """ Calculate Spearman correlation """
    corr = spearmanr(y_true, y_pred, axis=0)[0]
    return 0 if np.isnan(corr) else corr


def sharpe(y_true, y_pred):
    """ Calculates sharpe ratio (mean/std of spearman correlation) on each era,
    assuming that `era` is part of index"""
    y_preds = pd.Series(np.ravel(y_pred), index=y_true.index, name='y_pred')
    corr = (pd.concat([y_true, y_preds], axis=1)
            .reset_index()
            .groupby('era')
            .apply(lambda x: spearman(x['target'], x['y_pred'])))
    return corr.mean() / corr.std()


metrics = dict(spearman=make_scorer(spearman),
               sharpe=make_scorer(sharpe))