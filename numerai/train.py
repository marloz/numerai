import time
from copy import deepcopy
from functools import partial

from numerai import model_dispatcher
from numerai import scoring as score

import numpy as np
import pandas as pd
import skopt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_validate


# would like to use BayesSearchCV once this is solved: https://github.com/scikit-optimize/scikit-optimize/pull/988
class CustomBayesSearch:

    def __init__(self, model_name, n_iter=10, scoring=None, cv=None,
                 n_splits=3, groups=None, n_jobs=-1, refit='adj_sharpe', random_state=42, return_train_score=True,
                 optimizer='gp_minimize', callback=None):
        self.estimator = model_dispatcher.models[model_name]['estimator']
        self.search_spaces = model_dispatcher.models[model_name]['search_spaces']
        self.n_iter = n_iter
        self.scoring = score.scoring if scoring is None else scoring
        self.cv = LeaveOneGroupOut() if cv is None else cv
        self.groups = groups
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.refit = refit
        self.random_state = random_state
        self.return_train_score = return_train_score
        self.train_scores, self.train_times = [], []
        self.best_estimator_ = None
        self.optimizer = optimizer
        self._cv_results = pd.DataFrame()
        self.callback = skopt.callbacks.DeltaYStopper(delta=.01)

    def __repr__(self):
        return (f'Estimator: {self.estimator, self.estimator.get_params()}\n'
                f'Search spaces: {self.search_spaces}\n'
                f'Optimization iteraions: {self.n_iter}\n'
                f'Scoring function: {self.refit}\n'
                f'Metrics calculated: {self.scoring}\n'
                f'Cross validation: {self.cv}\n'
                f'Number of CV splits: {self.n_splits}\n'
                f'Using optimizer: {self.optimizer}\n')

    def objective(self, params, X, y):
        param_dict = dict(zip(list(self.search_spaces.keys()), params))
        estimator = deepcopy(self.estimator)
        estimator.set_params(**param_dict)

        scores = cross_validate(estimator, X, y, groups=self.groups,
                                scoring=self.scoring, cv=self.cv,
                                n_jobs=self.n_jobs, return_train_score=self.return_train_score)
        scores = pd.DataFrame(scores).apply('mean')
        self._cv_results = self._cv_results.append(scores, ignore_index=True)
        return -1 * scores[f'test_{self.refit}']

    def fit(self, X, y):
        self.groups = score.create_fold_groups(X, self.n_splits) if self.groups is None else self.groups
        objective = partial(self.objective, X=X, y=y)
        self.optimizer_results_ = getattr(skopt, self.optimizer)(objective,
                                                                 dimensions=list(self.search_spaces.values()),
                                                                 n_calls=self.n_iter,
                                                                 random_state=self.random_state,
                                                                 callback=self.callback,
                                                                 n_jobs=self.n_jobs,
                                                                 verbose=True)

        if self.refit:
            best_params = dict(zip(self.search_spaces.keys(), self.optimizer_results_.x))
            print(f'Fitting estimator with optimal parameters: \n{best_params}')
            self.best_estimator_ = self.estimator.set_params(**best_params)
            self.best_estimator_.fit(X, y)

        return self

    @property
    def cv_results_(self):
        res = self.optimizer_results_
        res = pd.DataFrame(res.x_iters, columns=self.search_spaces.keys())
        return (pd.concat([res, self._cv_results], axis=1)
                .assign(rank_test_score=lambda x: (-x[f'test_{self.refit}']).rank())
                .sort_values('rank_test_score'))
