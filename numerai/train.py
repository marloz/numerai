import time
from copy import deepcopy
from functools import partial

import numerai.splitter
import numerai.model_dispatcher
import numerai.scoring

import numpy as np
import pandas as pd
from skopt import gp_minimize


# would like to use BayesSearchCV once this is solved: https://github.com/scikit-optimize/scikit-optimize/pull/988
class CustomBayesSearch:

    def __init__(self, model_name, n_iter=10, scoring='sharpe', cv=None,
                 n_splits=3, groups=None, n_jobs=-1, refit=True, random_state=42, return_train_score=True):
        self.estimator = numerai.model_dispatcher.models[model_name]['estimator']
        self.search_spaces = numerai.model_dispatcher.models[model_name]['search_spaces']
        self.n_iter = n_iter
        self.scoring = numerai.scoring.metrics[scoring]
        self.cv = numerai.splitter.CustomSplitter(n_splits) if cv is None else cv
        self.groups = groups
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.refit = refit
        self.random_state = random_state
        self.return_train_score = return_train_score
        self.train_scores, self.train_times = [], []
        self.best_estimator_ = None

    def __repr__(self):
        return (f'Estimator: {self.estimator}\n'
                f'Search spaces: {self.search_spaces}\n'
                f'Optimization iteraions: {self.n_iter}\n'
                f'Scoring function: {self.scoring}\n'
                f'Cross validation: {self.cv}\n'
                f'Number of CV splits: {self.n_splits}\n')

    def optimize(self, params, X, y):
        param_dict = dict(zip(list(self.search_spaces.keys()), params))
        estimator = deepcopy(
            self.estimator)  # Creating copy of estimator because some implementation don't allow setting params to already fitted model
        estimator.set_params(**param_dict)

        test_score, train_score, train_time = [], [], []
        groups = numerai.splitter.create_fold_groups(X, self.n_splits) if self.groups is None else self.groups
        for train_idx, test_idx in self.cv.split(X, y, groups):

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            start_time = time.time()
            estimator.fit(X_train, y_train)
            test_score.append(self.scoring(estimator, X_test, y_test))

            if self.return_train_score:
                train_score.append(self.scoring(estimator, X_train, y_train))

            train_time.append(round(time.time() - start_time, 2))

        self.train_times.append(np.mean(train_time))
        self.train_scores.append(np.mean(train_score))

        return -1 * np.mean(test_score)

    def fit(self, X, y, callback=None):
        optim_fn = partial(self.optimize, X=X, y=y)
        self.optimizer_results_ = gp_minimize(optim_fn,
                                              dimensions=list(self.search_spaces.values()),
                                              n_calls=self.n_iter,
                                              random_state=self.random_state,
                                              callback=callback,
                                              n_jobs=self.n_jobs,
                                              verbose=True)

        if self.refit:
            best_params = dict(zip(self.search_spaces.keys(), self.optimizer_results_.x))
            print(f'Fitting estimator with optimal parameters: \n{best_params}')
            self.best_estimator_ = self.estimator.set_params(**best_params)
            self.best_estimator_.fit(X, y, groups=self.groups)

        return self

    @property
    def cv_results_(self):
        res = self.optimizer_results_
        return (pd.DataFrame(res.x_iters, columns=self.search_spaces.keys())
                .assign(mean_test_score=-res.func_vals,
                        rank_test_score=lambda x: (-x['mean_test_score']).rank(),
                        mean_train_score=self.train_scores,
                        mean_train_time=self.train_times)
                .sort_values('rank_test_score'))
