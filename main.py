import decouple
import numerapi
import os

from numerai.data import DataGetter, PredictionSubmitter
from numerai.splitter import create_fold_groups, CustomSplitter
from numerai.scoring import sharpe
from numerai.train import CustomBayesSearch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from skopt.space import Real, Categorical, Integer

if __name__ == '__main__':
    # Get user secrets to use numerapi
    PUBLIC_ID, SECRET_KEY = decouple.config('PUBLIC_ID'), decouple.config('SECRET_KEY')
    napi = numerapi.NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY, verbosity='info')

    # Get locally downloaded data path, if already available,
    # otherwise newest round data will be downloaded
    cwd = os.getcwd()
    load_from = [os.path.join(cwd, path) for path in os.listdir(cwd)
                 if 'numerai_dataset' in path and '.zip' not in path]
    load_from = load_from[0] if len(load_from) > 1 else None

    # Load training data
    data_getter = DataGetter(napi)
    data = data_getter.get_data(load_from=load_from,
                                training_or_tournament='training')

    # Split into features and target
    feature_cols = [col for col in data.columns if 'feature' in col]
    X, y = data[feature_cols], data.target

    # Create kfold splitter
    N_SPLITS = 3
    groups = create_fold_groups(X, n_splits=N_SPLITS)
    kfold = CustomSplitter(n_splits=N_SPLITS)

    # Create estimator and metrics
    estimator = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=100, class_weight='balanced')
    scorer = make_scorer(sharpe)

    # Define parameter search space
    search_spaces = {
        'max_features': Real(.05, .5),
        'max_samples': Real(.1, .99),
        'min_samples_leaf': Integer(1, 100),
    }

    opt = CustomBayesSearch(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=1_00,
        scoring=scorer,
        cv=kfold)

    opt.fit(X, y, groups)

    submitter = PredictionSubmitter(napi=napi,
                                    model=opt.best_estimator_,
                                    napi_user='mloz')

    submitter.submit()
