from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer


models = {
    'random_forest': {
        'estimator': RandomForestClassifier(random_state=42,
                                            class_weight='balanced',
                                            n_jobs=-1,
                                            n_estimators=100),
        'search_spaces': {
            'max_features': Real(.05, .5),
            'max_samples': Real(.1, .99),
            'min_samples_leaf': Integer(1, 100),
        }
    }
}
