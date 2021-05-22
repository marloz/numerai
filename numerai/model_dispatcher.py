from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer
from catboost import CatBoostClassifier

# Experiments revealed that increasing n_estimators generally improves performance, as expected,
# max_samples parameter also seemed to perform better when set to 1.0 (n_samples)
# Max features didn't seem to have an effect at all
# So the only important feature to tune was min_samples_leaf or depth of trees,
# Here around 75 turned out to be optimal
models = {
    'random_forest': {
        'estimator': RandomForestClassifier(random_state=42,
                                            class_weight='balanced',
                                            n_jobs=-1,
                                            n_estimators=1_00,
                                            max_samples=10_000,
                                            max_features='sqrt'),
        'search_spaces': {
            'min_samples_leaf': Integer(10, 100),
            # 'max_features': Real(.05, .99),
            # 'n_estimators': Integer(100, 200),
            # 'max_samples': Integer(10_000, 100_000)
        }
    },
    # Try tu tune learning_rate, depth and l2_leaf_reg
    # Boosting type and boostrap_type are set for optimal speed
    # subsample and rsm work similar to random forest's max_features and max_samples,
    # If trained on GPU probably should just be set defaults (setting <1 speeds up training)
    'catboost': {
        'estimator': CatBoostClassifier(iterations=1_00,
                                        loss_function='MultiCLass',
                                        early_stopping_rounds=1,
                                        learning_rate=.5,
                                        depth=5,
                                        l2_leaf_reg=5,
                                        boosting_type='Plain',
                                        bootstrap_type='Bernoulli',
                                        subsample=.2,
                                        random_seed=42,
                                        auto_class_weights='Balanced',
                                        task_type='CPU'  #Change if GPU is available
                                        # rsm=.2
                                        ),
        'search_spaces': {
            'rsm': Real(0.1, .2)
        }
    }
}
