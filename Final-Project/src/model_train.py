from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# apply oversmapling
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(sampling_strategy=0.2, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# define parameter grid
param_dist = {
    'n_estimators': [300],  # fixed to reduce combinations
    'max_depth': [9],
    'learning_rate': [0.05],  # fixed value that's working well
    'subsample': [0.8],  # keep consistent sampling
    'colsample_bytree': [0.8],  # fixed for speed
    'gamma': [3, 4],
    'scale_pos_weight': [1.4, 1.5, 1.6],
    'min_child_weight': [1],
}

# # use best parameters from earlier tuning
# best_params = {
#     'subsample': 0.8,
#     'scale_pos_weight': 1.5,
#     'n_estimators': 300,
#     'min_child_weight': 1,
#     'max_depth': 9,
#     'learning_rate': 0.05,
#     'gamma': 3,
#     'colsample_bytree': 0.8
# }

# initialize classifier with base config
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

# define F2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)
1
# random grid search setup
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=15,  # adjust depending on how long you're willing to wait
    scoring=f2_scorer,  # or a custom scorer
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# # initialize final model
# final_model = XGBClassifier(
#     **best_params,
#     objective='binary:logistic',
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42,
#     early_stopping_rounds=10  # add early stopping here
# )

# smaller sample for speed
sample_size = int(0.5 * len(X_train_resampled))
X_sample = X_train_resampled[:sample_size]
y_sample = y_train_resampled[:sample_size]

# run grid search
random_search.fit(X_sample, y_sample)

# best model and parameters
print("Best Parameters:\n", random_search.best_params_)
best_model = random_search.best_estimator_

# train XGBoost Model, replacing xgb.train with XGBClassifier
best_model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# save the trained best model
import joblib
joblib.dump(best_model, "best_xgb_model.pkl")
