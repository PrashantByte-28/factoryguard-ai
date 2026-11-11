import joblib
import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from src.data_loading import load_raw_data
from src.labeling import add_failure_label
from src.features import add_rolling_features, get_feature_columns
from src.split import time_split


# Building the dataset
data = load_raw_data()
df = add_failure_label(data["telemetry"], data["failures"])
df = add_rolling_features(df)
feature_cols = get_feature_columns(df)
train, val, test = time_split(df)


X_train, y_train = train[feature_cols], train["label"]
X_val, y_val = val[feature_cols], val["label"]


# imbalance handling: compute class weights
ratio = (y_train == 0).sum() / (y_train == 1).sum()


# Defining the model
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=ratio,
    subsample=0.8,
    subsample_freq=1,
    subsample_tree=0,
    colsample_bytree=0.8,
    metric="average_precision",
    random_state=42,
    verbose=-1,
)

# Model fitting
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

# Prediction probabilities and validation metrics
p_val = model.predict_proba(X_val)[:, 1]
print("\nVal ROC-AUC:", round(roc_auc_score(y_val, p_val), 4))
print("Val PR-AUC:", round(average_precision_score(y_val, p_val), 4), f"(random guessing = {y_val.mean():.4f})")


# Threshold tuning for precision-recall tradeoff
prec, rec, thr = precision_recall_curve(y_val, p_val)
idx = np.where(rec[:-1] >= 0.80)[0][-1]
print(f"\n Threshold {thr[idx]:.3f} -> recall {rec[idx]:.2f}, precision {prec[idx]:.2f}")

# Which feature did the model rely on?
imp = sorted(zip(model.feature_importances_, feature_cols), reverse=True)[:10]
print("\n Top 10 features:")
for score, name in imp:
    print(f"  {name:25s} {score}")

# Save the model 
joblib.dump({"model": model, "features": feature_cols, "threshold": float(thr[idx])}, "models/lgbm_baseline.joblib")