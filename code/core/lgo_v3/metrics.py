import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, brier_score_loss

#def regression_metrics(y_true, y_pred):
#    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_true))
#    return {"RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}

def classification_metrics(y_true, y_prob):
    y_prob = np.clip(np.nan_to_num(y_prob, nan=np.nanmean(y_true)), 0, 1)
    try: auroc = roc_auc_score(y_true, y_prob)
    except Exception: auroc = np.nan
    try: auprc = average_precision_score(y_true, y_prob)
    except Exception: auprc = np.nan
    try: brier = brier_score_loss(y_true, y_prob)
    except Exception: brier = np.nan
    return {"AUROC": auroc, "AUPRC": auprc, "Brier": brier}
