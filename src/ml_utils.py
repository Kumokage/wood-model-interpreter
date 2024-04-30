import base64
import matplotlib.pyplot as plt
import shap
from io import BytesIO
from typing import Any
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel


class TrainedModel(BaseModel):
    model: Any
    mae: float 


def learn_xgboost(X, y):
    parameters = {
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.5, 1, 1.5, 2, 2.5,],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    model = XGBRegressor(learning_rate=0.02, n_estimators=600, nthread=1, seed=0)
    clf = GridSearchCV(
        model, 
        parameters, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        refit=True,
        n_jobs=-1)
    clf.fit(X, y)
    return TrainedModel(model=clf.best_estimator_, mae=clf.best_score_)


def explain(model, X) -> str:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, max_display=20, plot_size=[10, 6], show=False)
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encode = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return 'data:image/png;base64,{}'.format(encode)
