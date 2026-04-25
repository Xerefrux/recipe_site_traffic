import os
import sys

import dill  # dill handles more complex object graphs than plain pickle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5, scoring="precision", n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            # zero_division=0 prevents warnings if a model never predicts High.
            test_precision = precision_score(
                y_test, y_test_pred, pos_label=1, zero_division=0
            )

            report[list(models.keys())[i]] = test_precision
            logging.info(f"{list(models.keys())[i]} → test precision: {test_precision:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)