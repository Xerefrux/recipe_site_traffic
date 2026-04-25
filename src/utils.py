import os
import sys

import dill  # dill handles more complex object graphs than plain pickle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Serializes any Python object to disk using dill.
    Called after training to persist both preprocessor.pkl and model.pkl
    so the inference pipeline can load them without retraining anything.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Deserializes a previously saved object back into memory.
    Called by PredictPipeline at inference time to load model.pkl
    and preprocessor.pkl from the artifacts/ directory.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains every model in the `models` dict, tunes hyperparameters via
    GridSearchCV (5-fold cross-validation), and returns a results dict
    mapping model name -> test-set precision on the High Traffic class.

    WHY PRECISION AND NOT ACCURACY OR R²?
    This is a binary classification task (High vs. Not High), so R² is
    inappropriate. Accuracy would also be misleading here because the
    classes are imbalanced (~60% High). Most importantly, the business
    goal is to MINIMISE false positives — cases where we predict High but
    the recipe is actually unpopular. Precision = TP / (TP + FP) directly
    measures this: "of the recipes we call High, what fraction actually are?"
    That is exactly what the product manager cares about.
    """
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # GridSearchCV tries every combination in the param grid with
            # 5-fold cross-validation, scoring each by precision so that
            # hyperparameter selection directly optimises the business metric.
            gs = GridSearchCV(model, para, cv=5, scoring="precision", n_jobs=-1)
            gs.fit(X_train, y_train)

            # Apply the best hyperparameters found and refit on the full
            # training set (GridSearchCV's refit=True does this automatically,
            # but we explicitly set_params here to keep the model object
            # consistent for later use).
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            # pos_label=1 means "High Traffic is the positive class".
            # zero_division=0 prevents warnings if a model never predicts High.
            test_precision = precision_score(
                y_test, y_test_pred, pos_label=1, zero_division=0
            )

            report[list(models.keys())[i]] = test_precision
            logging.info(f"{list(models.keys())[i]} → test precision: {test_precision:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)