import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    # The business requirement: correctly identify High Traffic recipes at least 80% of the time (i.e. precision >= 0.80).
    precision_threshold: float = 0.80


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(
                    random_state=42, max_iter=1000
                ),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(
                    random_state=42, eval_metric="logloss", verbosity=0
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }

            # Hyperparameter grids:
            param = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"],
                    "class_weight": [None, "balanced"],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 8, None],
                    "min_samples_leaf": [1, 5],
                    "class_weight": [None, "balanced"],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.5, 1.0, 1.5],
                },
            }

            logging.info("Starting automated model evaluation sweep.")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param,
            )

            # Select the model with the highest test-set precision.
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.precision_threshold:
                logging.warning(
                    f"Best model precision ({best_model_score:.4f}) is below "
                    f"the 80% business threshold. Consider collecting more "
                    f"data or engineering additional features."
                )

            logging.info(
                f"Best model: {best_model_name} | "
                f"Test Precision: {best_model_score:.4f}"
            )

            # Print a full comparison table and detailed report to the console
            y_pred = best_model.predict(X_test)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            acc  = accuracy_score(y_test, y_pred)

            print("\n" + "=" * 60)
            print("MODEL COMPARISON — Test Precision (High Traffic class)")
            print("=" * 60)
            for name, score in sorted(model_report.items(), key=lambda x: -x[1]):
                flag = " ← WINNER" if name == best_model_name else ""
                print(f"  {name:<28} {score * 100:.1f}%{flag}")
            print("=" * 60)
            print(f"\nWinning model : {best_model_name}")
            print(f"Accuracy      : {acc:.4f}  ({acc * 100:.1f}%)")
            print(f"Precision     : {prec:.4f}  ({prec * 100:.1f}%)")
            print(f"Recall        : {rec:.4f}  ({rec * 100:.1f}%)")
            print(f"Business goal : >= 80.0% precision")
            print(f"Goal met?     : {'YES' if prec >= 0.80 else 'NO'}")
            print("\nFull classification report:")
            print(
                classification_report(
                    y_test, y_pred, target_names=["Not High", "High"]
                )
            )

            # Save the winning model so the inference pipeline can load it at prediction time without retraining.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return prec

        except Exception as e:
            raise CustomException(e, sys)