import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Stateless prediction service. Every call to predict() loads the
    artifacts fresh from disk.

    WHY STATELESS?
    Flask may spin up multiple worker threads to handle concurrent requests.
    A stateless class (no instance-level model or preprocessor attribute)
    means there is no shared mutable state between threads — eliminating an
    entire class of concurrency bugs.

    THE CRITICAL CORRECTNESS PROPERTY:
    This pipeline calls preprocessor.transform(), NOT fit_transform().
    Using fit_transform() here would recompute imputation medians and scaler
    parameters on the single new input row — producing completely different
    transformation parameters than those used during training, and therefore
    completely wrong feature values. Loading the fitted preprocessor and
    calling transform() ensures the new input goes through the exact same
    arithmetic as the training data did.
    """

    def predict(self, features: pd.DataFrame):
        """
        Parameters
        ----------
        features : pd.DataFrame
            A single-row DataFrame with the same column names as the
            training feature set, BEFORE any preprocessing.

        Returns
        -------
        predictions : np.ndarray  — class labels (0 = Not High, 1 = High)
        probabilities : np.ndarray — P(High Traffic) for each row
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor artifacts.")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # transform() — not fit_transform() — see class docstring above.
            data_scaled = preprocessor.transform(features)

            predictions = model.predict(data_scaled)
            # predict_proba() returns [P(Not High), P(High)]; we want column 1.
            probabilities = model.predict_proba(data_scaled)[:, 1]

            return predictions, probabilities

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A data-transfer object that bridges the web form and the prediction
    pipeline. Each attribute corresponds to one form field.

    The as_data_frame() method serializes the fields into a DataFrame with
    the EXACT column names the ColumnTransformer was fitted on. If the names
    or order differ, sklearn will silently assign values to the wrong
    transformer — a dangerous bug that this explicit mapping prevents.
    """

    VALID_CATEGORIES = [
        "Chicken", "Beverages", "Breakfast", "Dessert",
        "Lunch/Snacks", "Meat", "One Dish Meal", "Pork",
        "Potato", "Vegetable",
    ]

    def __init__(
        self,
        calories: float,
        carbohydrate: float,
        sugar: float,
        protein: float,
        category: str,
        servings: int,
    ):
        self.calories = calories
        self.carbohydrate = carbohydrate
        self.sugar = sugar
        self.protein = protein
        self.category = category
        self.servings = servings

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Returns a single-row DataFrame whose column order matches the
        feature_columns list defined in DataTransformation, ensuring the
        ColumnTransformer routes each value to the correct sub-pipeline.
        """
        try:
            custom_data_input_dict = {
                "calories":     [self.calories],
                "carbohydrate": [self.carbohydrate],
                "sugar":        [self.sugar],
                "protein":      [self.protein],
                "category":     [self.category],
                "servings":     [self.servings],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)